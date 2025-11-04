# File: examples/paderkan.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io

# Import the correct model
from tabkan import PadeRKAN

# --- 0. Setup and Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- Running TabKAN Comprehensive Example for PadeRKAN on {device.upper()} ---")

# --- Helper function for robust model copying ---
def deepcopy_model(model):
    """
    Creates a deep copy of a PyTorch model by saving and loading its state_dict.
    """
    # This now uses the config attributes we added to the PadeRKAN class
    new_model = model.__class__(
        layers=model.layers_config,
        orders1=model.orders1_config,
        orders2=model.orders2_config
    ).to(device)
    
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    new_model.load_state_dict(torch.load(buffer))
    
    if hasattr(model, 'is_pretrained'):
        new_model.is_pretrained = model.is_pretrained
    return new_model

# --- 1. Data Setup (Same as before) ---
print("\n--- [Step 1] Creating Synthetic Datasets ---")
X_source, y_source = make_classification(n_samples=5000, n_features=20, n_informative=10, n_redundant=5, n_classes=2, random_state=42)
X_target, y_target = make_classification(n_samples=500, n_features=15, n_informative=8, n_redundant=2, n_classes=2, random_state=123)
source_dim = X_source.shape[1]
target_dim = X_target.shape[1]
padding = np.zeros((X_target.shape[0], source_dim - target_dim))
X_target_padded = np.hstack((X_target, padding))
X_source, y_source = torch.tensor(X_source, dtype=torch.float32), torch.tensor(y_source, dtype=torch.long)
X_target, y_target = torch.tensor(X_target_padded, dtype=torch.float32), torch.tensor(y_target, dtype=torch.long)
def create_dataset_dict(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    return {
        "train_input": X_train.to(device), "train_label": y_train.to(device),
        "test_input": X_test.to(device), "test_label": y_test.to(device),
    }
source_dataset = create_dataset_dict(X_source, y_source)
target_dataset = create_dataset_dict(X_target, y_target)
print(f"Source Dataset: {source_dataset['train_input'].shape[0]} train samples, {source_dim} features.")
print(f"Target Dataset: {target_dataset['train_input'].shape[0]} train samples, {target_dim} features (padded to {source_dim}).")


# --- 2. Hyperparameter Tuning for PadeRKAN ---
print("\n--- [Step 2] Hyperparameter Tuning for PadeRKAN ---")
# PadeRKAN has two sets of orders: one for the numerator (orders1) and one for the denominator (orders2)
search_space = {
    "depth": {"type": "int", "low": 1, "high": 2},
    "neurons_layer_0": {"type": "int", "low": 16, "high": 32},
    "neurons_layer_1": {"type": "int", "low": 8, "high": 16},
    "orders1_layer_0": {"type": "int", "low": 2, "high": 4}, # Numerator orders
    "orders1_layer_1": {"type": "int", "low": 2, "high": 4},
    "orders2_layer_0": {"type": "int", "low": 2, "high": 4}, # Denominator orders
    "orders2_layer_1": {"type": "int", "low": 2, "high": 4},
    "lr": {"type": "float", "low": 1e-2, "high": 1.0, "log": True},
    "steps": {"type": "categorical", "choices": [20]}
}

# NOTE: The OptunaTuner might need to be adjusted to handle 'orders1' and 'orders2'
# If it fails, the fix is in tabkan/tuner.py to add 'orders1_layer_' and 'orders2_layer_' to its search logic.
# For now, let's assume it works or we fix it as needed.

best_params = PadeRKAN.tune(
    model_class=PadeRKAN,
    dataset=source_dataset,
    search_space=search_space,
    n_trials=5,
    device=device
)
print("Best hyperparameters found:", best_params)

# --- 3. Pre-training on Source Dataset ---
print("\n--- [Step 3] Pre-training Model on Source Data ---")
depth = best_params.pop("depth")
lr = best_params.pop("lr")
steps = best_params.pop("steps")
layers = [source_dim] + [best_params[f'neurons_layer_{i}'] for i in range(depth)] + [2]
# Get both sets of orders
orders1 = [best_params[f'orders1_layer_{i}'] for i in range(depth)]
orders2 = [best_params[f'orders2_layer_{i}'] for i in range(depth)]

pretrained_model = PadeRKAN(layers=layers, orders1=orders1, orders2=orders2).to(device)
pretrained_model.pretrain(
    source_dataset, 
    steps=steps, 
    loss_fn=nn.CrossEntropyLoss(),
    lr=lr
)

# --- 4. Interpretability: Analyze the Pre-trained Model ---
print("\n--- [Step 4] Analyzing Feature Importance from Pre-trained Model ---")
try:
    feature_importance = pretrained_model.get_feature_importance(layer_index=0)
    plt.figure(figsize=(12, 6))
    plt.bar(range(source_dim), feature_importance.cpu().detach().numpy())
    plt.title("Feature Importance from Pre-trained PadeRKAN")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance (L1 Norm of Weights)")
    plt.show()
except (AttributeError, NotImplementedError, IndexError) as e:
    print(f"Could not get feature importance for PadeRKAN: {e}")

# --- 5. Transfer Learning & Comparison ---
print("\n--- [Step 5] Fine-Tuning and Comparison ---")
def evaluate(model, dataset, model_name):
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(dataset['test_input']), dim=1)
        accuracy = (preds == dataset['test_label']).float().mean().item()
        print(f"[{model_name}] Test Accuracy: {accuracy:.4f}")
    return accuracy

# Scenario A: Standard Fine-Tuning
model_ft_standard = deepcopy_model(pretrained_model)
model_ft_standard.finetune(
    target_dataset, 
    method='standard',
    steps=100,
    loss_fn=nn.CrossEntropyLoss(),
    lr=0.1
)
acc_ft_standard = evaluate(model_ft_standard, target_dataset, "Fine-tuned (Standard)")

# Scenario B: GRPO Fine-Tuning
try:
    model_ft_grpo = deepcopy_model(pretrained_model)
    model_ft_grpo.finetune(
        target_dataset,
        method='grpo',
        steps=200,
        lr=1e-3,
        batch=32
    )
    acc_ft_grpo = evaluate(model_ft_grpo, target_dataset, "Fine-tuned (GRPO)")
except (NotImplementedError, RuntimeError) as e:
    print(f"[Fine-tuned (GRPO)] SKIPPED: {e}")
    acc_ft_grpo = 0.0

# Scenario C: Training from Scratch on Target Data
print("\nTraining a model from scratch on the small target dataset...")
scratch_model = PadeRKAN(layers=layers, orders1=orders1, orders2=orders2).to(device)
scratch_model.fit(
    target_dataset, 
    steps=steps, 
    loss_fn=nn.CrossEntropyLoss(), 
    lr=lr
)
acc_scratch = evaluate(scratch_model, target_dataset, "Trained from Scratch")

# --- 6. Final Results ---
print("\n--- [Step 6] Final Performance Comparison ---")
results = {
    "Trained from Scratch": acc_scratch,
    "Fine-tuned (Standard)": acc_ft_standard,
    "Fine-tuned (GRPO)": acc_ft_grpo
}
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
print(results_df)

print("\n--- PadeRKAN Example Finished ---")