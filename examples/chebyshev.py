import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Assuming 'tabkan' is installed and in the Python path
from tabkan import ChebyshevKAN

# --- 0. Setup and Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- Running TabKAN Comprehensive Example on {device.upper()} ---")

# --- 1. Data Setup: Simulating a Transfer Learning Scenario ---
print("\n--- [Step 1] Creating Synthetic Datasets ---")

# SOURCE DATASET: A larger dataset with more features and samples
X_source, y_source = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=10, # 10 features are useful
    n_redundant=5,    # 5 are linear combinations of informative features
    n_classes=2,
    random_state=42
)

# TARGET DATASET: A smaller dataset with fewer samples and overlapping features
# The first 10 features are the same as the source's informative features
# The next 5 are new, totally different features
X_target, y_target = make_classification(
    n_samples=500,
    n_features=15,
    n_informative=8, # 5 of these overlap with source, 3 are new
    n_redundant=2,
    n_classes=2,
    random_state=123
)

# Align features for transfer learning (pad target with zeros to match source dim)
source_dim = X_source.shape[1]
target_dim = X_target.shape[1]
if target_dim < source_dim:
    padding = np.zeros((X_target.shape[0], source_dim - target_dim))
    X_target_padded = np.hstack((X_target, padding))
else:
    X_target_padded = X_target # Should not happen in this example

# Convert to Tensors
X_source, y_source = torch.tensor(X_source, dtype=torch.float32), torch.tensor(y_source, dtype=torch.long)
X_target, y_target = torch.tensor(X_target_padded, dtype=torch.float32), torch.tensor(y_target, dtype=torch.long)

# Create dataset dictionaries for TabKAN
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

# --- 2. Hyperparameter Tuning on the Source Task ---
print("\n--- [Step 2] Hyperparameter Tuning for ChebyshevKAN ---")

# Define a search space for Optuna
# Note: For a real run, n_trials should be higher (e.g., 50-100)
search_space = {
    "depth": {"type": "int", "low": 1, "high": 2},
    "neurons_layer_0": {"type": "int", "low": 16, "high": 32},
    "neurons_layer_1": {"type": "int", "low": 8, "high": 16},
    "orders_layer_0": {"type": "int", "low": 2, "high": 4},
    "orders_layer_1": {"type": "int", "low": 2, "high": 4},
    "lr": {"type": "float", "low": 1e-2, "high": 1.0, "log": True},
    "steps": {"type": "categorical", "choices": [20]}
}

# Create a dummy model instance to run the tuner
best_params = ChebyshevKAN.tune(
    model_class=ChebyshevKAN, # Pass the class itself
    dataset=source_dataset,
    search_space=search_space,
    n_trials=5, # Using a small number for a quick demo
    device=device
)
print("Best hyperparameters found:", best_params)

# --- 3. Pre-training on Source Dataset ---
print("\n--- [Step 3] Pre-training Model on Source Data ---")

# Build the model with the best parameters found
depth = best_params.pop("depth")
lr = best_params.pop("lr")
steps = best_params.pop("steps")
layers = [source_dim] + [best_params[f'neurons_layer_{i}'] for i in range(depth)] + [2]
orders = [best_params[f'orders_layer_{i}'] for i in range(depth)]

pretrained_model = ChebyshevKAN(layers=layers, orders=orders).to(device)

# Pre-train the model
pretrained_model.pretrain(
    source_dataset, 
    steps=steps, 
    loss_fn=nn.CrossEntropyLoss(),
    lr=lr
)

# --- 4. Interpretability: Analyze the Pre-trained Model ---
print("\n--- [Step 4] Analyzing Feature Importance from Pre-trained Model ---")

# Get feature importance from the first KAN layer
feature_importance = pretrained_model.get_feature_importance(layer_index=0)
print("Feature importances (first 10):", feature_importance.cpu().detach().numpy()[:10])

# Visualize the importance
plt.figure(figsize=(12, 6))
plt.bar(range(source_dim), feature_importance.cpu().detach().numpy())
plt.title("Feature Importance from Pre-trained ChebyshevKAN")
plt.xlabel("Feature Index")
plt.ylabel("Importance (L1 Norm of Coefficients)")
plt.axvspan(-0.5, 9.5, color='green', alpha=0.2, label='Informative Features')
plt.axvspan(9.5, 14.5, color='orange', alpha=0.2, label='Redundant Features')
plt.axvspan(14.5, 19.5, color='red', alpha=0.2, label='Noisy Features')
plt.legend()
plt.show()

# --- 5. Transfer Learning & Comparison ---
print("\n--- [Step 5] Fine-Tuning and Comparison ---")

# Helper function for evaluation
def evaluate(model, dataset, model_name):
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(dataset['test_input']), dim=1)
        accuracy = (preds == dataset['test_label']).float().mean().item()
        print(f"[{model_name}] Test Accuracy: {accuracy:.4f}")
    return accuracy

# Scenario A: Standard Fine-Tuning
from copy import deepcopy
model_ft_standard = deepcopy(pretrained_model)
model_ft_standard.finetune(
    target_dataset, 
    method='standard',
    steps=100,
    loss_fn=nn.CrossEntropyLoss(),
    lr=0.1
)
acc_ft_standard = evaluate(model_ft_standard, target_dataset, "Fine-tuned (Standard)")

# Scenario B: GRPO Fine-Tuning (Wrapped in try-except as it's a new feature)
try:
    model_ft_grpo = deepcopy(pretrained_model)
    model_ft_grpo.finetune(
        target_dataset,
        method='grpo',
        steps=200, # GRPO often benefits from more steps
        lr=1e-3,   # And a smaller learning rate
        batch=32
    )
    acc_ft_grpo = evaluate(model_ft_grpo, target_dataset, "Fine-tuned (GRPO)")
except NotImplementedError as e:
    print(f"[Fine-tuned (GRPO)] SKIPPED: {e}")
    acc_ft_grpo = 0.0

# Scenario C: Training from Scratch on Target Data
print("\nTraining a model from scratch on the small target dataset...")
scratch_model = ChebyshevKAN(layers=layers, orders=orders).to(device)
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

print("\n--- Example Finished ---")
