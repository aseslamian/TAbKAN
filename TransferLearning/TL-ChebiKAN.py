import math
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.optim import LBFGS
import torch.nn as nn

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
DATA1_PATH = "/home/data3/Ali/Code/TabMixer-review/DATA/Credit-g(CG)/TransferLearning/data1/data_processed.csv"
DATA2_PATH = "/home/data3/Ali/Code/TabMixer-review/DATA/Credit-g(CG)/TransferLearning/data2/data_processed.csv"

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 64

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Chebyshev polynomial tensors
        cheby = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:
            cheby[:, :, 1] = x
        for i in range(2, self.degree + 1):
            cheby[:, :, i] = 2 * x * cheby[:, :, i - 1].clone() - cheby[:, :, i - 2].clone()
        # Compute the Chebyshev interpolation
        y = torch.einsum('bid,iod->bo', cheby, self.cheby_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y
        

class fKAN(nn.Module):
    def __init__(self, layers, orders):
        super(fKAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-2):
            self.layers.append(ChebyKANLayer(layers[i], layers[i+1], orders[i]))
        self.layers.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def load_and_preprocess_data(path, source_columns=None, scaler=None, label_encoder=None):
    data = pd.read_csv(path)
    data.fillna(0, inplace=True)
    
    # Align features with source dataset
    if source_columns is not None:
        # Add missing columns with zeros
        missing_cols = set(source_columns) - set(data.columns)
        for col in missing_cols:
            data[col] = 0
        # Remove extra columns
        data = data[source_columns]
    
    # Separate features and target
    X = data.drop("target_label", axis=1).values.astype(np.float32)
    y = data["target_label"].values
    
    # Handle labels
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    else:
        y = label_encoder.transform(y)
    
    # Handle features
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return (
        torch.tensor(X).to(device),
        torch.tensor(y).to(device),
        scaler,
        label_encoder,
        data.columns.tolist()  # Return columns for alignment
    )

def prepare_model(input_dim, output_dim):
    # Best parameters from previous configuration
    return fKAN(
        layers=[input_dim, 65, 130, 130, output_dim],
        orders=[2, 6, 2]
    ).to(device)

def freeze_layers_except_last_cheby(model):
    cheby_indices = [i for i, layer in enumerate(model.layers) 
                    if isinstance(layer, ChebyKANLayer)]
    
    if not cheby_indices:
        raise ValueError("No ChebyKANLayer found in model")
    
    last_cheby_idx = max(cheby_indices)
    
    # Freeze all layers before last ChebyKANLayer
    for idx, layer in enumerate(model.layers):
        if idx < last_cheby_idx:
            for param in layer.parameters():
                param.requires_grad = False
                
    # Unfreeze last ChebyKANLayer and subsequent layers
    for layer in model.layers[last_cheby_idx:]:
        for param in layer.parameters():
            param.requires_grad = True

def fit(model, dataset, optimizer, loss_fn, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(dataset['train_input'])
        loss = loss_fn(outputs, dataset['train_label'])
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step(closure=lambda: loss)
        
        # Validation
        with torch.no_grad():
            val_outputs = model(dataset['test_input'])
            val_loss = loss_fn(val_outputs, dataset['test_label'])
            
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {loss.item():.4f}, '
              f'Val Loss: {val_loss.item():.4f}')

# Main execution flow
if __name__ == "__main__":
    # Step 1: Train on Data1 (source domain)
    print("Training on source domain (data1)...")

    # X1, y1, scaler, le = load_and_preprocess_data(DATA1_PATH)
    X1, y1, scaler, le, source_columns = load_and_preprocess_data(DATA1_PATH)

    X1_train, X1_val, y1_train, y1_val = train_test_split(X1, y1, test_size=0.2, random_state=42)
    
    model = prepare_model(X1.shape[1], len(le.classes_))
    optimizer = LBFGS(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    
    fit(model, {
        'train_input': X1_train,
        'train_label': y1_train,
        'test_input': X1_val,
        'test_label': y1_val
    }, optimizer, loss_fn)
    
    # Step 2: Fine-tune on Data2 (target domain)
    print("\nFine-tuning on target domain (data2)...")

   
    # X2, y2, _, _ = load_and_preprocess_data(DATA2_PATH, scaler, le)
    X2, y2, _, _, _ = load_and_preprocess_data(DATA2_PATH,source_columns=source_columns, scaler=scaler, label_encoder=le)

    X2_train, X2_val, y2_train, y2_val = train_test_split(X2, y2, test_size=0.2, random_state=42)


    # Freeze layers except last ChebyKANLayer
    freeze_layers_except_last_cheby(model)
    
    # Use smaller learning rate for fine-tuning
    optimizer = LBFGS(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01
    )
    
    fit(model, {
        'train_input': X2_train,
        'train_label': y2_train,
        'test_input': X2_val,
        'test_label': y2_val
    }, optimizer, loss_fn, epochs=50)
    
    # Final evaluation
    with torch.no_grad():
        outputs = model(X2_val)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y2_val).sum().item() / y2_val.size(0)
        print(f"\nFinal Transfer Learning Accuracy: {accuracy:.4f}")