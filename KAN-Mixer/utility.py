import math
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
from tqdm import tqdm
from torch.optim import LBFGS
import torch.nn as nn
from scipy.io import arff
from torch.utils.data import DataLoader, TensorDataset 
import torch.optim as optim

################################################################################################################


TRIALS = 3
EPOCHS = 100
NUM_SEED = 10
# BATCH_SIZE = 64

################################################################################################################
def fit(model, criterion, optimizer, X_train, y_train, epochs=10, batch_size=32, device='cuda'):
    
    history = {'train_loss': [], 'val_loss': []}
    # print ("Batch size: ", batch_size)
    # Convert to float32 once and move to device
    X_train, y_train = X_train.float().to(device), y_train.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    
    for epoch in range(epochs):
        
        model.train()
        epoch_train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x_batch.size(0)

        # Calculate epoch metrics
        train_loss = epoch_train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # print(f'Epoch {epoch+1}/{epochs}')
    
    return history
################################################################################################################

def evaluate_model_auc(model, X_test, y_test, output_shape, device='cuda', batch_size=64):
        
    # Create test dataset and loader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.to(device)
    model.eval()
    
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            
            # Compute probabilities (for binary classification, probability for class 1)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            all_probabilities.append(probabilities.cpu())
            all_labels.append(batch_y.cpu())
    
    test_probabilities = torch.cat(all_probabilities)
    test_labels = torch.cat(all_labels)
    
    # Calculate metrics
    test_auc = roc_auc_score(test_labels.numpy(), test_probabilities.numpy())    
    test_predictions = (test_probabilities > 0.5).long()
    test_accuracy = (test_predictions == test_labels).float().mean().item()
      
    return test_auc, test_accuracy
################################################################################################################

def create_data_loaders(X_train, y_train, X_valid, y_valid, batch_size=32):
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader
################################################################################################################

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                valid_loss += loss.item() * batch_x.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_acc = correct / len(valid_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
        model.train()