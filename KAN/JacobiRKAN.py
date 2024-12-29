# This code run by Conda KAN in Catbert
# JacobiRKAN 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from sklearn.metrics import roc_auc_score
from rkan.torch import JacobiRKAN, PadeRKAN

# Loading and Preprocessing Data
# data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/Test_2/DATA/Credit-g(CG)/Preprocessed-test/data_processed.csv')
data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/Test_2/DATA/income (IC)/Preprocessed-test/data_processed.csv')  

labels = data['target_label'].values
data.drop('target_label', axis=1, inplace=True)

#######################################################################
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns
scaler = StandardScaler()
numerical_data = scaler.fit_transform(data[numerical_cols])
tokenizer = get_tokenizer("basic_english")
tokenized_texts = [tokenizer(str(text)) for text in data[categorical_cols[0]]]
vocab = set([word for text in tokenized_texts for word in text])
word_to_idx = {word: idx for idx, word in enumerate(vocab, 1)}
indexed_tokens = [[word_to_idx[word] for word in text] for text in tokenized_texts]
max_sequence_length = 100
padded_tokens = pad_sequence([torch.tensor(text[:max_sequence_length]) for text in indexed_tokens], batch_first=True, padding_value=0)
X_combined = np.concatenate([numerical_data, padded_tokens.numpy()], axis=1)
#######################################################################

if labels.dtype == 'object':
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
else:
    labels_encoded = labels
labels_encoded = torch.tensor(labels_encoded, dtype=torch.float32)
X_train, X_test, y_train, y_test = train_test_split(X_combined, labels_encoded, test_size=0.2, random_state=42)

#######################################################################

class MixedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = MixedDataset(X_train, y_train)
test_dataset = MixedDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the Model using JacobiRKAN and PadeRKAN
input_dim = X_combined.shape[1]
model = nn.Sequential(
    nn.Linear(input_dim, 16),
    JacobiRKAN(3),      
    nn.Linear(16, 32),
    PadeRKAN(2, 6),     
    nn.Linear(32, 1),
    nn.Sigmoid() 
)

# Define Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

# Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features).squeeze()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    accuracy = correct / total
    auc = roc_auc_score(all_labels, all_outputs)
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test AUC: {auc:.4f}')

# Train and Evaluate the Model
train_model(model, train_loader, criterion, optimizer, num_epochs=100)
evaluate_model(model, test_loader)
