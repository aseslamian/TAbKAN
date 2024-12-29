import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
import math  
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=True, smooth_initialization=False):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        
        grid_norm_factor = (torch.arange(gridsize) + 1) ** 2 if smooth_initialization else np.sqrt(gridsize)
        
        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) / 
                                          (np.sqrt(inputdim) * grid_norm_factor))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1)) 
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        y = torch.sum(c * self.fouriercoeffs[0:1], (-2, -1)) 
        y += torch.sum(s * self.fouriercoeffs[1:2], (-2, -1))
        if self.addbias:
            y += self.bias
        y = torch.reshape(y, outshape)
        return y

class KAN_Fourier(nn.Module):
    def __init__(self, layers_hidden, gridsize=10, smooth_initialization=False):
        super(KAN_Fourier, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(NaiveFourierKANLayer(in_features, out_features, gridsize, 
                                                    smooth_initialization=smooth_initialization))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(x)

data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/Test_2/DATA/Credit-g(CG)/Preprocessed-test/data_processed.csv')

labels = data['target_label'].values
data.drop('target_label', axis=1, inplace=True)
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

if labels.dtype == 'object':
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
else:
    labels_encoded = labels
labels_encoded = torch.tensor(labels_encoded, dtype=torch.float32)
X_train, X_test, y_train, y_test = train_test_split(X_combined, labels_encoded, test_size=0.2, random_state=42)

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

input_dim = X_combined.shape[1]
kan_layers = [input_dim, 64, 1]  
model = KAN_Fourier(kan_layers, gridsize=10, smooth_initialization=True)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')

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
    
    print(f'Test Accuracy: {accuracy}')
    print(f'Test AUC: {auc}')

train_model(model, train_loader, criterion, optimizer)
evaluate_model(model, test_loader)
