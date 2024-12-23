# This code run by Conda KAN in Catbert
# rKAN (Rational KAN) with early stopping

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

######################################################################################################################################################

# data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/DATA/Credit-g(CG)/Preprocessed-test/data_processed.csv')
# data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/DATA/Credit-approval(CA)/Preprocessed-test/data_processed.csv')
# data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/DATA/dress-sale(DS)/Preprocessed-test/data_processed.csv')
# data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/DATA/Adult(AD)/Preprocessed-test/data_processed.csv')
# data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/DATA/cylinder-bands(CB)/Preprocessed-test/data_processed.csv')
# data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/DATA/Blastchar(BL)/Preprocessed-test/data_processed.csv')
# data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/DATA/insurance+company(IO)/Preprocessed-test/data_processed.csv')
data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/DATA/income (IC)/Preprocessed-test/data_processed.csv')

###################################################################################################################################################### 

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

# Define Custom Dataset
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

# Define the RationalFunction and rKAN Models
class RationalFunction(nn.Module):
    def __init__(self):
        super(RationalFunction, self).__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.a * x / (1 + self.b * x)

class rKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(rKAN, self).__init__()
        self.psi = nn.ModuleList([RationalFunction() for _ in range(input_dim)])
        self.phi = nn.ModuleList([RationalFunction() for _ in range(2 * input_dim + 1)])
        self.summation_weights = nn.Parameter(torch.randn(2 * input_dim + 1, input_dim))

    def forward(self, x):
        psi_outputs = [self.psi[i](x[:, i:i+1]) for i in range(x.size(1))]
        summation = sum(self.summation_weights[i, j] * psi_outputs[j] for i in range(2 * x.size(1) + 1) for j in range(x.size(1)))
        phi_outputs = [self.phi[i](summation) for i in range(2 * x.size(1) + 1)]
        output = sum(phi_outputs)
        return torch.sigmoid(output) 

# Initialize the rKAN Model
input_dim = X_combined.shape[1]
hidden_dim = 20  
output_dim = 1
model = rKAN(input_dim, hidden_dim, output_dim)

# Define Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Training Loop with Early Stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    model.train()
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)

    for epoch in range(num_epochs):
        total_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate_model(model, val_loader, criterion)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

# Evaluation Function
def evaluate_model(model, test_loader, criterion=None):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_outputs = []
    total_loss = 0

    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features).squeeze()
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    accuracy = correct / total
    auc = roc_auc_score(all_labels, all_outputs)

    if criterion:
        return total_loss / len(test_loader)
    else:
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Test AUC: {auc:.4f}')
        return accuracy, auc

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
train_dataset = MixedDataset(X_train, y_train)
val_dataset = MixedDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train and Evaluate the Model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100)
evaluate_model(model, test_loader)
