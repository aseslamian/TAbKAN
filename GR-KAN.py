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
import math  
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

######################################################################################################################################################

class GRKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        group_size=8,
        rational_order=(5, 4),
        scale_base=1.0,
        scale_rational=1.0,
        base_activation=torch.nn.Identity,
    ):
        super(GRKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.rational_order = rational_order

        # Linear weight parameters for each group
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.rational_a = torch.nn.Parameter(torch.Tensor(group_size, rational_order[0] + 1))
        self.rational_b = torch.nn.Parameter(torch.Tensor(group_size, rational_order[1] + 1))
        
        self.scale_base = scale_base
        self.scale_rational = scale_rational
        self.base_activation = base_activation()
        
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5) * self.scale_base)
        torch.nn.init.kaiming_uniform_(self.rational_a, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.rational_b, a=math.sqrt(5))

    def rational_activation(self, x, a, b):
        # Rational function approximation for activation
        numerator = sum(a[i] * x**i for i in range(len(a)))
        denominator = 1 + torch.abs(sum(b[i] * x**i for i in range(1, len(b))))
        return numerator / denominator

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        # Apply linear transformation
        base_output = F.linear(self.base_activation(x), self.weight)

        # Group-wise rational activation
        group_size = min(self.group_size, self.in_features)
        feature_split_size = self.in_features // group_size
        grouped_x = x.view(x.size(0), group_size, feature_split_size)
        grouped_output = []
        for i in range(group_size):
            grouped_output.append(self.rational_activation(grouped_x[:, i, :], self.rational_a[i], self.rational_b[i]))
        rational_output = torch.cat(grouped_output, dim=1)

        return base_output + self.scale_rational * rational_output

######################################################################################################################################################

class KAT(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        group_size=8,
        rational_order=(5, 4),
        scale_base=1.0,
        scale_rational=1.0,
        base_activation=torch.nn.Identity,
    ):
        super(KAT, self).__init__()
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                GRKANLinear(
                    in_features,
                    out_features,
                    group_size=group_size,
                    rational_order=rational_order,
                    scale_base=scale_base,
                    scale_rational=scale_rational,
                    base_activation=base_activation,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(x)  # Apply sigmoid activation to ensure output is between 0 and 1

######################################################################################################################################################

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

######################################################################################################################################################

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

# Initialize KAT model
input_dim = X_combined.shape[1]
kat_layers = [input_dim, 64, 1]  
model = KAT(kat_layers, group_size=8, rational_order=(5, 4), scale_base=1.0, scale_rational=1.0, base_activation=torch.nn.Identity)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

######################################################################################################################################################

def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')

######################################################################################################################################################

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
