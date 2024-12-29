  # This code run on Catbert for KAN wavelet transformation

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
import math
import torch.nn.functional as F


######################################################################################################################################################

data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/DATA/Credit-g(CG)/Preprocessed-test/data_processed.csv')
# data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/DATA/Credit-approval(CA)/Preprocessed-test/data_processed.csv')
# data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/DATA/dress-sale(DS)/Preprocessed-test/data_processed.csv')
# data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/DATA/Adult(AD)/Preprocessed-test/data_processed.csv')
# data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/DATA/cylinder-bands(CB)/Preprocessed-test/data_processed.csv')
# data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/DATA/Blastchar(BL)/Preprocessed-test/data_processed.csv')
# data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/DATA/insurance+company(IO)/Preprocessed-test/data_processed.csv')
# data = pd.read_csv('/home/data3/Ali/Code/KAN/TabKAN/DATA/income (IC)/Preprocessed-test/data_processed.csv')

######################################################################################################################################################

labels = data['target_label'].values
data.drop('target_label', axis=1, inplace=True)
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns
scaler = StandardScaler()
numerical_data = scaler.fit_transform(data[numerical_cols])

# Tokenization and padding
tokenizer = get_tokenizer("basic_english")
tokenized_texts = [tokenizer(str(text)) for text in data[categorical_cols[0]]]
vocab = set([word for text in tokenized_texts for word in text])
word_to_idx = {word: idx for idx, word in enumerate(vocab, 1)}
indexed_tokens = [[word_to_idx[word] for word in text] for text in tokenized_texts]
max_sequence_length = 100
padded_tokens = pad_sequence([torch.tensor(text[:max_sequence_length]) for text in indexed_tokens], batch_first=True, padding_value=0)

# Ensure correct shapes for concatenation
numerical_data_tensor = torch.tensor(numerical_data, dtype=torch.float32)
padded_tokens_tensor = padded_tokens.float()
X_combined = torch.cat((numerical_data_tensor, padded_tokens_tensor), dim=1)

# Encode labels
if labels.dtype == 'object':
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
else:
    labels_encoded = labels
labels_encoded = torch.tensor(labels_encoded, dtype=torch.float32)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_combined, labels_encoded, test_size=0.2, random_state=42)

# Define Custom Dataset
class MixedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = MixedDataset(X_train, y_train)
test_dataset = MixedDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

######################################################################################################################################################
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        # Linear weights for combining outputs
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features)) #not used; you may like to use it for wieghting base activation and adding it like Spl-KAN paper
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # Base activation function #not used for this experiment
        self.base_activation = nn.SiLU()

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        # Implementation of different wavelet types
        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2)-1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi**0.25)) * term1 * term2
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'dog':
            # Implementing Derivative of Gaussian Wavelet 
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'meyer':
            # Implement Meyer Wavelet here
            # Constants for the Meyer wavelet transition boundaries
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1/2,torch.ones_like(v),torch.where(v >= 1,torch.zeros_like(v),torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)
            # Meyer wavelet calculation using the auxiliary function
            wavelet = torch.sin(pi * v) * meyer_aux(v)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'shannon':
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)  # sinc(x) = sin(pi*x) / (pi*x)

            # Applying a Hamming window to limit the infinite support of the sinc function
            window = torch.hamming_window(x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype, device=x_scaled.device)
            # Shannon wavelet is the product of the sinc function and the window
            wavelet = sinc * window
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        else:
            raise ValueError("Unsupported wavelet type")

        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        base_output = F.linear(x, self.weight1)
        combined_output =  wavelet_output #+ base_output 

        # Apply batch normalization
        return self.bn(combined_output)
######################################################################################################################################################
class KAN(nn.Module):
    def __init__(self, layers_hidden, wavelet_type='mexican_hat'):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(KANLinear(in_features, out_features, wavelet_type))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
######################################################################################################################################################
# Initialize the KAN Model
input_dim = X_combined.shape[1]
hidden_dim = 100
output_dim = 1
layers_hidden = [input_dim, hidden_dim, output_dim]  # Adjust according to your architecture
model = KAN(layers_hidden, wavelet_type='mexican_hat')

# Define Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

######################################################################################################################################################
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
        
######################################################################################################################################################
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
            predicted = (outputs > 0).float()
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
