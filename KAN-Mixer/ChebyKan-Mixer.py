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
from torch.utils.data import DataLoader, TensorDataset 
import torch.optim as optim
from utility import create_data_loaders, train_model

################################################################################################################

# --- KAN Block Components -------------------------------------------
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        # x shape: (batch, input_dim)
        x = torch.reshape(x, (-1, self.inputdim))
        # Normalize x into [-1, 1]
        x = torch.tanh(x)
        # Build Chebyshev polynomial tensor: shape (batch, input_dim, degree+1)
        cheby = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:
            cheby[:, :, 1] = x
        for i in range(2, self.degree + 1):
            cheby[:, :, i] = 2 * x * cheby[:, :, i - 1].clone() - cheby[:, :, i - 2].clone()
        # Interpolate using the learned Chebyshev coefficients
        y = torch.einsum('bid,iod->bo', cheby, self.cheby_coeffs)
        y = y.view(-1, self.outdim)
        return y

class fKAN(nn.Module):
    def __init__(self, layers, orders):
        super(fKAN, self).__init__()
        self.layers = nn.ModuleList()
        # Create ChebyKAN layers for all but the final layer
        for i in range(len(layers) - 2):
            self.layers.append(ChebyKANLayer(layers[i], layers[i + 1], orders[i]))
        # Final layer as a simple Linear mapping
        self.layers.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        # x is expected to be (batch, input_dim)
        for layer in self.layers:
            x = layer(x)
        return x

################################################################################################################
# --- Mixer Layer with KAN for Token and Channel Mixing ---------------
class MixerLayerKAN(nn.Module):
    def __init__(self, num_tokens, token_dim, channel_dim, token_order=3, channel_order=3):
        super(MixerLayerKAN, self).__init__()
        self.norm1 = nn.LayerNorm(channel_dim)
        self.token_mixing = fKAN(layers=[num_tokens, token_dim, num_tokens], orders=[token_order])
        
        self.norm2 = nn.LayerNorm(channel_dim)
        self.channel_mixing = fKAN(layers=[channel_dim, channel_dim * 2, channel_dim], orders=[channel_order])
        
    def forward(self, x):

        y = self.norm1(x)
        y = y.transpose(1, 2)
        B, C, T = y.shape
        y = y.reshape(B * C, T)
        y = self.token_mixing(y)
        y = y.reshape(B, C, T).transpose(1, 2)
        x = x + y

        # ----- Channel Mixing -----
        y = self.norm2(x)
        B, T, C = y.shape
        y = y.reshape(B * T, C)
        y = self.channel_mixing(y)
        y = y.reshape(B, T, C)
        x = x + y
        return x

# --- MLPMixer Model with KAN for Tabular Data -----------------------
class KANMixer(nn.Module):
    def __init__(self, num_features, num_classes, num_layers=4, token_dim=64, channel_dim=128,
                 token_order=3, channel_order=3):
        """
        num_features: Number of tokens (features/columns)
        num_classes: Number of target classes
        num_layers: Number of Mixer layers
        token_dim: Hidden size for token mixing
        channel_dim: Embedding dimension for each token
        token_order, channel_order: Degree for Chebyshev polynomial in token and channel mixing respectively
        """
        super(KANMixer, self).__init__()
        self.num_tokens = num_features
        # Embed each feature (scalar) into a vector of size channel_dim.
        self.embedding = nn.Linear(1, channel_dim)
        
        # Sequence of Mixer layers using KAN blocks.
        self.mixer_layers = nn.Sequential(*[
            MixerLayerKAN(num_tokens=self.num_tokens, token_dim=token_dim, channel_dim=channel_dim,
                          token_order=token_order, channel_order=channel_order)
            for _ in range(num_layers)
        ])
        
        # Final normalization and classification head.
        self.norm = nn.LayerNorm(channel_dim)
        self.fc = nn.Linear(channel_dim, num_classes)
        
    def forward(self, x):
     
        x = x.unsqueeze(-1) 
        x = self.embedding(x)  
        x = self.mixer_layers(x)  
        x = self.norm(x)
        x = x.mean(dim=1)  
        logits = self.fc(x) 

        return logits

################################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset_P = ['CG']
dataset_P = ['AD', 'IC']

seeds = np.random.randint(10, 1000, 5)

for DATA in dataset_P:

    # data = pd.read_csv("/home/data3/Ali/Code/KAN/Afzal/DATA/%s.csv" % DATA)
    data = pd.read_csv("../DATA/%s.csv" % DATA)
    
    results_AUC = []

    data.fillna(0, inplace=True)
    y = data["target_label"].values
    data.drop("target_label", axis=1, inplace=True)
    X = data.values.astype(np.float32)

    if y.dtype == "object":
        label_encoder = LabelEncoder()
        y = torch.tensor(label_encoder.fit_transform(y))
    else:
        y = torch.tensor(y)

    for seed in seeds:

        X_temp, X_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.1, random_state=42)

        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train)).to(device)
        X_valid = torch.tensor(scaler.transform(X_valid)).to(device)
        X_test = torch.tensor(scaler.transform(X_test)).to(device)
        X_temp = torch.tensor(scaler.transform(X_temp)).to(device)

        y_train = torch.nn.functional.one_hot(y_train.long(), num_classes=2).to(device).float()
        y_valid = torch.nn.functional.one_hot(y_valid.long(), num_classes=2).to(device).float()
        y_test = torch.nn.functional.one_hot(y_test.long(), num_classes=2).to(device).float()
        y_temp = torch.nn.functional.one_hot(y_temp.long(), num_classes=2).to(device).float()

        input_shape = X_train.shape[1]
        output_shape = y_train.shape[1]

        if output_shape > 1:
            y_train = y_train.argmax(dim=1)
            y_valid = y_valid.argmax(dim=1)
            y_test = y_test.argmax(dim=1)
            y_temp = y_temp.argmax(dim=1)

        ################################################################################################################

        model = KANMixer(
            num_features=input_shape, 
            num_classes=output_shape, 
            num_layers=4, 
            token_dim=64, 
            channel_dim=128, 
            token_order=3, 
            channel_order=3
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        train_loader, valid_loader = create_data_loaders(X_train, y_train, X_valid, y_valid, batch_size=32)
        train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=50)

        ################################################################################################################

        y_pred = model(X_test.to(device)).argmax(dim=1)
        y_pred = (y_pred).cpu()
        test_auc = roc_auc_score(y_test.cpu(), y_pred.cpu())
        print("Test AUC: ", test_auc)
        results_AUC.append(test_auc)

    top_result = sorted(results_AUC, reverse=True)[:1]
    # average_top_10 = sum(top_10_results) / len(top_10_results)
    print("Dataset:", DATA)
    print("Best AUC: ", top_result)
