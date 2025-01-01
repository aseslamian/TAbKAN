#!/usr/bin/env python
# coding: utf-8

import math
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from fckan import KANLayer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
from tqdm import tqdm
from torch.optim import LBFGS
import torch.nn as nn

########################################

# def fit(model, dataset, steps=100, loss_fn=None, lr=1.0, batch=-1):
#     pbar = tqdm(range(steps), desc='Training', ncols=100)
#     optimizer = LBFGS(
#         model.parameters(),
#         lr=lr,
#         history_size=10,
#         line_search_fn="strong_wolfe",
#         tolerance_grad=1e-32,
#         tolerance_change=1e-32
#     )

#     results = {'train_loss': [], 'test_loss': []}

#     if batch == -1 or batch > dataset['train_input'].shape[0]:
#         batch_size = dataset['train_input'].shape[0]
#         batch_size_test = dataset['test_input'].shape[0]
#     else:
#         batch_size = batch
#         batch_size_test = batch

#     def closure():
#         optimizer.zero_grad()
#         train_input = dataset['train_input'][train_id].to(device)
#         train_label = dataset['train_label'][train_id].to(device)
#         pred = model(train_input)
#         loss = loss_fn(pred, train_label)
#         loss.backward()
#         return loss

#     for _ in pbar:
#         train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
#         test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

#         optimizer.step(closure)

#         train_loss = closure()
#         test_input = dataset['test_input'][test_id].to(device)
#         test_label = dataset['test_label'][test_id].to(device)
#         test_loss = loss_fn(model(test_input), test_label)

#         results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
#         results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
#         pbar.set_description(
#             "| train_loss: %.2e | test_loss: %.2e " % (
#                 torch.sqrt(train_loss).cpu().detach().numpy(),
#                 torch.sqrt(test_loss).cpu().detach().numpy()
#             )
#         )

#     return results
########################################
def fit(model, dataset, steps=100, loss_fn=None, lr=1.0, batch=32):  # Default batch size is now 32
    pbar = tqdm(range(steps), desc='Training', ncols=100)
    optimizer = LBFGS(
        model.parameters(),
        lr=lr,
        history_size=10,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-32,
        tolerance_change=1e-32
    )

    results = {'train_loss': [], 'test_loss': []}

    # Adjust batch sizes
    batch_size = min(batch, dataset['train_input'].shape[0])  # Limit batch size to dataset size
    batch_size_test = min(batch, dataset['test_input'].shape[0])  # Similarly for test batches

    def closure():
        optimizer.zero_grad()
        train_input = dataset['train_input'][train_id].to(device)
        train_label = dataset['train_label'][train_id].to(device)
        pred = model(train_input)
        loss = loss_fn(pred, train_label)
        loss.backward()
        return loss

    for _ in pbar:
        # Randomly sample a batch of data
        train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
        test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

        optimizer.step(closure)

        train_loss = closure()
        test_input = dataset['test_input'][test_id].to(device)
        test_label = dataset['test_label'][test_id].to(device)
        test_loss = loss_fn(model(test_input), test_label)

        # Track loss values
        results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
        results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
        pbar.set_description(
            "| train_loss: %.2e | test_loss: %.2e " % (
                torch.sqrt(train_loss).cpu().detach().numpy(),
                torch.sqrt(test_loss).cpu().detach().numpy()
            )
        )

    return results

########################################

class fKAN(nn.Module):
    def __init__(self, layers):
        super(fKAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 2):
            self.layers.append(KANLayer(layers[i], layers[i + 1]))
        self.layers.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def objective(trial):
    depth = trial.suggest_int("depth", 1, MAX_DEPTH)
    width = [trial.suggest_int(f"neurons_layer_{i}", 5, MAX_NEURONS, step=5) for i in range(depth)]
    width = [input_shape] + width + [output_shape]

    model = fKAN(width).to(device)
    history = fit(model, dataset, steps=EPOCHS, loss_fn=torch.nn.CrossEntropyLoss())

    y_score = model(X_valid).cpu()
    y_pred = (y_score > 0.5).int()

    return f1_score(y_valid.cpu(), y_pred, average="macro")


########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

EPOCHS = 2
TRIALS = 3

MAX_DEPTH = 20
MAX_NEURONS = 200

DATA = 'CA'
data = pd.read_csv(f"/home/data3/Ali/Code/KAN/Afzal/DATA/{DATA}.csv")

data.fillna(0, inplace=True)
y = data["target_label"].values
data.drop("target_label", axis=1, inplace=True)
X = data.values.astype(np.float32)

if y.dtype == "object":
    label_encoder = LabelEncoder()
    y = torch.tensor(label_encoder.fit_transform(y))
else:
    y = torch.tensor(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train)).to(device)
X_valid = torch.tensor(scaler.transform(X_valid)).to(device)
X_test = torch.tensor(scaler.transform(X_test)).to(device)

y_train = torch.nn.functional.one_hot(y_train.long(), num_classes=2).to(device).float()
y_valid = torch.nn.functional.one_hot(y_valid.long(), num_classes=2).to(device).float()
y_test = torch.nn.functional.one_hot(y_test.long(), num_classes=2).to(device).float()

input_shape = X_train.shape[1]
output_shape = y_train.shape[1]

print(f"Reduce number of features from {data.shape[1]} to {input_shape}.")

dataset = {
    "train_input": X_train,
    "train_label": y_train,
    "test_input": X_valid,
    "test_label": y_valid,
}

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=TRIALS)

best_params = study.best_params
print('Best Parameters:', best_params)

depth = best_params["depth"]
width = [best_params[f"neurons_layer_{i}"] for i in range(depth)]
width = [input_shape] + width + [output_shape]

model = fKAN(width).to(device)
history = fit(model, dataset, steps=EPOCHS, loss_fn=torch.nn.CrossEntropyLoss())

y_score = model(X_test).cpu()
y_pred = (y_score > 0.5).int()

print(classification_report(y_test.cpu(), y_pred))
print("ROC-AUC: %.4f" % roc_auc_score(y_test.cpu(), y_pred))

results = study.trials_dataframe()
results.to_csv(f'results-fcKAN-{DATA}-{datetime.now()}.csv')
