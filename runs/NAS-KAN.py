#!/usr/bin/env python
# coding: utf-8

import math

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from kan import KAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 10
TRIALS = 20

MAX_DEPTH = 10
MAX_GRID = 9
MAX_K = 5
MAX_NEURONS = 100


DATA = 'CA'
data = pd.read_csv("../DATA/%s.csv" % DATA)
data.fillna(0, inplace=True)
y = data["target_label"].values
data.drop("target_label", axis=1, inplace=True)
X = data.values.astype(np.float32)

if y.dtype == "object":
    label_encoder = LabelEncoder()
    y = torch.tensor(label_encoder.fit_transform(y))
else:
    y = torch.tensor(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, stratify=y_train, test_size=0.1, random_state=42
)


# pca = PCA(n_components=30)
# X_train = torch.tensor(pca.fit_transform(X_train)).to(device)
# X_valid = torch.tensor(pca.transform(X_valid)).to(device)
# X_test = torch.tensor(pca.transform(X_test)).to(device)

X_train = torch.tensor(X_train).to(device)
X_valid = torch.tensor(X_valid).to(device)
X_test = torch.tensor(X_test).to(device)


y_train = torch.nn.functional.one_hot(y_train.long(), num_classes=2).to(device).float()
y_valid = torch.nn.functional.one_hot(y_valid.long(), num_classes=2).to(device).float()
y_test = torch.nn.functional.one_hot(y_test.long(), num_classes=2).to(device).float()


input_shape = X_train.shape[1]
output_shape = y_train.shape[1]

print("Reduce number of features from %d to %d." % (data.shape[1], input_shape))

dataset = {
    "train_input": X_train,
    "train_label": y_train,
    "test_input": X_valid,
    "test_label": y_valid,
}


def objective(trial):
    depth = trial.suggest_int("depth", 1, MAX_DEPTH)
    grid = trial.suggest_int("grid", 1, MAX_GRID, step=2)
    k = trial.suggest_int("k", 1, MAX_K)

    width = [trial.suggest_int(f"neurons_layer_{i}", 5, MAX_NEURONS, step=5) for i in range(depth)]
    width = [input_shape] + width + [output_shape]

    model = KAN(width=width, grid=grid, k=k, device=device)

    history = model.fit(dataset, steps=EPOCHS, loss_fn=torch.nn.CrossEntropyLoss())

    y_score = model(X_valid).cpu()
    y_pred = (y_score > 0.5).int()

    return f1_score(y_valid.cpu(), y_pred, average="macro")


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=TRIALS)


best_params = study.best_params
print('best_params:', best_params)
depth = best_params["depth"]
grid = best_params["grid"]
k = best_params["k"]
width = [best_params[f"neurons_layer_{i}"] for i in range(depth)]
width = [input_shape] + width + [output_shape]

model = KAN(width=width, grid=grid, k=k, device=device)

history = model.fit(dataset, steps=EPOCHS)

y_score = model(X_test).cpu()
y_pred = (y_score > 0.5).int()


print(classification_report(y_test.cpu(), y_pred, target_names=label_encoder.classes_))
print("ROC-AUC: %.4f" % roc_auc_score(y_test.cpu(), y_pred))


results = study.trials_dataframe()
vars = results[["params_depth", "params_grid", "params_k"]]
values = results["value"].round(2)
results.to_csv('results-%s-%s.csv' % (DATA, datetime.now()))

