# This is a simple example of using tabkan for tabular supervised learning task.

import math
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from datetime import datetime
import optuna

from tabkan import ChebyshevKAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
 
############## What YOU SHOULD DO ##################################
EPOCHS = 100
TRIALS = 1000
MAX_DEPTH = 20
MAX_NEURONS = 200

DATA = 'CA'
data = pd.read_csv(f"/pat/to/your/DATA/{DATA}.csv")
#####################################################################


## Feature Engineering ####
data.fillna(0, inplace=True)
y = data["target_label"].values
data.drop("target_label", axis=1, inplace=True)
X = data.values.astype(np.float32)
if y.dtype == "object":
    label_encoder = LabelEncoder()
    y = torch.tensor(label_encoder.fit_transform(y))
else:
    y = torch.tensor(y)


# Train / validation / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1, random_state=42)

# Scaling and tensor conversion
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32).to(device)
X_valid = torch.tensor(scaler.transform(X_valid), dtype=torch.float32).to(device)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32).to(device)

y_train = torch.nn.functional.one_hot(y_train.long(), num_classes=2).float().to(device)
y_valid = torch.nn.functional.one_hot(y_valid.long(), num_classes=2).float().to(device)
y_test = torch.nn.functional.one_hot(y_test.long(), num_classes=2).float().to(device)

input_shape = X_train.shape[1]
output_shape = y_train.shape[1]

dataset = {
    "train_input": X_train,
    "train_label": y_train,
    "test_input": X_valid,
    "test_label": y_valid,
}

# Ceate model and tune hyperparameters
model = ChebyshevKAN()

## For Fine tune a New dataset
best_params = model.tune(
    dataset=dataset,
    input_shape=input_shape,
    output_shape=output_shape,
    device=device,
    trials=TRIALS,
    EPOCHS=EPOCHS,
    MAX_DEPTH=MAX_DEPTH,
    MAX_NEURONS=MAX_NEURONS
)

###### IF YOU HAVE ALREADY THE PARAMETERS ###
# best_params = {
#     "depth": 3,
#     "neurons_layer_0": 65,
#     "neurons_layer_1": 130,
#     "neurons_layer_2": 130,
#     "orders_layer_0": 2,
#     "orders_layer_1": 6,
#     "orders_layer_2": 2,        
# }

# print("Best hyperparameters:", best_params)

# Build model with best params
depth = best_params["depth"]
width = [best_params[f"neurons_layer_{i}"] for i in range(depth)]
orders = [best_params[f"orders_layer_{i}"] for i in range(depth)]

model = ChebyshevKAN(layers=[input_shape] + width + [output_shape], orders=orders).to(device)
history = model.fit(dataset, steps=EPOCHS, loss_fn=torch.nn.CrossEntropyLoss())

#Evaluate
y_score = model(X_test).cpu()
y_pred = (y_score > 0.5).int()

result = roc_auc_score(y_test.cpu(), y_pred)
print("ROC-AUC scores: %.4f" % result)
