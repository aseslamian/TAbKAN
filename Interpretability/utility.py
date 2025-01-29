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


################################
def fit(model, dataset, steps=100, loss_fn=None, lr=1., batch=-1):
    pbar = tqdm(range(steps), desc='description', ncols=100)
    optimizer = LBFGS(model.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)

    results = {}
    results['train_loss'] = []
    results['test_loss'] = []

    if batch == -1 or batch > dataset['train_input'].shape[0]:
        batch_size = dataset['train_input'].shape[0]
        batch_size_test = dataset['test_input'].shape[0]
    else:
        batch_size = batch
        batch_size_test = batch

    global train_loss

    def closure():
        global train_loss
        optimizer.zero_grad()
        pred = model.forward(dataset['train_input'][train_id])
        train_loss = loss_fn(pred, dataset['train_label'][train_id])
        objective = train_loss
        objective.backward()
        return objective

    for _ in pbar:


        train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
        test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

        optimizer.step(closure)

        test_loss = loss_fn(model.forward(dataset['test_input'][test_id]), dataset['test_label'][test_id])

        results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
        results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
        pbar.set_description("| train_loss: %.2e | test_loss: %.2e " % (torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy()))

    return results

# def objective(trial):
#     depth = trial.suggest_int("depth", 1, MAX_DEPTH)
#     width = [trial.suggest_int(f"neurons_layer_{i}", 5, MAX_NEURONS, step=5) for i in range(depth)]
#     grids = [trial.suggest_int(f"grids_{i}", 1, MAX_GRID) for i in range(depth)]
#     width = [input_shape] + width + [output_shape]
#     model = fKAN(width, grids).to(device)

#     history = fit(model, dataset, steps=EPOCHS, loss_fn=torch.nn.CrossEntropyLoss())

#     y_score = model(X_valid).cpu()
#     y_pred = (y_score > 0.5).int()

#     return f1_score(y_valid.cpu(), y_pred, average="macro")
###############################