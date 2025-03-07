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
from scipy.io import arff
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

################################
def fit(model, dataset, steps=100, loss_fn=torch.nn.MSELoss(), lr=1.0, batch=-1):
    from torch.optim import LBFGS
    from tqdm import tqdm
    import numpy as np

    pbar = tqdm(range(steps), desc='Training', ncols=100)
    optimizer = LBFGS(model.parameters(),
                      lr=lr,
                      history_size=10,
                      line_search_fn="strong_wolfe",
                      tolerance_grad=1e-32,
                      tolerance_change=1e-32)

    results = {'train_loss': [], 'test_loss': []}

    # Determine batch sizes
    if batch == -1 or batch > dataset['train_input'].shape[0]:
        batch_size = dataset['train_input'].shape[0]
        batch_size_test = dataset['test_input'].shape[0]
    else:
        batch_size = batch
        batch_size_test = batch

    for _ in pbar:
        # Randomly select indices for train and test batches
        train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
        test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

        def closure():
            optimizer.zero_grad()
            pred = model(dataset['train_input'][train_id])
            loss = loss_fn(pred, dataset['train_label'][train_id])
            loss.backward()
            return loss

        optimizer.step(closure)

        # Evaluate losses without affecting gradients
        with torch.no_grad():
            train_pred = model(dataset['train_input'][train_id])
            train_loss_value = loss_fn(train_pred, dataset['train_label'][train_id])
            test_pred = model(dataset['test_input'][test_id])
            test_loss_value = loss_fn(test_pred, dataset['test_label'][test_id])        
        
        # Optionally, compute RMSE by taking the square root of MSE
        rmse_train = torch.sqrt(train_loss_value).cpu().detach().numpy()
        rmse_test  = torch.sqrt(test_loss_value).cpu().detach().numpy()
        results['train_loss'].append(rmse_train)
        results['test_loss'].append(rmse_test)

        pbar.set_description("| train_loss: %.2e | test_loss: %.2e " % (rmse_train, rmse_test))

    return results

################################
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, activation=None):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.activation = activation  # Could be e.g. nn.Tanh(), nn.ReLU(), or None

        self.cheby_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))
        
    def forward(self, x):
        # Reshape and apply activation if provided
        x = torch.reshape(x, (-1, self.inputdim))
        if self.activation:
            x = self.activation(x)
        
        # Initialize Chebyshev tensor
        cheby = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        
        if self.degree > 0:
            # Use clone() to avoid in-place modification issues
            cheby[:, :, 1] = x.clone()
        
        for i in range(2, self.degree + 1):
            # Clone intermediate results to ensure previous tensors are not overwritten
            cheby[:, :, i] = 2 * x * cheby[:, :, i - 1].clone() - cheby[:, :, i - 2].clone()
        
        # Compute the Chebyshev interpolation
        y = torch.einsum('bid,iod->bo', cheby, self.cheby_coeffs)
        return y
    
##########################################################################################
class fKAN(nn.Module):
    def __init__(self, layers, orders, activation=None):
        super(fKAN, self).__init__()
        self.layers = nn.ModuleList()
        # Use ChebyKANLayer for hidden layers
        for i in range(len(layers)-2):
            self.layers.append(
                ChebyKANLayer(layers[i], layers[i+1], orders[i], activation=activation)
            )
        # Final layer is linear (no activation for regression)
        self.layers.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
##########################################################################################
def prepare_model(input_dim, output_dim, best):
    depth = best['depth']
    layers = [input_dim]
    for i in range(depth):
        layers.append(best[f'neurons_layer_{i}'])
    layers.append(output_dim)
    orders = [best.get(f'orders_layer_{i}', 1) for i in range(depth)]
    
    return fKAN(layers, orders).to(device)

##########################################################################################

def objective(trial):
    # Sample hyperparameters
    depth = trial.suggest_int("depth", 1, MAX_DEPTH)
    width = [trial.suggest_int(f"neurons_layer_{i}", 5, MAX_NEURONS, step=5) for i in range(depth)]
    grids = [trial.suggest_int(f"grids_{i}", 1, MAX_GRID) for i in range(depth)]
    width = [input_shape] + width + [output_shape]
    model = fKAN(width, grids, activation=nn.Tanh()).to(device)
    history = fit(model, dataset, steps=EPOCHS, loss_fn=torch.nn.MSELoss())
    y_pred = model(X_valid).cpu().detach().numpy()
    y_true = y_valid.cpu().detach().numpy()
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    return rmse, abs(r2 - 1)

###############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

print("Cheb-KAN")

EPOCHS = 100
TRIALS = 100

MAX_DEPTH = 10
MAX_GRID = 10
MAX_NEURONS = 50

DATA = 'cpu'

data = arff.loadarff("/home/data3/Ali/Code/KAN/Regression/Dataset/%s.arff" % DATA)
data = pd.DataFrame(data[0])

data.fillna(0, inplace=True)
y = data["usr"].values
data.drop("usr", axis=1, inplace=True)
X = data.values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train)).to(device)
X_valid = torch.tensor(scaler.transform(X_valid)).to(device)
X_test = torch.tensor(scaler.transform(X_test)).to(device)

y_train = torch.tensor(y_train).to(device).float().view(-1, 1)
y_valid = torch.tensor(y_valid).to(device).float().view(-1, 1)
y_test = torch.tensor(y_test).to(device).float().view(-1, 1)

input_shape = X_train.shape[1]
output_shape = 1  # Regression target is a single value

print("-----------------------------------------------------------------------------------------")
print("Reduce number of features from %d to %d." % (data.shape[1], input_shape))

dataset = {
    "train_input": X_train,
    "train_label": y_train,
    "test_input": X_valid,
    "test_label": y_valid,
}

# Create a multi-objective study
study = optuna.create_study(directions=["minimize", "minimize"])
study.optimize(objective, n_trials=TRIALS)


BEST = study.best_params
print("Best hyperparameters", BEST)

model = prepare_model(input_shape, output_shape, BEST)
history = fit(model, dataset, steps=EPOCHS, loss_fn = torch.nn.MSELoss())

y_pred = model(X_test).cpu().detach().numpy()
y_true = y_test.cpu().detach().numpy()

# Calculate regression metrics
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
r2  = r2_score(y_true, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# results = study.trials_dataframe()
# results.to_csv('results-fourierKAN-%s-%s.csv' % (DATA, datetime.now()))
