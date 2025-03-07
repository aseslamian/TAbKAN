import math
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.optim import LBFGS
import torch.nn as nn
import optuna
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import os
import torch.nn.functional as F

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 64
TRIALS = 100

MAX_DEPTH = 20
MAX_NEURONS = 200

##########################################################################################################################
class PadeRKAN(nn.Module):
    def __init__(self, degree_p, degree_q):
        """
        Initialize the PadeRKAN module.

        Args:
            degree_p (int): Degree of the P polynomial.
            degree_q (int): Degree of the Q polynomial.
        """
        super(PadeRKAN, self).__init__()

        if not 0 < degree_p < 7 or not 0 < degree_q < 7:
            raise ValueError('Both degree_p and degree_q must be between one and six (inclusive).')

        self.degree_p = degree_p
        self.degree_q = degree_q

        # Define the trainable parameters for the P polynomial
        self.alpha_p = nn.Parameter(torch.ones(1))
        self.beta_p = nn.Parameter(torch.ones(1))
        self.zeta_p = nn.Parameter(torch.zeros(1))
        self.w_p = nn.Parameter(torch.ones(self.degree_p))

        # Define the trainable parameters for the Q polynomial
        self.alpha_q = nn.Parameter(torch.ones(1))
        self.beta_q = nn.Parameter(torch.ones(1))
        self.zeta_q = nn.Parameter(torch.zeros(1))
        self.w_q = nn.Parameter(torch.ones(self.degree_q))

    def forward(self, inputs):
        """
        Forward pass of the PadeRKAN module.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the Pade rational function.
        """
        # Normalize parameters for the P polynomial
        normalized_alpha_p = F.elu(self.alpha_p, 1)
        normalized_beta_p = F.elu(self.beta_p, 1)
        normalized_zeta_p = torch.sigmoid(self.zeta_p)

        # Normalize parameters for the Q polynomial
        normalized_alpha_q = F.elu(self.alpha_q, 1)
        normalized_beta_q = F.elu(self.beta_q, 1)
        normalized_zeta_q = torch.sigmoid(self.zeta_q)

        # Normalize inputs
        normalized_inputs = torch.sigmoid(inputs)

        # Calculate the P polynomial
        p = self.w_p[0] + self.w_p[1] * normalized_inputs
        for deg in range(2, self.degree_p):
            p += self.w_p[deg] * shifted_jacobi_polynomial(
                normalized_inputs,
                deg,
                normalized_alpha_p,
                normalized_beta_p,
                normalized_zeta_p,
                0,
                1,
                backend=torch
            )

        # Calculate the Q polynomial
        q = self.w_q[0] + self.w_q[1] * normalized_inputs
        for deg in range(2, self.degree_q):
            q += self.w_q[deg] * shifted_jacobi_polynomial(
                normalized_inputs,
                deg,
                normalized_alpha_q,
                normalized_beta_q,
                normalized_zeta_q,
                0,
                1,
                backend=torch
            )

        # Return the Pade rational function
        return p / q
    
##########################################################################################
class fKAN(nn.Module):
    def __init__(self, layers, orders1, orders2):
        super(fKAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(PadeRKAN(orders1[i], orders2[i]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
##########################################################################################
def prepare_model(input_dim, output_dim, best_params):

    depth = best_params["depth"]
    width = [best_params[f"neurons_layer_{i}"] for i in range(depth)]
    width = [input_shape] + width + [output_shape]
    orders1 = [best_params[f"orders1_layer_{i}"] for i in range(depth)]
    orders2 = [best_params[f"orders2_layer_{i}"] for i in range(depth)]
    
    return fKAN(width, orders1, orders2).to(device)
###########################################################################################
def freeze_all_except_last_cheby(model):
    cheby_indices = [i for i, layer in enumerate(model.layers) if isinstance(layer, PadeRKAN)] 
    if not cheby_indices:
        raise ValueError("No PadeRKAN found in model.")
    last_cheby_idx = max(cheby_indices)
    
    # Unfreeze last Cheby layer and output layer
    for idx, layer in enumerate(model.layers):
        if idx == last_cheby_idx or idx == len(model.layers) - 1:
            for param in layer.parameters():
                param.requires_grad = True
        else:
            for param in layer.parameters():
                param.requires_grad = False
    return model
###########################################################################################

def shifted_jacobi_polynomial(x, n, alpha, beta, zeta, a, b, backend):
    if n == 1:
        return (
            alpha - beta + (alpha + beta + 2) * (2 * x**zeta - a - b) / (b - a)
        ) / 2
    elif n == 2:
        return (
            ((alpha + 1) * (alpha + 2)) / 2
            + (
                (alpha + 2)
                * (3 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1)
            )
            / 2
            + (
                (3 + alpha + beta)
                * (4 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 2
            )
            / 8
        )
    elif n == 3:
        return (
            ((alpha + 1) * (alpha + 2) * (3 + alpha)) / 6
            + (
                (alpha + 2)
                * (3 + alpha)
                * (4 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1)
            )
            / 4
            + (
                (3 + alpha)
                * (4 + alpha + beta)
                * (5 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 2
            )
            / 8
            + (
                (4 + alpha + beta)
                * (5 + alpha + beta)
                * (6 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 3
            )
            / 48
        )
    elif n == 4:
        return (
            ((alpha + 1) * (alpha + 2) * (3 + alpha) * (4 + alpha)) / 24
            + (
                (alpha + 2)
                * (3 + alpha)
                * (4 + alpha)
                * (5 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1)
            )
            / 12
            + (
                (3 + alpha)
                * (4 + alpha)
                * (5 + alpha + beta)
                * (6 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 2
            )
            / 16
            + (
                (4 + alpha)
                * (5 + alpha + beta)
                * (6 + alpha + beta)
                * (7 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 3
            )
            / 48
            + (
                (5 + alpha + beta)
                * (6 + alpha + beta)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 4
            )
            / 384
        )
    elif n == 5:
        return (
            ((alpha + 1) * (alpha + 2) * (alpha + 3) * (alpha + 4) * (alpha + 5)) / 120
            + (
                (alpha + 2)
                * (alpha + 3)
                * (alpha + 4)
                * (alpha + 5)
                * (6 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1)
            )
            / 48
            + (
                (alpha + 3)
                * (alpha + 4)
                * (alpha + 5)
                * (6 + alpha + beta)
                * (7 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 2
            )
            / 48
            + (
                (alpha + 4)
                * (alpha + 5)
                * (6 + alpha + beta)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 3
            )
            / 96
            + (
                (alpha + 5)
                * (6 + alpha + beta)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 4
            )
            / 384
            + (
                (6 + alpha + beta)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * (10 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 5
            )
            / 3840
        )
    elif n == 6:
        return (
            (
                (alpha + 1)
                * (alpha + 2)
                * (alpha + 3)
                * (alpha + 4)
                * (alpha + 5)
                * (6 + alpha)
            )
            / 720
            + (
                (alpha + 2)
                * (alpha + 3)
                * (alpha + 4)
                * (alpha + 5)
                * (6 + alpha)
                * (7 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1)
            )
            / 240
            + (
                (alpha + 3)
                * (alpha + 4)
                * (alpha + 5)
                * (6 + alpha)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 2
            )
            / 192
            + (
                (alpha + 4)
                * (alpha + 5)
                * (6 + alpha)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 3
            )
            / 288
            + (
                (alpha + 5)
                * (6 + alpha)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * (10 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 4
            )
            / 768
            + (
                (6 + alpha)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * (10 + alpha + beta)
                * (11 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 5
            )
            / 3840
            + (
                (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * (10 + alpha + beta)
                * (11 + alpha + beta)
                * (12 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 6
            )
            / 46080
        )
    elif n > 6:
        raise ValueError(
            f"The current implementation supports a maximum degree of 6, but you entered {n}. Higher degrees may lead to numerical instabilities, overfitting, and increased computational complexity. Please consider using a lower degree."
        )
    elif n <= 0:
        raise ValueError(
            "Degrees must be positive. Zero or Negative degrees are not allowed."
        )

############################################################################################
def fit(model, dataset, steps=100, loss_fn=None, lr=1., batch=-1):
    
    pbar = tqdm(range(steps), desc='Training', ncols=100)
    # Only optimize parameters that require gradients.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = LBFGS(trainable_params, lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)

    results = {'train_loss': [], 'test_loss': []}

    # Determine batch sizes
    if batch == -1 or batch > dataset['train_input'].shape[0]:
        batch_size = dataset['train_input'].shape[0]
        batch_size_test = dataset['test_input'].shape[0]
    else:
        batch_size = batch
        batch_size_test = batch

    # Using a global to hold the train loss across closure calls (for reporting)
    global train_loss

    def closure():
        global train_loss
        optimizer.zero_grad()
        # Use a random batch (captured from the outer scope)
        pred = model(dataset['train_input'][train_id])
        train_loss = loss_fn(pred, dataset['train_label'][train_id])
        train_loss.backward()
        return train_loss

    for _ in pbar:
        # Sample a random batch from the training and test sets
        train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
        test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

        optimizer.step(closure)
        # Evaluate on the test batch without gradients
        with torch.no_grad():
            test_loss = loss_fn(model(dataset['test_input'][test_id]), dataset['test_label'][test_id])
        
        # Record losses (taking square root if desired for display; adjust if needed)
        train_loss_val = torch.sqrt(train_loss.detach()).cpu().numpy()
        test_loss_val  = torch.sqrt(test_loss.detach()).cpu().numpy()
        results['train_loss'].append(train_loss_val)
        results['test_loss'].append(test_loss_val)
        pbar.set_description("| train_loss: %.2e | test_loss: %.2e " % (train_loss_val, test_loss_val))
        
    return results

#############################################################################################
def objective(trial):
    depth = trial.suggest_int("depth", 1, MAX_DEPTH)

    width = [trial.suggest_int(f"neurons_layer_{i}", 5, MAX_NEURONS, step=5) for i in range(depth)]
    orders1 = [trial.suggest_int(f"orders1_layer_{i}", 2, 6) for i in range(depth)]
    orders2 = [trial.suggest_int(f"orders2_layer_{i}", 2, 6) for i in range(depth)]
    width = [input_shape] + width + [output_shape]
    model = fKAN(width, orders1, orders2).to(device)

    history = fit(model, dataset, steps=EPOCHS, loss_fn=torch.nn.CrossEntropyLoss())

    y_score = model(X_valid).cpu()
    y_pred = (y_score > 0.5).int()

    return f1_score(y_valid.cpu(), y_pred, average="macro")

#############################################################################################

def load_and_preprocess_data(path, path2, device, num_classes=2, seed=42):

    # --- Load the source dataset ---
    data = pd.read_csv(path)
    data.fillna(0, inplace=True)
    
    # Separate target and features from source data
    y1 = data["target_label"].values
    data.drop("target_label", axis=1, inplace=True)
    X1 = data.values.astype(np.float32)
    source_columns = data.columns.tolist()
    
   
    data2 = pd.read_csv(path2)
    data2.fillna(0, inplace=True)
    
    y2 = data2["target_label"].values
    data2.drop("target_label", axis=1, inplace=True)
    X2 = data2.values.astype(np.float32)
    

    missing_cols_target = set(source_columns) - set(data2.columns)
    for col in missing_cols_target:
        data2[col] = 0
    data2 = data2[source_columns]
    X2 = data2.values.astype(np.float32)
    
    # --- Label Encoding ---
    # For the source dataset
    if y1.dtype == "object":
        label_encoder = LabelEncoder()
        y1 = label_encoder.fit_transform(y1)
    else:
        label_encoder = None
    y1 = torch.tensor(y1)
    
    # For the target dataset: if label_encoder exists, use it; otherwise, fit a new one.
    if y2.dtype == "object":
        if label_encoder is not None:
            y2 = label_encoder.transform(y2)
        else:
            label_encoder = LabelEncoder()
            y2 = label_encoder.fit_transform(y2)
    y2 = torch.tensor(y2)

    # Split data: first into train and test, then split a validation set from train
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, stratify=y1, test_size=0.2, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1, random_state=seed)

    # Initialize and apply the scaler on features
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train)).to(device)
    X_valid = torch.tensor(scaler.transform(X_valid)).to(device)
    X_test  = torch.tensor(scaler.transform(X_test)).to(device)

    # One-hot encode the labels
    y_train = torch.nn.functional.one_hot(y_train.long(), num_classes=num_classes).to(device).float()
    y_valid = torch.nn.functional.one_hot(y_valid.long(), num_classes=num_classes).to(device).float()
    y_test  = torch.nn.functional.one_hot(y_test.long(), num_classes=num_classes).to(device).float()

    dataset = {
            "train_input": X_train,
            "test_input": X_valid,
            "X_Test": X_test,
            "train_label": y_train,
            "test_label": y_valid,
            "y_Test": y_test,        
        }
    
    ############################################################
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, stratify=y2, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1, random_state=42)

    # Initialize and apply the scaler on features
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train)).to(device)
    X_valid = torch.tensor(scaler.transform(X_valid)).to(device)
    X_test  = torch.tensor(scaler.transform(X_test)).to(device)

    # One-hot encode the labels
    y_train = torch.nn.functional.one_hot(y_train.long(), num_classes=num_classes).to(device).float()
    y_valid = torch.nn.functional.one_hot(y_valid.long(), num_classes=num_classes).to(device).float()
    y_test  = torch.nn.functional.one_hot(y_test.long(), num_classes=num_classes).to(device).float()

    # Determine shapes
    input_shape = X_train.shape[1]
    output_shape = y_train.shape[1]

    dataset2 = {
            "train_input": X_train,
            "test_input": X_valid,
            "X_Test": X_test,
            "train_label": y_train,
            "test_label": y_valid,
            "y_Test": y_test,        
        }

    return dataset, dataset2, source_columns, data2.columns, label_encoder
#############################################################################################

dataset_P = ['Credit-g(CG)', 'Credit-approval(CA)', 'dress-sale(DS)', 'Adult(AD)', 'cylinder-bands(CB)', 'Blastchar(BL)', 'insurance+company(IO)' ,'income (IC)']


for DATA in dataset_P:

    print(DATA)
    path = f"/mnt/gpfs2_4m/scratch/aes255/Afzal/run_Cheb/DATA/{DATA}/Preprocessed"
    parent_path = os.path.dirname(path)
    target_path = os.path.join(parent_path, 'TransferLearning')

    DATA2_PATH = os.path.join(target_path, 'data1', 'data_processed.csv')
    DATA1_PATH = os.path.join(target_path, 'data2', 'data_processed.csv')

    dataset, dataset2, source_columns, target_columns, label_encoder = load_and_preprocess_data(DATA1_PATH, DATA2_PATH, device, num_classes=2)

    input_shape = dataset['train_input'].shape[1]
    output_shape = dataset['y_Test'].shape[1]

    X_valid = dataset['test_input']
    y_valid = dataset['test_label']

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=TRIALS)
    BEST = study.best_params
    #############################################################################################

    seeds = np.random.randint(0, 1000, size=10)

    results = []
    model = prepare_model(input_shape, output_shape, BEST)
    history = fit(model, dataset, steps=EPOCHS, loss_fn=torch.nn.CrossEntropyLoss())
    model = freeze_all_except_last_cheby(model)

    for SEED in seeds:

        dataset, dataset2, source_columns, target_columns, label_encoder = load_and_preprocess_data(DATA1_PATH, DATA2_PATH, device, num_classes=2, seed = SEED)
        input_shape = dataset2["train_input"].shape[1]
        output_shape = dataset2["train_label"].shape[1]   
        
        # Transfer learning on dataset2 (only last Cheby layer trainable)
        history = fit(model, dataset2, steps=EPOCHS, loss_fn=torch.nn.CrossEntropyLoss(), lr=0.01) 

        X_test = dataset2['X_Test'] 
        y_test = dataset2['y_Test']

        y_score = model(X_test).cpu()
        y_pred = (y_score > 0.5).int()

        result = roc_auc_score(y_test.cpu(), y_pred)
        results.append(result)

    top_10_results = sorted(results, reverse=True)[:10]
    average_top_10 = sum(top_10_results) / len(top_10_results)
    print("Average of top 10 ROC-AUC scores: %.4f" % average_top_10)