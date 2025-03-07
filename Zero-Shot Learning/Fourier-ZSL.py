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
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 64
TRIALS = 2
MAX_DEPTH = 10
MAX_NEURONS = 200

##########################################################################################################################
class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=True, smooth_initialization=False):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        grid_norm_factor = (torch.arange(gridsize) + 1) ** 2 if smooth_initialization else np.sqrt(gridsize)

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                          (np.sqrt(inputdim) * grid_norm_factor))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        y = torch.sum(c * self.fouriercoeffs[0:1], (-2, -1))
        y += torch.sum(s * self.fouriercoeffs[1:2], (-2, -1))
        if self.addbias:
            y += self.bias
        y = torch.reshape(y, outshape)
        return y
    
##########################################################################################
class fKAN(nn.Module):
    def __init__(self, layers, gridsizes):
        super(fKAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 2):
            self.layers.append(NaiveFourierKANLayer(layers[i], layers[i + 1], gridsize=gridsizes[i]))
        self.layers.append(nn.Linear(layers[len(layers) - 2], layers[len(layers) - 1]))

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
    orders = [best[f'orders_layer_{i}'] for i in range(depth)]
    
    return fKAN(layers, orders).to(device)
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
    orders = [trial.suggest_int(f"orders_layer_{i}", 2, 6) for i in range(depth)]
    width = [input_shape] + width + [output_shape]
    model = fKAN(width, orders).to(device)

    history = fit(model, dataset, steps=EPOCHS, loss_fn=torch.nn.CrossEntropyLoss())

    y_score = model(X_valid).cpu()
    y_pred = (y_score > 0.5).int()

    return f1_score(y_valid.cpu(), y_pred, average="macro")
#############################################################################################

def load_and_preprocess_data(path, path2, path3, device, num_classes=2, seed=42):

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

    data3 = pd.read_csv(path3)
    data3.fillna(0, inplace=True)
    
    y3 = data3["target_label"].values
    data3.drop("target_label", axis=1, inplace=True)
    X3 = data3.values.astype(np.float32)
    

    missing_cols_target = set(source_columns) - set(data2.columns)
    for col in missing_cols_target:
        data2[col] = 0
    data2 = data2[source_columns]
    X2 = data2.values.astype(np.float32)

    missing_cols_target = set(source_columns) - set(data3.columns)
    for col in missing_cols_target:
        data3[col] = 0
    data3 = data3[source_columns]
    X3 = data3.values.astype(np.float32)

    ############################################################
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

    # For the target dataset: if label_encoder exists, use it; otherwise, fit a new one.
    if y3.dtype == "object":
        if label_encoder is not None:
            y3 = label_encoder.transform(y3)
        else:
            label_encoder = LabelEncoder()
            y3 = label_encoder.fit_transform(y3)
    y3 = torch.tensor(y3)

    ############################################################
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
    
    ############################################################
    X_train, X_test, y_train, y_test = train_test_split(X3, y3, stratify=y3, test_size=0.2, random_state=42)
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

    dataset3 = {
            "train_input": X_train,
            "test_input": X_valid,
            "X_Test": X_test,
            "train_label": y_train,
            "test_label": y_valid,
            "y_Test": y_test,        
        }

    return dataset, dataset2, dataset3, source_columns, data2.columns, label_encoder
#############################################################################################
def fine_tune_grpo(model, dataset, steps=100, lr=1e-3, batch=-1, num_samples=6, beta=0.01, device='cuda'):
    # Move model to device and set to training mode
    model = model.to(device)
    model.train()

    # Create Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    results = {'train_loss': [], 'test_loss': []}

    # Determine batch sizes
    N_train = dataset['train_input'].shape[0]
    N_test  = dataset['test_input'].shape[0]

    if batch == -1 or batch > N_train:
        batch_size = N_train
        batch_size_test = N_test
    else:
        batch_size = batch
        batch_size_test = min(batch, N_test)

    pbar = tqdm(range(steps), desc='Fine-tuning GRPO', ncols=100)
    for step in pbar:
        # Sample a training batch
        train_idx = np.random.choice(N_train, batch_size, replace=False)
        train_inputs = dataset['train_input'][train_idx].to(device)
        train_labels = dataset['train_label'][train_idx].to(device)

        # Use a mutable container (list) to store the loss value
        train_loss = [None]

        # Closure for optimization
        def closure():
            optimizer.zero_grad()
            logits = model(train_inputs)  # Shape: [batch_size, 2]
            probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
            
            # For binary classification, extract probability of class 1.
            binary_probs = probs[:, 1]  # Shape: [batch_size]

            # Sample predictions using Bernoulli sampling.
            sampled_outputs = []
            for _ in range(num_samples):
                # Sample either 0 or 1 for each input.
                sample = torch.bernoulli(binary_probs).long().unsqueeze(1)  # Shape: [batch_size, 1]
                sampled_outputs.append(sample)
            # Concatenate along dim=1: now shape is [batch_size, num_samples]
            sampled_outputs = torch.cat(sampled_outputs, dim=1)

            # Compute rewards (1 if correct, 0 otherwise)
            
            # labels_2d = train_labels.view(-1, 1)
            labels_2d = train_labels[:, 1].view(-1, 1)  # Shape: [batch_size, 1]
            
            rewards = (sampled_outputs == labels_2d).float()  # 1 if correct, 0 otherwise

            # Compute advantage by subtracting the mean reward (baseline)
            mean_reward = rewards.mean(dim=1, keepdim=True)
            advantages = rewards - mean_reward

            # Compute log probabilities from the model's output
            log_probs = F.log_softmax(logits, dim=1)  # Shape: [batch_size, 2]
            # Gather the log probability of each sampled output
            selected_log_probs = log_probs.gather(1, sampled_outputs)  # Shape: [batch_size, num_samples]

            # Compute policy gradient loss (negative for gradient ascent)
            policy_loss = (-selected_log_probs * advantages).mean()

            # Compute KL divergence penalty
            with torch.no_grad():
                old_logits = model(train_inputs)  # Use the same model for old logits (or load a reference model)
                old_probs = F.softmax(old_logits, dim=1)
            kl_div = compute_kl_divergence(probs, old_probs)

            # Total loss = policy loss + KL penalty
            loss = policy_loss + beta * kl_div
            loss.backward()
            train_loss[0] = loss.item()  # Store the loss value in the mutable container
            return loss

        # Take an optimization step
        optimizer.step(closure)

        # Store the training loss
        results['train_loss'].append(train_loss[0])

        # Evaluate on a test batch
        test_idx = np.random.choice(N_test, batch_size_test, replace=False)
        test_inputs = dataset['test_input'][test_idx].to(device)
        test_labels = dataset['test_label'][test_idx].to(device)
        with torch.no_grad():
            test_logits = model(test_inputs)
            test_probs = F.softmax(test_logits, dim=1)
            binary_test_probs = test_probs[:, 1]
            sampled_test_outputs = []
            for _ in range(num_samples):
                sample_test = torch.bernoulli(binary_test_probs).long().unsqueeze(1)
                sampled_test_outputs.append(sample_test)
            sampled_test_outputs = torch.cat(sampled_test_outputs, dim=1)

            # test_labels_2d = test_labels.view(-1, 1)
            test_labels_2d = test_labels[:,1].view(-1, 1)
            
            test_rewards = (sampled_test_outputs == test_labels_2d).float()
            test_mean_reward = test_rewards.mean(dim=1, keepdim=True)
            test_advantages = test_rewards - test_mean_reward

            test_log_probs = F.log_softmax(test_logits, dim=1)
            test_selected_log_probs = test_log_probs.gather(1, sampled_test_outputs)
            test_loss = (-test_selected_log_probs * test_advantages).mean()

        results['test_loss'].append(test_loss.item())
        pbar.set_description(f"| train_loss: {train_loss[0]:.4f} | test_loss: {test_loss.item():.4f} |")

    return model, results

############################################################################################################
def compute_kl_divergence(new_probs, old_probs):
    # Create categorical distributions
    new_dist = Categorical(probs=new_probs)
    old_dist = Categorical(probs=old_probs)
    
    # Compute KL divergence
    return kl_divergence(new_dist, old_dist).mean()

#############################################################################################
dataset_P = ['Credit-g(CG)', 'Adult(AD)' , 'Credit-g(CG)', 'Blastchar(BL)', 'Credit-approval(CA)', 'cylinder-bands(CB)', 'dress-sale(DS)','income (IC)', 'insurance+company(IO)']
# dataset_P = ['Blastchar(BL)']
# DATA = 'Credit-g(CG)'

for DATA in dataset_P:

    print(DATA)
    # path = f"/mnt/gpfs2_4m/scratch/aes255/Afzal/run_Cheb/DATA/{DATA}/Preprocessed"
    path = f"/home/data3/Ali/Code/KAN/TL-Test/DATA/{DATA}/Preprocessed"

    parent_path = os.path.dirname(path)
    target_path = os.path.join(parent_path, 'ZSL')

    DATA1_PATH = os.path.join(target_path, 'data1', 'data_processed.csv')
    DATA2_PATH = os.path.join(target_path, 'data2', 'data_processed.csv')
    DATA3_PATH = os.path.join(target_path, 'data2', 'data_processed.csv')

    dataset, dataset2, dataset3, source_columns, target_columns, label_encoder = load_and_preprocess_data(DATA1_PATH, DATA2_PATH, DATA3_PATH, device, num_classes=2)

    input_shape = dataset['train_input'].shape[1]
    output_shape = dataset['y_Test'].shape[1]

    X_valid = dataset['test_input']
    y_valid = dataset['test_label']

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=TRIALS)

    BEST = study.best_params
    #############################################################################################
    input_shape = dataset2['train_input'].shape[1]
    output_shape = dataset2['y_Test'].shape[1]

    X_valid = dataset2['test_input']
    y_valid = dataset2['test_label']

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=TRIALS)

    BEST2 = study.best_params
    ##########################################################################################################################

    seeds = np.random.randint(10, 1000, size=10)

    results = []
    model2 = prepare_model(input_shape, output_shape, BEST)
    history = fit(model2, dataset, steps=EPOCHS, loss_fn=torch.nn.CrossEntropyLoss())

    model3 = prepare_model(input_shape, output_shape, BEST2)
    history = fit(model3, dataset, steps=EPOCHS, loss_fn=torch.nn.CrossEntropyLoss())

    for SEED in seeds:
        dataset, dataset2, dataset3, source_columns, target_columns, label_encoder = load_and_preprocess_data(DATA1_PATH, DATA2_PATH, DATA3_PATH, device, num_classes=2, seed=SEED)

        input_shape = dataset3["train_input"].shape[1]
        output_shape = dataset3["train_label"].shape[1]   

        X_test = dataset3['X_Test'] 
        y_test = dataset3['y_Test']

        y_score = model2(X_test).cpu()
        y_pred2 = (y_score > 0.5).int()

        y_score = model3(X_test).cpu()
        y_pred3 = (y_score > 0.5).int()

        y_pred = (y_pred2+y_pred3)/2

        result = roc_auc_score(y_test.cpu(), y_pred)
        results.append(result)

    top_10_results = sorted(results, reverse=True)[:10]
    average_top_10 = sum(top_10_results) / len(top_10_results)
    print("Average of top 10 ROC-AUC scores: %.4f" % average_top_10)

##########################################################################################################################