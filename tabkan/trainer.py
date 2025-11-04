# ===== ./tabkan/trainer.py =====
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import LBFGS

def fit_lbfgs(model, dataset, steps=100, loss_fn=None, lr=1., batch=-1):
    """
    A centralized training function using the L-BFGS optimizer.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataset (dict): A dictionary containing the training and testing data.
        steps (int): The number of training steps.
        loss_fn: The loss function.
        lr (float): The learning rate.
        batch (int): The batch size.

    Returns:
        dict: A dictionary containing the training and testing losses.
    """
    # Filter for only the parameters that require gradients.
    # This is crucial for fine-tuning, as it ensures the optimizer
    # only "sees" the unfrozen layers (e.g., the classifier head).
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Check if there are any parameters to train.
    if not trainable_params:
        print("Warning: No trainable parameters found. Skipping training.")
        return {"train_loss": [], "test_loss": []}

    optimizer = LBFGS(trainable_params, lr=lr, history_size=10, line_search_fn="strong_wolfe",
                      tolerance_grad=1e-32, tolerance_change=1e-32)

    results = {"train_loss": [], "test_loss": []}
    pbar = tqdm(range(steps), desc="Training with L-BFGS", ncols=100, leave=False)

    train_inputs = dataset["train_input"]
    train_labels = dataset["train_label"]
    test_inputs = dataset["test_input"]
    test_labels = dataset["test_label"]

    if batch == -1 or batch > train_inputs.shape[0]:
        batch_size = train_inputs.shape[0]
        batch_size_test = test_inputs.shape[0]
    else:
        batch_size = batch
        batch_size_test = batch

    for _ in pbar:
        train_id = np.random.choice(train_inputs.shape[0], batch_size, replace=False)
        test_id = np.random.choice(test_inputs.shape[0], batch_size_test, replace=False)

        # Use a list to hold loss for access outside closure scope.
        # This is a common pattern when using LBFGS in PyTorch, as the
        # closure function needs to be able to modify a variable from
        # the outer scope.
        current_loss = [None]

        def closure():
            optimizer.zero_grad()
            pred = model(train_inputs[train_id])
            loss = loss_fn(pred, train_labels[train_id])
            loss.backward()
            current_loss[0] = loss.item()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            test_loss = loss_fn(model(test_inputs[test_id]), test_labels[test_id])

        train_loss_sqrt = np.sqrt(current_loss[0])
        test_loss_sqrt = torch.sqrt(test_loss).cpu().item()

        results['train_loss'].append(train_loss_sqrt)
        results['test_loss'].append(test_loss_sqrt)

        pbar.set_description(f"| train_loss: {train_loss_sqrt:.2e} | test_loss: {test_loss_sqrt:.2e} ")

    return results
