# ===== ./tabkan/transfer_learning.py =====
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence
from tqdm import tqdm
import numpy as np
from torch import nn
from copy import deepcopy

def freeze_layers(model, layers_to_freeze):
    """
    Freezes specified layers of a model.

    Args:
        model (nn.Module): The model.
        layers_to_freeze (list of int): Indices of layers to freeze.
    """
    for i, (name, param) in enumerate(model.named_parameters()):
        if any(f'layers.{idx}.' in name for idx in layers_to_freeze):
            param.requires_grad = False
    return model




def freeze_feature_extractor(model):
    """
    Freezes all layers in the model except for the final layer (the classifier head).
    This is the standard approach for transfer learning feature extraction.
    This version handles native TabKAN, wrapped SplineKAN, and other wrapped
    KANs like FractionalKAN which end with a standard nn.Linear layer.
    """
    # Freeze all parameters by default first
    for param in model.parameters():
        param.requires_grad = False

    layer_list_to_inspect = None

    # Case 1: Handle native models (Chebyshev, Fourier) and wrapped Linear->Activation models (fKAN, rkan)
    if hasattr(model, 'layers') and isinstance(model.layers, nn.ModuleList):
        layer_list_to_inspect = model.layers

    # Case 2: Handle the wrapped SplineKAN model
    elif hasattr(model, 'model') and hasattr(model.model, 'act_fun'):
        # For SplineKAN, the list of layers is inside the .model attribute
        layer_list_to_inspect = model.model.act_fun

    else:
        # If neither structure is found, raise the error
        raise TypeError("Model does not have a recognizable layer structure for freezing (.layers or .model.act_fun).")

    # Now, unfreeze the parameters of the last layer in the identified list
    if layer_list_to_inspect and len(layer_list_to_inspect) > 0:
        last_layer = layer_list_to_inspect[-1]
        for param in last_layer.parameters():
            param.requires_grad = True
    else:
        # This case should ideally not be hit if the model is valid
        raise ValueError("Identified layer list is empty, cannot unfreeze the last layer.")

    return model

def _compute_kl_divergence(p_probs, q_probs):
    """
    Helper function for stable KL divergence calculation.
    It computes the KL divergence between two categorical distributions.

    Args:
        p_probs (torch.Tensor): The probabilities of the first distribution.
        q_probs (torch.Tensor): The probabilities of the second distribution.

    Returns:
        torch.Tensor: The mean KL divergence.
    """
    p_dist = Categorical(probs=p_probs)
    q_dist = Categorical(probs=q_probs)
    return kl_divergence(p_dist, q_dist).mean()

def fine_tune_grpo(model, dataset, steps=100, lr=1e-3, batch=-1, num_samples=6, beta=0.01, device='cpu', **kwargs):
    """
    Fine-tunes a model using Group Relative Policy Optimization (GRPO).
    This method is a form of reinforcement learning that is used to
    fine-tune a model on a target dataset. It is particularly useful
    when the target dataset is small and the model is prone to overfitting.

    Args:
        model (nn.Module): The model to be fine-tuned.
        dataset (dict): A dictionary containing the training and testing data.
        steps (int): The number of training steps.
        lr (float): The learning rate.
        batch (int): The batch size.
        num_samples (int): The number of samples to draw from the policy for each input.
        beta (float): The KL divergence penalty coefficient.
        device (str): The device to use for training ('cuda' or 'cpu').
        **kwargs: Additional arguments.

    Returns:
        dict: A dictionary containing the training history.
    """
    model.to(device)
    model.train() # Set model to training mode

    # Create a deepcopy of the model in its current state to act as the reference policy.
    ref_model = deepcopy(model)
    ref_model.eval()

    # GRPO typically uses Adam, not L-BFGS. We only optimize unfrozen parameters.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)

    results = {'train_loss': [], 'test_loss': []}

    train_inputs = dataset["train_input"].to(device)
    train_labels = dataset["train_label"].to(device) # Expects integer labels for rewards

    if train_labels.ndim > 1 and train_labels.shape[1] > 1: # Convert from one-hot if needed
        train_labels = torch.argmax(train_labels, dim=1)

    N_train = train_inputs.shape[0]
    batch_size = N_train if batch == -1 else min(batch, N_train)

    pbar = tqdm(range(steps), desc='Fine-tuning with GRPO', ncols=100, leave=False)

    for step in pbar:
        # Sample a random training batch
        train_idx = np.random.choice(N_train, batch_size, replace=False)
        inputs = train_inputs[train_idx]
        labels = train_labels[train_idx]

        optimizer.zero_grad()

        # --- Policy Gradient Calculation ---
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)

        # Sample multiple actions (predictions) from the policy
        policy_dist = Categorical(probs=probs)
        sampled_actions = policy_dist.sample((num_samples,)).transpose(0, 1) # Shape: [batch_size, num_samples]

        # Calculate rewards: 1 if prediction is correct, 0 otherwise
        rewards = (sampled_actions == labels.unsqueeze(1)).float()

        # Calculate advantage (baseline is the mean reward over samples)
        baseline = rewards.mean(dim=1, keepdim=True)
        advantages = rewards - baseline

        # Calculate policy loss
        # 1. Get the log probabilities for ALL actions (classes 0 and 1)
        log_probs_all = F.log_softmax(logits, dim=1)  # Shape: [batch_size, num_classes]

        # 2. Use `gather` to select the log probabilities for the actions we actually sampled.
        #    `sampled_actions` needs to be [batch_size, num_samples, 1] for gather.
        selected_log_probs = torch.gather(
            log_probs_all.unsqueeze(1).expand(-1, num_samples, -1), # Expand to match sample count
            2,
            sampled_actions.unsqueeze(-1)
        ).squeeze(-1) # Shape becomes: [batch_size, num_samples]

        # 3. Calculate policy loss using the correctly gathered log probabilities
        policy_loss = -(selected_log_probs * advantages).mean()

        # --- KL Divergence Penalty ---
        with torch.no_grad():
            ref_logits = ref_model(inputs)
            ref_probs = F.softmax(ref_logits, dim=1)

        kl_div = _compute_kl_divergence(probs, ref_probs.detach())

        # --- Total Loss and Optimization Step ---
        total_loss = policy_loss + beta * kl_div
        total_loss.backward()
        optimizer.step()

        results['train_loss'].append(total_loss.item())
        pbar.set_description(f"| grpo_loss: {total_loss.item():.4f}")

    return results
