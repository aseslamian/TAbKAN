# ===== ./tabkan/models.py =====
import torch.nn as nn
from .tuner import OptunaTuner
from .trainer import fit_lbfgs
from .transfer_learning import freeze_feature_extractor, fine_tune_grpo

class KANSequential(nn.Module):
    """
    A simple sequential container for KAN layers.
    This container is used to build KAN models with multiple layers.
    """
    def __init__(self, layers):
        """
        Initializes the KANSequential container.

        Args:
            layers (list of torch.nn.Module): A list of KAN layers.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass of the KANSequential container.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x

class KAN(nn.Module):
    """
    Abstract base class for all KAN model variants in the TabKAN framework.
    It provides a consistent API for fitting and tuning models.
    This class is not meant to be instantiated directly. Instead, use one of the
    concrete implementations such as ChebyshevKAN, FourierKAN, etc.
    """
    def __init__(self):
        """
        Initializes the KAN model.
        """
        super(KAN, self).__init__()

    def fit(self, dataset, steps=100, loss_fn=None, lr=1., batch=-1, **kwargs):
        """
        Fits the model using the L-BFGS optimizer by default.

        Args:
            dataset (dict): A dictionary containing the training and validation data.
            steps (int): The number of optimization steps.
            loss_fn (torch.nn.Module): The loss function to use.
            lr (float): The learning rate.
            batch (int): The batch size. If -1, the entire dataset is used as a single batch.
            **kwargs: Additional arguments for the optimizer.

        Returns:
            dict: A dictionary containing the training history.
        """
        return fit_lbfgs(self, dataset, steps, loss_fn, lr, batch, **kwargs)

    @staticmethod
    def tune(model_class, dataset, search_space, n_trials=50, direction="maximize", device='cuda'):
        """
        Tunes hyperparameters for a given KAN model class using Optuna.
        This is a static method, call it like: ChebyshevKAN.tune(...)

        Args:
            model_class (class): The KAN model class to tune.
            dataset (dict): A dictionary containing the training and validation data.
            search_space (dict): A dictionary defining the hyperparameter search space for Optuna.
            n_trials (int): The number of trials to run.
            direction (str): The direction of optimization ('maximize' or 'minimize').
            device (str): The device to use for training ('cuda' or 'cpu').

        Returns:
            dict: A dictionary containing the best hyperparameters found by Optuna.
        """
        # Pass the model CLASS to the tuner, not an instance
        tuner = OptunaTuner(model_class, dataset, search_space, device)
        best_params = tuner.tune(n_trials=n_trials, direction=direction)
        return best_params

    def pretrain(self, dataset, **kwargs):
        """
        Pre-trains the entire network on a source dataset.

        Args:
            dataset (dict): The source dataset.
            **kwargs: Additional arguments for the training function.

        Returns:
            dict: A dictionary containing the training history.
        """
        print("Starting pre-training...")
        # Just call the standard fit method to train all weights
        history = self.fit(dataset, **kwargs)
        self.is_pretrained = True
        print("Pre-training finished.")
        return history

    def finetune(self, dataset, method='standard', **kwargs):
        """
        Fine-tunes the model on a target dataset after pre-training.

        Args:
            dataset (dict): The target dataset.
            method (str): The fine-tuning method. Can be 'standard' (L-BFGS on
                          the last layer) or 'grpo' (Group Relative Policy Optimization).
            **kwargs: Additional arguments for the training function (steps, lr, etc.).
        """
        if not self.is_pretrained:
            raise RuntimeError("Model must be pre-trained before fine-tuning. Call .pretrain() first.")

        # Freeze the feature extractor layers, leaving the final classifier head trainable
        freeze_feature_extractor(self)
        print("Feature extractor layers frozen for fine-tuning.")

        if method == 'standard':
            print("Starting standard fine-tuning (L-BFGS on classifier head)...")
            # Use the standard L-BFGS trainer, which now only affects unfrozen layers
            return self.fit(dataset, **kwargs)
        elif method == 'grpo':
            print("Starting GRPO fine-tuning...")
            # Call the specialized GRPO trainer
            return fine_tune_grpo(self, dataset, **kwargs)
        else:
            raise ValueError(f"Unknown fine-tuning method: {method}. Use 'standard' or 'grpo'.")

