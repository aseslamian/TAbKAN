# ===== ./tabkan/spline/model.py =====
from kan import KAN as OriginalKAN
from ..models import KAN
import torch

class SplineKAN(KAN):
    def __init__(self, width, grid, k, device='cpu'):
        """
        Initializes the SplineKAN model.

        Args:
            width (list of int): A list of integers representing the number of neurons in each layer.
            grid (int): The number of grid points for the spline.
            k (int): The order of the spline.
            device (str): The device to run the model on.
        """
        super(SplineKAN, self).__init__()
        self.model = OriginalKAN(width=width, grid=grid, k=k, device=device)

    def forward(self, x):
        """
        Forward pass of the SplineKAN model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.model(x)

    def fit(self, dataset, steps=100, loss_fn=None, lr=1., batch=-1, **kwargs):
        """
        Trains the SplineKAN model.

        Args:
            dataset (dict): A dictionary containing the training data.
            steps (int): The number of training steps.
            loss_fn: The loss function.
            lr (float): The learning rate.
            batch (int): The batch size.
            **kwargs: Additional arguments for the original KAN trainer.

        Returns:
            dict: A dictionary containing the training history.
        """
        # The original KAN library has its own trainer, so we call it directly.
        return self.model.fit(dataset, steps=steps, loss_fn=loss_fn, lr=lr, batch=batch, **kwargs)

    def get_feature_importance(self, layer_index=0):
        """
        Computes feature importance for a specified layer of the wrapped pykan model.

        Importance is measured as the L1 norm of the scaling parameters
        (scale_base and scale_sp) for each input feature.

        Args:
            layer_index (int): The index of the layer to analyze. Default is 0.

        Returns:
            torch.Tensor: A tensor representing the importance of each input feature.
        """
        # The 'pykan' library stores its layers in the 'act_fun' attribute list.
        if layer_index >= len(self.model.act_fun):
            raise IndexError(f"Layer index {layer_index} is out of bounds for model with {len(self.model.act_fun)} layers.")

        # Access the specific KANLayer from the underlying pykan model
        kan_layer = self.model.act_fun[layer_index]

        # 'kan_layer.scale_base' shape: (in_dim, out_dim)
        # 'kan_layer.scale_sp' shape: (in_dim, out_dim)

        # Calculate importance from the base function scaling
        # Sum absolute values across the output dimension (dim=1) to get importance per input feature
        base_importance = torch.sum(torch.abs(kan_layer.scale_base), dim=1)

        # Calculate importance from the spline function scaling
        spline_importance = torch.sum(torch.abs(kan_layer.scale_sp), dim=1)

        # Total importance is the sum of the two scaling components
        total_importance = base_importance + spline_importance

        return total_importance
