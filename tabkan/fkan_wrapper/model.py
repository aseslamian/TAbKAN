# ===== ./tabkan/fkan_wrapper/model.py =====
import torch.nn as nn
from fkan.torch import FractionalJacobiNeuralBlock as fJNB
from ..models import KAN
import torch

class FractionalKAN(KAN):
    """
    A KAN model that uses Fractional Jacobi Neural Blocks (fJNB) as activation functions.
    This model is a wrapper around the `fkan` library.
    """
    def __init__(self, layers, orders):
        """
        Initializes the FractionalKAN model.

        Args:
            layers (list of int): A list of integers representing the number of neurons in each layer.
            orders (list of int): A list of integers representing the order of the fractional Jacobi polynomials for each layer.
                                  The length of this list should be len(layers) - 2.
        """
        super(FractionalKAN, self).__init__()

        # Store the configuration so we can easily re-create the model
        self.layers_config = layers
        self.orders_config = orders

        self.layers = nn.ModuleList()
        # The model is a sequence of Linear layers followed by fJNB activation functions.
        # The final layer is a Linear layer without an activation function.
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(fJNB(orders[i]))

    def forward(self, x):
        """
        Forward pass of the FractionalKAN model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def get_feature_importance(self, layer_index=0):
        """
        Computes feature importance for a specified layer of the FractionalKAN.

        Since the fJNB block comes after the linear layer, the importance is best
        represented by the magnitude of the weights of the preceding linear layer.

        Args:
            layer_index (int): The index of the KAN block (e.g., 0 for the first one).
                               This corresponds to the linear layer at index `2 * layer_index`.

        Returns:
            torch.Tensor: A tensor representing the importance of each input feature.
        """
        # A KAN block consists of a Linear layer and an fJNB layer.
        # The linear layer that corresponds to the first KAN block is at index 0.
        # The linear layer for the second KAN block is at index 2, etc.
        linear_layer_index = 2 * layer_index

        if linear_layer_index >= len(self.layers) or not isinstance(self.layers[linear_layer_index], nn.Linear):
            raise IndexError(f"Could not find a corresponding Linear layer for KAN block at index {layer_index}.")

        linear_layer = self.layers[linear_layer_index]

        # Importance is the L1 norm of the weights of the linear layer.
        # Weight shape: (out_features, in_features). We want to sum across out_features.
        importance = torch.sum(torch.abs(linear_layer.weight.T), dim=1)

        return importance
