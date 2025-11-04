# ===== ./tabkan/rkan_wrapper/model.py =====
import torch.nn as nn
from rkan.torch import JacobiRKAN as ExtJacobiRKAN, PadeRKAN as ExtPadeRKAN
from ..models import KAN
import torch

class JacobiRKAN(KAN):
    def __init__(self, layers, orders):
        """
        Initializes the JacobiRKAN model.

        Args:
            layers (list of int): A list of integers representing the number of neurons in each layer.
            orders (list of int): A list of integers representing the order of the Jacobi polynomials for each layer.
        """
        super(JacobiRKAN, self).__init__()

        # Store the configuration so we can easily re-create the model
        self.layers_config = layers
        self.orders_config = orders


        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(ExtJacobiRKAN(orders[i]))

    def forward(self, x):
        """
        Forward pass of the JacobiRKAN model.

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
        Computes feature importance for JacobiRKAN.
        Importance is the L1 norm of the weights of the preceding linear layer.
        """
        linear_layer_index = 2 * layer_index
        if linear_layer_index >= len(self.layers) or not isinstance(self.layers[linear_layer_index], nn.Linear):
            raise IndexError(f"Could not find a corresponding Linear layer for KAN block at index {layer_index}.")

        linear_layer = self.layers[linear_layer_index]
        importance = torch.sum(torch.abs(linear_layer.weight.T), dim=1)
        return importance


class PadeRKAN(KAN):
    def __init__(self, layers, orders1, orders2):
        """
        Initializes the PadeRKAN model.

        Args:
            layers (list of int): A list of integers representing the number of neurons in each layer.
            orders1 (list of int): A list of integers representing the order of the numerator polynomials for each layer.
            orders2 (list of int): A list of integers representing the order of the denominator polynomials for each layer.
        """
        super(PadeRKAN, self).__init__()

        # Store the configuration so we can easily re-create the model
        self.layers_config = layers
        self.orders1_config = orders1 # Corrected from 'orders'
        self.orders2_config = orders2 # Corrected from 'orders'

        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(ExtPadeRKAN(orders1[i], orders2[i]))

    def forward(self, x):
        """
        Forward pass of the PadeRKAN model.

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
        Computes feature importance for JacobiRKAN.
        Importance is the L1 norm of the weights of the preceding linear layer.
        """
        linear_layer_index = 2 * layer_index
        if linear_layer_index >= len(self.layers) or not isinstance(self.layers[linear_layer_index], nn.Linear):
            raise IndexError(f"Could not find a corresponding Linear layer for KAN block at index {layer_index}.")

        linear_layer = self.layers[linear_layer_index]
        importance = torch.sum(torch.abs(linear_layer.weight.T), dim=1)
        return importance
