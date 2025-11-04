# ===== ./tabkan/fourier/model.py =====
import torch
import torch.nn as nn
from .layer import NaiveFourierKANLayer
from ..models import KAN

class FourierKAN(KAN):
    def __init__(self, layers, gridsizes):
        super(FourierKAN, self).__init__()
        self.layers_config = layers
        self.gridsizes_config = gridsizes
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 2):
            self.layers.append(NaiveFourierKANLayer(layers[i], layers[i + 1], gridsize=gridsizes[i]))
        self.layers.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_feature_importance(self, layer_index=0):
        """
        Computes feature importance based on the L1 norm of Fourier coefficients
        for a specified layer.

        Args:
            layer_index (int): The index of the NaiveFourierKANLayer to analyze.

        Returns:
            torch.Tensor: A tensor representing the importance of each input feature.
        """
        if not isinstance(self.layers[layer_index], NaiveFourierKANLayer):
            raise TypeError(f"Layer {layer_index} is not a NaiveFourierKANLayer.")
        
        # Importance is L1 norm of sin and cos coefficients
        coeffs = self.layers[layer_index].fouriercoeffs
        feature_importance = torch.sum(torch.abs(coeffs), dim=(0, 1, 3))
        return feature_importance