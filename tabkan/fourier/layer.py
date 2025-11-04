# ===== ./tabkan/fourier/layer.py =====
import torch
import torch.nn as nn
import numpy as np


class NaiveFourierKANLayer(nn.Module):
    """
    A KAN-style layer using Fourier series as the learnable activation functions.
    This layer implements the activation function as a truncated Fourier series,
    where the coefficients of the series are learned parameters.
    """

    def __init__(self, inputdim, outdim, gridsize, addbias=True, smooth_initialization=False):
        """
        Initializes the NaiveFourierKANLayer.

        Args:
            inputdim (int): The number of input features.
            outdim (int): The number of output features.
            gridsize (int): The number of Fourier coefficients to use for each activation function.
            addbias (bool): Whether to add a bias term to the output.
            smooth_initialization (bool): If True, initializes the Fourier coefficients in a way that encourages smoother functions.
        """
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        # The grid_norm_factor is used to scale the initial Fourier coefficients.
        # A larger factor leads to smaller initial coefficients, which can help with training stability.
        grid_norm_factor = (torch.arange(gridsize) + 1) ** 2 if smooth_initialization else np.sqrt(gridsize)
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, outdim, inputdim, gridsize) / (np.sqrt(inputdim) * grid_norm_factor)
        )
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        """
        Forward pass of the NaiveFourierKANLayer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        xshp = x.shape
        outshape = xshp[:-1] + (self.outdim,)

        # Reshape the input tensor to a 2D tensor of shape (batch_size, inputdim).
        x = x.reshape(-1, self.inputdim)

        # Create a grid of Fourier basis functions.
        k = torch.arange(1, self.gridsize + 1, device=x.device).view(1, 1, 1, self.gridsize)
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)

        # Calculate the cosine and sine components of the Fourier series.
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        # Calculate the output of the layer by multiplying the Fourier basis functions by the learned coefficients.
        y = torch.sum(c * self.fouriercoeffs[0:1], dim=(-2, -1))
        y += torch.sum(s * self.fouriercoeffs[1:2], dim=(-2, -1))

        if self.addbias:
            y += self.bias

        # Reshape the output tensor to the original batch shape.
        return y.reshape(outshape)
