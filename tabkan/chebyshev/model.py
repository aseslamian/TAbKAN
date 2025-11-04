import torch
import torch.nn as nn
from ..models import KAN  # Import the base KAN class from your library

class ChebyKANLayer(nn.Module):
    """
    A KAN-style layer using Chebyshev polynomials as the learnable activation functions.
    This is inspired by the original KAN paper but replaces B-splines with
    a basis of Chebyshev polynomials, which are well-suited for function approximation.
    """
    def __init__(self, input_dim, output_dim, degree):
        """
        Initializes the ChebyKANLayer.

        Args:
            input_dim (int): The number of input features.
            output_dim (int): The number of output features.
            degree (int): The degree of the Chebyshev polynomials.
        """
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        # Trainable coefficients for the Chebyshev polynomial expansion.
        # Shape: (input_dim, output_dim, degree + 1)
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        # Initialize coefficients with a small standard deviation for stability.
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        """
        Forward pass of the ChebyKANLayer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Reshape input to (batch_size, input_dim)
        x = x.reshape(-1, self.inputdim)

        # Normalize input to the [-1, 1] range, which is the domain of Chebyshev polynomials.
        x = torch.tanh(x)

        # Initialize a tensor to store the Chebyshev polynomial values T_0, T_1, ..., T_degree.
        # Shape: (batch_size, input_dim, degree + 1)
        cheby = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)

        # Calculate polynomial values using the recurrence relation: T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)
        if self.degree > 0:
            cheby[:, :, 1] = x  # T_1(x) = x

        for i in range(2, self.degree + 1):
            # We .clone() the tensors on the right side to prevent an inplace modification
            # error during backpropagation. This is crucial for PyTorch's autograd engine.
            cheby[:, :, i] = 2 * x * cheby[:, :, i - 1].clone() - cheby[:, :, i - 2].clone()

        # Compute the layer's output by performing a tensor contraction (einsum).
        # This multiplies each polynomial value T_i(x) by its corresponding learned coefficient
        # and sums them up for each input-output neuron pair.
        # 'b' is batch, 'i' is input dim, 'd' is degree, 'o' is output dim.
        # Result shape: (batch_size, output_dim)
        y = torch.einsum('bid,iod->bo', cheby, self.cheby_coeffs)

        # Reshape to ensure the output dimension is correctly set
        return y.view(-1, self.outdim)


class ChebyshevKAN(KAN):
    """
    A full Kolmogorov-Arnold Network built from ChebyKANLayer modules.
    This model inherits from the base KAN class, giving it .fit() and .tune() methods.
    """
    def __init__(self, layers, orders):
        """
        Initializes the ChebyshevKAN model.

        Args:
            layers (list of int): A list defining the architecture, e.g., [input_dim, hidden1, hidden2, output_dim].
            orders (list of int): A list of polynomial degrees for each corresponding ChebyKANLayer.
                                  The length should be len(layers) - 2.
        """
        super(ChebyshevKAN, self).__init__()

        # Store configuration for inspection
        self.layers_config = layers
        self.orders_config = orders

        self.layers = nn.ModuleList()

        # Create a ChebyKANLayer for each hidden layer transition
        for i in range(len(layers) - 2):
            if i >= len(orders):
                raise ValueError(f"Not enough 'orders' provided. Expected {len(layers) - 2}, got {len(orders)}.")

            self.layers.append(
                ChebyKANLayer(input_dim=layers[i], output_dim=layers[i + 1], degree=orders[i])
            )

        # Add a standard Linear layer as the final classifier head
        self.layers.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        """
        Forward pass of the ChebyshevKAN model.

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
        Computes feature importance based on the L1 norm of Chebyshev coefficients
        for a specified layer. High-magnitude coefficients imply a feature has a
        stronger influence on the output of that layer.

        Args:
            layer_index (int): The index of the ChebyKANLayer to analyze.

        Returns:
            torch.Tensor: A 1D tensor of shape (input_dim,) where each element
                          is the importance score for the corresponding feature.
        """
        if not isinstance(self.layers[layer_index], ChebyKANLayer):
            raise TypeError(f"Layer {layer_index} is not a ChebyKANLayer. Cannot compute feature importance.")

        # Importance is the L1 norm of coefficients across all output dimensions and polynomial degrees.
        coeffs = self.layers[layer_index].cheby_coeffs
        feature_importance = torch.sum(torch.abs(coeffs), dim=(1, 2))
        return feature_importance
