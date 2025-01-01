import torch
import torch.nn as nn
import torch.nn.functional as F
from fckan_interpolation import linear_interpolation

def phi(x, w1, w2, b1, b2, n_sin):
    """
    phi function that integrates sinusoidal embeddings with MLP layers.
    """
    device = x.device  # Ensure tensors are on the same device as input
    omega = (2 ** torch.arange(0, n_sin, device=device)).float().reshape(-1, 1)
    omega_x = F.linear(x, omega, bias=None)
    x = torch.cat([x, torch.sin(omega_x), torch.cos(omega_x)], dim=-1)

    x = F.linear(x, w1, bias=b1)
    x = F.silu(x)
    x = F.linear(x, w2, bias=b2)
    return x

class KANLayer(nn.Module):
    """
    A layer in a Kolmogorov–Arnold Networks (KAN).
    """
    def __init__(self, dim_in, dim_out, fcn_hidden=32, fcn_n_sin=3):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(dim_in, dim_out, fcn_hidden, 1 + fcn_n_sin * 2))
        self.W2 = nn.Parameter(torch.randn(dim_in, dim_out, 1, fcn_hidden))
        self.B1 = nn.Parameter(torch.randn(dim_in, dim_out, fcn_hidden))
        self.B2 = nn.Parameter(torch.randn(dim_in, dim_out, 1))

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fcn_hidden = fcn_hidden
        self.fcn_n_sin = torch.tensor(fcn_n_sin).long()

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)
        nn.init.zeros_(self.B1)
        nn.init.zeros_(self.B2)

    def map(self, x):
        """
        Maps input tensor x through phi function in a vectorized manner.
        """
        device = x.device  # Ensure tensors are on the same device as input
        self.W1 = self.W1.to(device)
        self.W2 = self.W2.to(device)
        self.B1 = self.B1.to(device)
        self.B2 = self.B2.to(device)
        self.fcn_n_sin = self.fcn_n_sin.to(device)

        F_vmap = torch.vmap(
            torch.vmap(phi, (None, 0, 0, 0, 0, None), 0),
            (0, 0, 0, 0, 0, None), 0
        )
        return F_vmap(x.unsqueeze(-1), self.W1, self.W2, self.B1, self.B2, self.fcn_n_sin).squeeze(-1)

    def forward(self, x):
        """
        Forward pass of the KANLayer.
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        batch, dim_in = x.shape
        assert dim_in == self.dim_in

        batch_f = torch.vmap(self.map, 0, 0)
        phis = batch_f(x)  # [batch, dim_in, dim_out]
        return phis.sum(dim=1)

class KANInterpoLayer(nn.Module):
    """
    A layer in Kolmogorov–Arnold Networks with interpolation.
    """
    def __init__(self, dim_in, dim_out, num_x=64, x_min=-2, x_max=2):
        super().__init__()
        self.X = torch.linspace(x_min, x_max, num_x)
        self.Y = nn.Parameter(torch.randn(dim_in, dim_out, num_x))

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.Y)

    def map(self, x):
        """
        Maps input tensor x through interpolation.
        """
        device = x.device  # Ensure tensors are on the same device as input
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)

        F_vmap = torch.vmap(
            torch.vmap(linear_interpolation, (None, None, 0), 0),
            (0, None, 0), 0
        )
        return F_vmap(x.unsqueeze(-1), self.X, self.Y).squeeze(-1)

    def forward(self, x):
        """
        Forward pass of the KANInterpoLayer.
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        batch, dim_in = x.shape
        assert dim_in == self.dim_in

        batch_f = torch.vmap(self.map, 0, 0)
        phis = batch_f(x)  # [batch, dim_in, dim_out]
        return phis.sum(dim=1)

def smooth_penalty(model):
    """
    Computes a smoothness penalty for the model.
    """
    p = 0
    if isinstance(model, KANInterpoLayer):
        dx = model.X[1] - model.X[0]
        grad = model.Y[:, :, 1:] - model.Y[:, :, :-1]
        return torch.norm(grad, 2) / dx

    for layer in model:
        if isinstance(layer, KANInterpoLayer):
            dx = layer.X[1] - layer.X[0]
            grad = layer.Y[:, :, 1:] - layer.Y[:, :, :-1]
            p += torch.norm(grad, 2) / dx
    return p
