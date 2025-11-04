# ===== ./tabkan/mixer.py =====
import torch.nn as nn
from .models import KANSequential

class MixerLayerKAN(nn.Module):
    """
    A single layer of the KAN Mixer.

    Args:
        num_tokens (int): The number of tokens in the input.
        token_dim (int): The dimension of the token mixing layer's bottleneck.
        channel_dim (int): The dimension of the channel mixing layer.
        kan_layer_class (nn.Module): The class of the KAN layer to use.
        token_kan_kwargs (dict): The keyword arguments for the token mixing KAN layer.
        channel_kan_kwargs (dict): The keyword arguments for the channel mixing KAN layer.
    """
    def __init__(self, num_tokens, token_dim, channel_dim, kan_layer_class, token_kan_kwargs, channel_kan_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(channel_dim)

        # Token mixing: num_tokens -> token_dim -> num_tokens
        token_layers = [
            kan_layer_class(num_tokens, token_dim, **token_kan_kwargs),
            nn.Linear(token_dim, num_tokens)  # A simple linear layer is common for the second part
        ]
        self.token_mixing = KANSequential(token_layers)

        self.norm2 = nn.LayerNorm(channel_dim)

        # Channel mixing: channel_dim -> hidden_dim -> channel_dim
        channel_hidden_dim = channel_dim * 2
        channel_layers = [
            kan_layer_class(channel_dim, channel_hidden_dim, **channel_kan_kwargs),
            kan_layer_class(channel_hidden_dim, channel_dim, **channel_kan_kwargs)
        ]
        self.channel_mixing = KANSequential(channel_layers)

    def forward(self, x):
        # --- Token Mixing ---
        y = self.norm1(x)
        y = y.transpose(1, 2)
        B, C, T = y.shape
        y = y.reshape(B * C, T)   # Reshape to 2D for the mixing block
        y = self.token_mixing(y)
        y = y.reshape(B, C, T).transpose(1, 2) # Reshape back to 3D and transpose
        x = x + y

        # --- Channel Mixing ---
        y = self.norm2(x)
        B, T, C = y.shape
        y = y.reshape(B * T, C)   # Reshape to 2D for the mixing block
        y = self.channel_mixing(y)
        y = y.reshape(B, T, C)    # Reshape back to 3D
        x = x + y
        return x


class KANMixer(nn.Module):
    """
    A KAN Mixer model.

    Args:
        num_features (int): The number of input features (tokens).
        num_classes (int): The number of output classes.
        kan_layer_class (nn.Module): The class of the KAN layer to use for mixing.
        num_layers (int): The number of mixer layers.
        token_dim (int): The hidden dimension for token mixing.
        channel_dim (int): The embedding dimension for each token.
        token_kan_kwargs (dict): The keyword arguments for the token mixing KAN layer.
        channel_kan_kwargs (dict): The keyword arguments for the channel mixing KAN layer.
    """
    def __init__(self, num_features, num_classes, kan_layer_class, num_layers=4, token_dim=64, channel_dim=128,
                 token_kan_kwargs=None, channel_kan_kwargs=None):
        super().__init__()
        self.embedding = nn.Linear(1, channel_dim)
        self.mixer_layers = nn.Sequential(*[
            MixerLayerKAN(num_features, token_dim, channel_dim, kan_layer_class,
                          token_kan_kwargs or {}, channel_kan_kwargs or {})
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(channel_dim)
        self.fc = nn.Linear(channel_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.mixer_layers(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.fc(x)
