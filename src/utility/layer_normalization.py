import torch
import torch.nn as nn
from src.constants import embedding_dim, eps

class LayerNormalize(nn.Module):
    """
    This module normalizes the inputs across the last dimension. This is
    useful for stabilizing the training of deep neural networks, especially
    in models like Transformers.
    """
    def __init__(self):
        super().__init__()
        # Learnable parameters for scaling and shifting
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))
    
    def forward(self, x):
        """
        Applies the Layer Normalization transformation.

        Returns
        -------
        torch.Tensor: The normalized and scaled tensor.
        """
        # Calculate mean and variance along the last dimension
        # 'keepdim=True' ensures the output shape is compatible for broadcasting
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)

        # Normalize the input
        # eps: A small constant to prevent division by zero
        norm_x = (x - mean)/torch.sqrt(var + eps)

        # Apply the learned scale and shift
        return self.scale * norm_x + self.shift