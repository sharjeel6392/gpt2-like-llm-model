import torch
import torch.nn as nn
from src.constants import embedding_dim, eps

class LayerNormalize(nn.Module):
    def __init__(self, embedding_dim = embedding_dim, eps = eps):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        norm_x = (x - mean)/torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift