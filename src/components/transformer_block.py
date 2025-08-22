import torch.nn as nn
from src.constants import *

class DummyTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x