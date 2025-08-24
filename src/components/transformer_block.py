import torch.nn as nn
from src.constants import *
from src.components.multi_head_attention import MultiHeadAttention
from src.components.feed_forward import FeedForward
from src.components.layer_normalization import LayerNormalize

class TransformerBlock(nn.Module):
    """
    A single transformer block consisting of multi-head self-attention and a feed-forward network, backed by
    Layer Normalization and shortcut connections. 
    Attributes
    ----------
    -attn: MultiHeadAttention
        The multi-head self-attention mechanism.
    -ffn: FeedForward
        The feed-forward neural network.
    -norm1: LayerNormalize
        Layer normalization applied before the attention mechanism.
    -norm2: LayerNormalize
        Layer normalization applied before the feed-forward network.
    -drop_shortcut: nn.Dropout
        Dropout applied to the output of attention and feed-forward layers before adding the shortcut connection.
    
    Methods
    -------
    -forward(x):
        Performs the forward pass of the transformer block.
    """
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = embedding_dim,
            d_out = embedding_dim,
            context_length = context_length,
            num_heads = n_heads,
            dropout = drop_rate,
            qkv_bias = qkv_bias
        )
        self.ffn = FeedForward()
        self.norm1 = LayerNormalize()
        self.norm2 = LayerNormalize()
        self.drop_shortcut = nn.Dropout(drop_rate)    

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x