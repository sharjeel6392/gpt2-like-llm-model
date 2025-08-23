import torch
from src.components.feed_forward import FeedForward
from src.constants import embedding_dim
# Testing the FeedForward module
if __name__ == '__main__':
    ffn = FeedForward()
    x = torch.randn(2,3, embedding_dim)  # Example input: batch_size=2, seq_len=3
    out = ffn(x)
    print(out.shape)