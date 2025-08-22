import torch
import torch.nn as nn
from src.constants import vocab_size, embedding_dim, context_length, drop_rate, n_layers
from transformer_block import DummyTransformerBlock
from layer_normalization import LayerNormalize

class GPTMODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = nn.Embedding(context_length, embedding_dim)
        self.drop_emb = nn.Dropout(drop_rate)
        
        self.transformer_blocks = nn.Sequential(*[DummyTransformerBlock() for _ in range(n_layers)])

        self.final_norm = LayerNormalize(embedding_dim)
        self.out_head = nn.Linear(embedding_dim, vocab_size, bias= False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device= in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        
        logits = self.out_head(x)
        return logits
    