import torch
import torch.nn as nn
from src.constants import vocab_size, embedding_dim, context_length, drop_rate, n_layers
from src.utility.transformer_block import TransformerBlock
from src.utility.layer_normalization import LayerNormalize



class GPTMODEL(nn.Module):
    """
    A simplified Generative Pre-trained Transformer (GPT) model.

    This model processes input sequences by combining token and positional
    embeddings, passing them through a series of transformer blocks, and
    then generating logits for the next token prediction. 

    Attributes
    ----------
    - vocab_size: int
        The size of the vocabulary (number of unique tokens).
    - context_length: int
        The maximum length of input sequences.
    - embedding_dim: int
        The dimensionality of the token and positional embeddings.
    - drop_rate: float
        The dropout rate for the embedding layer.
    - n_layers: int
        The number of transformer blocks in the model.
    """
    def __init__(self):
        super().__init__()

        # Token embeddings to convert input indices into dense vectors
        self.tok_emb = nn.Embedding(vocab_size, embedding_dim)

        # Positional embeddings to encode the position of each token
        self.pos_emb = nn.Embedding(context_length, embedding_dim)

        # Dropout layer for regularization
        self.drop_emb = nn.Dropout(drop_rate)
        
        # A sequence of transformer blocks that form the core of the model
        self.transformer_blocks = nn.Sequential(*[TransformerBlock() for _ in range(n_layers)])

        # A final layer normalization and linear head for output
        self.final_norm = LayerNormalize()
        self.out_head = nn.Linear(embedding_dim, vocab_size, bias= False)

    def forward(self, in_idx):
        """
        Performs the forward pass of the model.

        Parameters
        ----------
        - in_idx:  torch.Tensor
            A tensor of shape (batch_size, seq_len) containing the input token indices.

        Returns
        -------
        - logits: torch.Tensor
            The output logits tensor of shape (batch_size, seq_len, vocab_size),
            representing the model's predictions for the next token in the sequence.
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device= in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        
        logits = self.out_head(x)
        return logits
    