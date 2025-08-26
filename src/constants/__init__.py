# Vocabulary size
vocab_size = 50257
# Maximum number of input tokens the model can handle via positional embeddings
context_length = 256

# Represents the embedding size. Each embedding is a 768 dimensional vector
embedding_dim = 768

# Indicates the number of attention heads in the multi-head attention mechanism
n_heads = 12

# Specifies the number of transformer blocks in the model
n_layers = 12

# Indicates the intensity of the dropout mechanism to prevent overfitting. 0.1 implies a 10% random drop out of hidden units.
drop_rate = 0.1

# Determines whether to include a bias vector in the Linear layers of the multihead attention for query, key and value computations.
qkv_bias = False


# Small epsilon value for numerical stability in normalization layers
eps = 1e-5

# Tokenizer 
TOKENIZER = "gpt2"

# Train-test split
TRAIN_RATIO = 0.9