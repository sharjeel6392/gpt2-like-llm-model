import torch
def generate_text_simple(model, idx, max_new_tokens, content_size):
    """
    Generates text using the provided model by autoregressively predicting the next tokens.
    Parameters
    ----------
    - model: nn.Module
        The language model used for text generation.
    - idx: torch.Tensor
        A tensor of shape (batch_size, seq_len) containing the input token indices.
    - max_new_tokens: int
        The maximum number of new tokens to generate.
    - content_size: int
        The size of the context window for the model.
    
    Returns
    -------
    - idx: torch.Tensor
        A tensor of shape (batch_size, seq_len + max_new_tokens) containing the original and generated token indices.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -content_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim = -1)
        idx_next = torch.argmax(probs, dim = -1, keepdim = True)
        idx = torch.cat((idx, idx_next), dim = 1)
    return idx
