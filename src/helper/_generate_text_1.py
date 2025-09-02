import torch
import torch.nn as nn
def generate_text_simple(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, 
                         content_size: int, temperature: float =0.0, top_k:int = 0, eos_id = None):
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
    - temperature: float
        Temperature scaling factor
    - top_k: int
        number of samples to be included; top_k sampling
    - eos_id: int
        end-of-squence id. Specifies a premature termination condition

    
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

        # top_k sampling
        if top_k > 0:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        # temprature scaling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples= 1)
        else:
            probs = torch.softmax(logits, dim = -1)
            idx_next = torch.argmax(probs, dim = -1, keepdim = True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim = 1)
    return idx
