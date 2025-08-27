import torch
def text_to_token_ids(text, tokenizer) -> torch.Tensor:
    """
    Convert input text to a list of token IDs using the provided tokenizer.

    Parameters
    ----------
    - text: str
        The input text to be tokenized.

    Returns
    -------
    - encoded_tensor: torch.Tensor
        A tensor containing the token IDs corresponding to the input text.
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor