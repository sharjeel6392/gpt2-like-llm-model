from src.constants import TOKENIZER
import tiktoken

def token_ids_to_text(token_ids, tokenizer) -> str:
    """
    Convert a list of token IDs back to text using the provided tokenizer.

    Parameters
    ----------
    - token_ids: List[int]
        A list of token IDs to be converted back to text.
        
    Returns
    -------
    - text: str
        The decoded text corresponding to the input token IDs.
    """
    flat = token_ids.squeeze(0)
    text = tokenizer.decode(flat.tolist())
    return text