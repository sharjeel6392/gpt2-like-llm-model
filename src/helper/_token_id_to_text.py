import tiktoken
from src.constants import TOKENIZER
def token_ids_to_text(token_ids: list[int]) -> str:
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
    tokenizer = tiktoken.get_encoding(TOKENIZER) # Using GPT-2 tokenizer
    text = tokenizer.decode(token_ids)
    return text