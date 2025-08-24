from src.constants import context_length
import torch
import tiktoken
from src.components.gpt_model import GPTMODEL
from src.components.generate_text_1 import generate_text_simple


def generate_text(start_context, max_new_tokens):
    """
    Generate text using the model by predicting one token at a time.
    
    Args:
        start_context: The initial text context to start generation from.
        model: The trained language model.
        idx: Tensor of shape (1, sequence_length) containing the initial input tokens.
        max_new_tokens: The number of new tokens to generate.    
    """
    tokenizer = tiktoken.get_encoding("gpt2") # Using GPT-2 tokenizer
    encoded = tokenizer.encode(start_context)
    print(f'Encoded input: \n{encoded}')

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    model = GPTMODEL()
    model.eval()
    out = generate_text_simple(
        model = model,
        idx = encoded_tensor,
        max_new_tokens = 6,
        content_size = context_length
    )
    decoded_text = tokenizer.decode(out.squeeze().tolist())

    return decoded_text