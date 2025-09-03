from src.constants import context_length
from src.utility.gpt_model import GPTMODEL
from src.helper._generate_text_1 import generate_text_simple
from src.helper._text_to_token_id import text_to_token_ids
from src.helper._token_id_to_text import token_ids_to_text
from src.components.load_weights_into_gpt import get_params


def generate_text(start_context, tokenizer):
    """
    Generate text using the model by predicting one token at a time.
    
    Args:
        start_context: The initial text context to start generation from.
        model: The trained language model.
        idx: Tensor of shape (1, sequence_length) containing the initial input tokens.
        max_new_tokens: The number of new tokens to generate.    
    """
    encoded_tensor = text_to_token_ids(start_context, tokenizer)
    model = GPTMODEL()
    model.eval()
    out = generate_text_simple(
        model = model,
        idx = encoded_tensor,
        max_new_tokens = 6,
        content_size = context_length,
        top_k=50,
        temperature=1.5
    )
    decoded_text = token_ids_to_text(out[0].tolist(), tokenizer)

    return decoded_text