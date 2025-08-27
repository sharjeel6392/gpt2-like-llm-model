import torch
from src.helper._text_to_token_id import text_to_token_ids
from src.helper._generate_text_1 import generate_text_simple
from src.helper._token_id_to_text import token_ids_to_text

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(text=start_context, tokenizer=tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model = model, idx = encoded, max_new_tokens= 50, content_size= context_size)
    decoded_text = token_ids_to_text(token_ids=token_ids, tokenizer=tokenizer)
    print(decoded_text.replace('\n', ''))
    model.train()