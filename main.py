import torch
from src.utility.feed_forward import FeedForward
from src.constants import *
from src.utility.multi_head_attention import MultiHeadAttention
from src.utility.generate_text import generate_text
import tiktoken
# ================================================================================
# Testing the FeedForward module
# if __name__ == '__main__':
#     ffn = FeedForward()
#     x = torch.randn(2,3, embedding_dim)  # Example input: batch_size=2, seq_len=3
#     out = ffn(x)
#     print(out.shape)

# # ================================================================================
# # Testing the MultiHeadAttention module
#     torch.manual_seed(123)

#     inputs = torch.tensor(
#         [
#             [0.43, 0.15, 0.89], # Your     (x^1)
#             [0.55, 0.87, 0.66], # journey  (x^2)
#             [0.57, 0.85, 0.64], # starts   (x^3)
#             [0.22, 0.58, 0.33], # with     (x^4)
#             [0.77, 0.25, 0.10], # one      (x^5)
#             [0.05, 0.80, 0.55] # step      (x^6)
#         ]
#     )
#     batch = torch.stack((inputs, inputs), dim = 0)
#     context_length = batch.shape[1]
#     d_in, d_out = inputs.shape[1], 4
#     num_heads = 2
#     multi_head = MultiHeadAttention(
#         d_in= d_in,
#         d_out= d_out,
#         context_length= context_length,
#         dropout= 0.0,
#         num_heads= num_heads
#     )

#     context_vectors = multi_head(batch)
#     print(f'Length of context vector: {context_vectors.shape}')

# ================================================================================

# Testing the GPT model

# torch.manual_seed(123)
# model = GPTMODEL()
# batch = torch.tensor(
#     [
#         [6109, 3626, 6100, 345],
#         [6109, 1110, 6622, 257]
#     ]
# )
# out = model(batch)

# print(f'Input batch:\n{batch}')
# print(f'\nLogits shape: {out.shape}')
# print(f'\nLogits:\n{out}')
# total_params = sum(p.numel() for p in model.parameters())
# print(f'\nTotal number of parameters in the model: {total_params}')

# total_size_bytes = total_params * 4
# total_size_mb = total_size_bytes / (1024 ** 2)
# print(f'Total model size: {total_size_mb:.2f} MB')

# ================================================================================

# Generating text using the model


# start_context = "Hello, I am"
# decoded_text = generate_text(
#     start_context= start_context,
#     max_new_tokens = 6,
# )

# print(f'Decoded text: \n{decoded_text}')
# ===============================================================================

from src.components.data_ingestion import read_and_load_data
from src.utility.gpt_model import GPTMODEL
from src.loss.calc_loss_loader import total_loss

model = GPTMODEL()
file_path = './data/the_verdict.txt'
train_loader, validation_loader = read_and_load_data(file_path)

print('Train loader: ')
for x, y in train_loader:
    print(x.shape, y.shape)

print('Validation loader: ')
for x, y in validation_loader:
    print(x.shape, y.shape)

total_loss(model, train_loader, validation_loader)
