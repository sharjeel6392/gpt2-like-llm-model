import tiktoken
from gpt_model import GPTMODEL
from src.constants import *
import torch

tokenizer = tiktoken.get_encoding("gpt2")
batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

print(f'pre-stacking. Batch: \n{batch}')

batch = torch.stack(batch, dim = 0)

print(f'post-stacking. Batch: \n{batch}')

torch.manual_seed(123)
model = GPTMODEL()
logits = model(batch)
print(f'Shape of the output: {logits.shape}')
print(f'Logits: \n{logits}')

