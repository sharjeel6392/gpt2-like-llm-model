import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # tokenize the entire text

        token_ids = tokenizer.encode(txt)

        # uses a sliding window to chunk the book into overlapping sequence of max_length

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # returns the total number of rows in the dataset.

    def __len__(self):
        return len(self.input_ids)
    
    # Returns a single row from the dataset.

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

