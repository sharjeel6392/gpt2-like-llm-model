import tiktoken
from src.constants import TOKENIZER, TRAIN_RATIO, context_length
from src.helper.data_loader import create_dataloader_v1


# func1: Read data file
# func2: split in train and validation
# func3: create train_loader and validation_loader
# create artifacts = train_loader and validation_loader
# func4: combines all and stores the artifacts

def read_and_load_data(filepath):
    tokenizer = tiktoken.get_encoding(TOKENIZER)
    with open(filepath, 'r', encoding = 'utf-8') as file:
        text_data = file.read()
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print(f'Total characters in the database: {total_characters}')
    print(f'Total number of tokens: {total_tokens}')

    split_index = int(TRAIN_RATIO * len(text_data))
    train_data = text_data[: split_index]
    validation_data = text_data[split_index: ]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size= 2,
        max_length= context_length,
        stride= context_length,
        drop_last= True,
        shuffle= True,
        num_workers= 0
    )

    validation_loader = create_dataloader_v1(
        validation_data,
        batch_size= 2,
        max_length= context_length,
        stride= context_length,
        drop_last= False,
        shuffle= False,
        num_workers= 0
    )

    return train_loader, validation_loader