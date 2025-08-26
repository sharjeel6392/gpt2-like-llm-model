import torch
from src.loss.calc_loss_batch import calc_loss_batch
def calc_loss_loader(data_loader, model, device, num_batches = None) -> float:
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch,
                target_batch,
                model,
                device
            )
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

def total_loss(model, train_loader, validation_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        validation_loss = calc_loss_loader(validation_loader, model, device)

    print(f'Training loss: {train_loss}')
    print(f'Validation loss: {validation_loss}')