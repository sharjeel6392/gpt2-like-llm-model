from src.loss.calc_loss_batch import calc_loss_batch
from src.helper._evaluate_model import evaluate_model
from src.helper._generate_and_print_sample import generate_and_print_sample

def train_model_simple(model, train_loader, val_loader, optimizer,
                       device, num_epochs, eval_freq, eval_iter,
                       start_context, tokenizer):
    train_losses, val_losses, track_token_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch=input_batch, target_batch=target_batch, model=model, device=device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model=model, train_loader=train_loader, 
                                                      val_loader=val_loader, device=device, eval_iter=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(tokens_seen)
                print(f'EP {epoch + 1} (Step {global_step:06d}): Train loss -> {train_loss:.3f}, Val loss -> {val_loss:.3f}')

        generate_and_print_sample(model=model, tokenizer=tokenizer, device=device, start_context=start_context)
    return train_losses, val_losses, track_token_seen