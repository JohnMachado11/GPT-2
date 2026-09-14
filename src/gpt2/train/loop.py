import torch

from gpt2.train.schedule import learning_rate_at
from gpt2.train.data import get_batch


@torch.no_grad()
def estimate_loss(model, token_data, batch_size, context_length, device, num_batches=20):
    model.eval()
    total = 0.0
    for _ in range(num_batches):
        token_ids, targets = get_batch(token_data, batch_size, context_length, device)
        logits, loss = model(token_ids, targets)
        total += loss.item()
    model.train()
    return total / num_batches


def train(
    model,
    optimizer,
    token_data,
    batch_size,
    context_length,
    num_steps,
    max_learning_rate,
    warmup_steps,
    min_learning_rate,
    val_data=None,
    eval_interval=100,
    device="cpu"
):
    model.train()
    history = {"steps": [], "train_loss": [], "val_steps": [], "val_loss": []}
    for step in range(num_steps):
        token_ids, targets = get_batch(token_data, batch_size, context_length, device)  # 0. fresh random batch
        logits, loss = model(token_ids, targets)          # 1. forward -> loss
        optimizer.zero_grad()                             # 2. clear stale gradients
        loss.backward()                                   #    fill fresh gradients (chain rule)
        learning_rate = learning_rate_at(                 # 3. this step's learning rate
            step, max_learning_rate, warmup_steps, num_steps, min_learning_rate
        )
        optimizer.learning_rate = learning_rate           #    hand it to AdamW
        optimizer.step()                                  # 4. step every weight downhill
        print(f"step {step:4d} | loss {loss.item():.4f} | lr {learning_rate:.5f}")  # 5. report progress
        history["steps"].append(step)
        history["train_loss"].append(loss.item())
        if val_data is not None and step % eval_interval == 0:   # 6. every eval_interval steps, check the held-out set (the overfitting alarm)
            val_loss = estimate_loss(model, val_data, batch_size, context_length, device)
            print(f"  step {step:4d} | val loss {val_loss:.4f}")
            history["val_steps"].append(step)
            history["val_loss"].append(val_loss)
    return history