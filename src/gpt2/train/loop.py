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
    max_gradient_norm,      # clip the total gradient size to this. 1.0 is the standard value:
                            # it's what the GPT-3 paper states ("we clip the global norm of the
                            # gradient at 1.0") and what nanoGPT uses to reproduce GPT-2. The GPT-2
                            # paper itself never published a clipping value.
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
        torch.nn.utils.clip_grad_norm_(                   # 3. cap the TOTAL size of all gradients
            model.parameters(), max_gradient_norm         #    if they add up to more than this, scale
        )                                                 #    them all down -- same direction, shorter
                                                          #    step. Stops one freak batch from
                                                          #    wrecking the weights.
        learning_rate = learning_rate_at(                 # 4. this step's learning rate
            step, max_learning_rate, warmup_steps, num_steps, min_learning_rate
        )
        optimizer.learning_rate = learning_rate           #    hand it to AdamW
        optimizer.step()                                  # 5. step every weight downhill
        print(f"step {step:4d} | loss {loss.item():.4f} | lr {learning_rate:.5f}")  # 6. report progress
        history["steps"].append(step)
        history["train_loss"].append(loss.item())
        if val_data is not None and step % eval_interval == 0:   # 7. every eval_interval steps, check the held-out set (the overfitting alarm)
            val_loss = estimate_loss(model, val_data, batch_size, context_length, device)
            print(f"  step {step:4d} | val loss {val_loss:.4f}")
            history["val_steps"].append(step)
            history["val_loss"].append(val_loss)
    return history