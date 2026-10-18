import math

import torch

from gpt2.train.schedule import learning_rate_at


@torch.no_grad()
def estimate_loss(model, batches, device, num_batches=20, use_bf16=False):
    model.eval()
    batches.reset()         # start from the same place every time, so val numbers are comparable
    total = 0.0
    for _ in range(num_batches):
        token_ids, targets = batches.next_batch()
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=use_bf16):
            logits, loss = model(token_ids, targets)
        total += loss.item()
    model.train()
    return total / num_batches


def train(
    model,
    optimizer,
    train_batches,          # a RandomWindows or ShardLoader: anything with next_batch()
    num_steps,
    max_learning_rate,
    warmup_steps,
    min_learning_rate,
    max_gradient_norm,      # cap the total gradient size; 1.0 from the GPT-3 paper + nanoGPT's GPT-2 retrain
    val_batches=None,
    eval_interval=100,
    device="cpu",
    use_bf16=False,         # run the forward pass in bf16 (NVIDIA GPUs); weights stay 32-bit
    accumulation_steps=1,   # micro-batches whose gradients add up into each step (1 = off)
):
    if accumulation_steps < 1:
        raise ValueError(f"accumulation_steps must be at least 1, got {accumulation_steps}")

    model.train()
    history = {"steps": [], "train_loss": [], "val_steps": [], "val_loss": []}

    for step in range(num_steps):
        optimizer.zero_grad()                             # 0. clear last step's gradients
        loss_sum = 0.0
        for _ in range(accumulation_steps):               # 1. several micro-batches; their gradients ADD UP
            token_ids, targets = train_batches.next_batch()   #    next batch of training windows
            # bf16 autocast: the matrix math runs in 16-bit when use_bf16 is on; a no-op when it's off
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=use_bf16):
                logits, loss = model(token_ids, targets)  #    forward -> loss
            (loss / accumulation_steps).backward()        # 2. backward, divided so N micro-batches count as ONE batch
            loss_sum += loss.detach()                     #    running total, for the report
        train_loss = (loss_sum / accumulation_steps).item()
        if not math.isfinite(train_loss):
            raise RuntimeError(f"step {step}: loss became {train_loss} -- training diverged (often a learning rate that's too high)")

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

        print(f"step {step:4d} | loss {train_loss:.4f} | lr {learning_rate:.5f}")  # 6. report progress
        history["steps"].append(step)
        history["train_loss"].append(train_loss)

        if val_batches is not None and step % eval_interval == 0:   # 7. every eval_interval steps, check the held-out set (the overfitting alarm)
            val_loss = estimate_loss(model, val_batches, device, use_bf16=use_bf16)
            print(f"  step {step:4d} | val loss {val_loss:.4f}")
            history["val_steps"].append(step)
            history["val_loss"].append(val_loss)

    return history