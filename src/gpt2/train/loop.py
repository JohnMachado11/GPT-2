import math
import time
from pathlib import Path

import torch

from gpt2.checkpoint import best_val_loss_so_far, load_last, save_best, save_last
from gpt2.train.schedule import learning_rate_at


@torch.no_grad()
def estimate_loss(model, batches, device, num_batches=20, use_bf16=False):
    model.eval()
    batches.reset()         # rewind (a ShardLoader, or a RandomWindows with a seed), so every check
                            # measures the same windows and the numbers are comparable
    total = 0.0
    for _ in range(num_batches):
        token_ids, targets = batches.next_batch()
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=use_bf16):
            loss = model(token_ids, targets)[1]     # keep only the loss: the big logits are freed right away
        total += loss.item()
    model.train()
    return total / num_batches


def run_settings(train_batches, num_steps, max_learning_rate, warmup_steps, min_learning_rate,
                 max_gradient_norm, accumulation_steps):
    # Everything that shapes the run. A resume must use the same values, or the run would quietly change
    # partway through. (The optimizer's own settings are checked separately, and the data is checked by
    # the batch source.) Stored as plain Python numbers: a numpy number would save fine for hours and
    # then fail to load at resume time.
    return {
        "num_steps": int(num_steps),
        "max_learning_rate": float(max_learning_rate),
        "warmup_steps": float(warmup_steps),    # float: e.g. 10% of 415 steps is 41.5
        "min_learning_rate": float(min_learning_rate),
        "max_gradient_norm": float(max_gradient_norm),
        "accumulation_steps": int(accumulation_steps),
        "batch_size": int(train_batches.batch_size),
        "context_length": int(train_batches.context_length),
    }


def duration(seconds):
    # a length of time, short and readable: 42s, 12m, 2.6h
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def train(
    model,
    optimizer,
    train_batches,          # a RandomWindows or ShardLoader (batch_size, context_length, next_batch, state...)
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
    run_dir=None,           # folder to keep checkpoints in (None = don't save); resume if it has a last.pt
    checkpoint_interval=500,    # save last.pt every this many steps (and always at the end)
    print_interval=1,       # print a progress line every this many steps (and always at the last one)
):
    if accumulation_steps < 1:
        raise ValueError(f"accumulation_steps must be at least 1, got {accumulation_steps}")

    if checkpoint_interval < 1:
        raise ValueError(f"checkpoint_interval must be at least 1, got {checkpoint_interval}")

    if eval_interval < 1:
        raise ValueError(f"eval_interval must be at least 1, got {eval_interval}")

    if print_interval < 1:
        raise ValueError(f"print_interval must be at least 1, got {print_interval}")

    if val_batches is not None and getattr(val_batches, "generator", "") is None:
        print("warning: the validation RandomWindows has no seed -- each check scores different random "
              "windows, so the numbers aren't comparable and best.pt is picked partly on noise. "
              "Pass seed=0 (any number).")

    if warmup_steps > num_steps:
        print(f"warning: warmup_steps ({warmup_steps}) is longer than the run ({num_steps} steps) -- "
              f"the learning rate will never reach its max or decay")

    if run_dir is not None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)    # fail NOW if the folder can't be made, not hours in

    model.train()
    history = {"steps": [], "train_loss": [], "val_steps": [], "val_loss": []}
    start_step = 0
    best_val_loss = float("inf")

    settings = run_settings(train_batches, num_steps, max_learning_rate, warmup_steps, min_learning_rate,
                            max_gradient_norm, accumulation_steps)

    # resume: if this run folder already has a last.pt, put everything back and continue from there
    if run_dir is not None and (Path(run_dir) / "last.pt").exists():
        last_step, history = load_last(run_dir, model, optimizer, train_batches, settings)
        start_step = last_step + 1

        if start_step >= num_steps:
            print(f"{Path(run_dir).name} already finished all {num_steps} steps -- nothing to resume")
        else:
            print(f"resuming {Path(run_dir).name} from step {start_step}")

    if run_dir is not None:
        # what a new best has to beat: best.pt's own record. It can be newer than last.pt (a new best
        # right before a disconnect), and it can exist even when last.pt doesn't (a disconnect before
        # the first last.pt save) -- so it's checked on every start, fresh or resumed.
        best_val_loss = best_val_loss_so_far(run_dir, model)

    # for the progress line: the speed counts only the training itself; the time left uses this session's
    # average step, validation and saves included, so it stays honest
    tokens_per_step = train_batches.batch_size * train_batches.context_length * accumulation_steps
    step_width = len(str(num_steps))                     # so the step numbers line up
    session_started = time.perf_counter()
    training_seconds, tokens_since_print = 0.0, 0

    for step in range(start_step, num_steps):
        step_started = time.perf_counter()
        optimizer.zero_grad()                             # 0. clear last step's gradients
        loss_sum = 0.0
        for _ in range(accumulation_steps):               # 1. several micro-batches; their gradients ADD UP
            token_ids, targets = train_batches.next_batch()   #    next batch of training windows
            # bf16 autocast: the matrix math runs in 16-bit when use_bf16 is on; a no-op when it's off
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(token_ids, targets)[1]       #    forward -> loss (the big logits are freed at once)
            (loss / accumulation_steps).backward()        # 2. backward, divided so N micro-batches count as ONE batch
            loss_sum += loss.detach()                     #    running total, for the report
        train_loss = (loss_sum / accumulation_steps).item()
        if not math.isfinite(train_loss):
            raise RuntimeError(f"step {step}: loss became {train_loss} -- training diverged (often a learning rate that's too high)")

        gradient_norm = torch.nn.utils.clip_grad_norm_(   # 3. cap the TOTAL size of all gradients
            model.parameters(), max_gradient_norm         #    if they add up to more than this, scale
        )                                                 #    them all down -- same direction, shorter
                                                          #    step. Stops one freak batch from
                                                          #    wrecking the weights.
        # A NaN/inf gradient -- possible even when the loss looks normal -- would turn EVERY weight into
        # NaN in the update below. Stop first, so the weights and the last checkpoint stay good.
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(
                f"step {step}: gradients became {gradient_norm.item()} -- training diverged. "
                f"Stopped BEFORE updating the weights, so the last checkpoint is still good."
            )

        learning_rate = learning_rate_at(                 # 4. this step's learning rate
            step, max_learning_rate, warmup_steps, num_steps, min_learning_rate
        )
        optimizer.learning_rate = learning_rate           #    hand it to AdamW
        optimizer.step()                                  # 5. step every weight downhill
        training_seconds += time.perf_counter() - step_started    #    (for the speed: training time only)
        tokens_since_print += tokens_per_step

        if step % print_interval == 0 or step == num_steps - 1:      # 6. report progress
            elapsed = time.perf_counter() - session_started
            steps_done = step - start_step + 1                    #    this session's
            seconds_left = elapsed / steps_done * (num_steps - step - 1)
            print(f"step {step:{step_width}d}/{num_steps} | loss {train_loss:.4f} | lr {learning_rate:.5f} | "
                  f"{tokens_since_print / training_seconds:,.0f} tokens/s | "
                  f"{duration(elapsed)} done, ~{duration(seconds_left)} left")
            training_seconds, tokens_since_print = 0.0, 0

        history["steps"].append(step)
        history["train_loss"].append(train_loss)

        # 7. every eval_interval steps -- and at the final step, so the finished model is always
        #    scored -- check the held-out set (the overfitting alarm)
        if val_batches is not None and (step % eval_interval == 0 or step == num_steps - 1):
            val_loss = estimate_loss(model, val_batches, device, use_bf16=use_bf16)
            print(f"  step {step:{step_width}d} | val loss {val_loss:.4f}")
            history["val_steps"].append(step)
            history["val_loss"].append(val_loss)

            if run_dir is not None and val_loss < best_val_loss:    # a new best: keep this model
                best_val_loss = val_loss
                save_best(run_dir, model, step, val_loss)

        # 8. save everything needed to resume: every checkpoint_interval steps, and always at the very end
        if run_dir is not None and ((step + 1) % checkpoint_interval == 0 or step == num_steps - 1):
            save_last(run_dir, model, optimizer, step, train_batches, history, settings)
            print(f"  step {step:{step_width}d} | saved last.pt")

    return history