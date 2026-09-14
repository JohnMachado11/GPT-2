import math


def learning_rate_at(
    step,
    max_learning_rate,
    warmup_steps,
    total_steps,
    min_learning_rate,
):
    if step < warmup_steps:
        return max_learning_rate * (step + 1) / warmup_steps
    if step > total_steps:
        return min_learning_rate

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    coefficient = 0.5 * (1 + math.cos(math.pi * progress))
    learning_rate = min_learning_rate + coefficient * (max_learning_rate - min_learning_rate)
    return learning_rate