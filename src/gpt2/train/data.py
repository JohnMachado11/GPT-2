import torch


def get_batch(token_data, batch_size, context_length, device="cpu"):
    if len(token_data) < context_length + 1:
        raise ValueError(
            f"need at least context_length+1 = {context_length + 1} tokens, "
            f"but this stream has {len(token_data)}. Add more text or lower context_length."
        )

    highest_start = len(token_data) - context_length - 1
    starts = torch.randint(0, highest_start + 1, (batch_size,))
    token_ids = torch.stack([token_data[s : s + context_length] for s in starts])
    targets = torch.stack([token_data[s + 1 : s + 1 + context_length] for s in starts])
    return token_ids.to(device), targets.to(device)