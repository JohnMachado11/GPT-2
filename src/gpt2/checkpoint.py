from pathlib import Path
from dataclasses import asdict
from datetime import datetime

import torch

from gpt2.model.config import GPTConfig
from gpt2.model.gpt import GPT


def save_checkpoint(model, config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = path
    counter = 2
    while path.exists():
        path = original.with_stem(f"{original.stem}-{counter}")
        counter += 1
    torch.save({"config": asdict(config), "model_state": model.state_dict()}, path)
    print(f"saved checkpoint -> {path}")
    return path


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def checkpoint_name(model, config):
    param_count = sum(p.numel() for p in model.parameters())
    params = f"{param_count / 1e6:.1f}M"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"gpt2_L{config.num_layers}_D{config.embed_dim}_{params}_{timestamp}.pt"