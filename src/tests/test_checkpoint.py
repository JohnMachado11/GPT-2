import torch

from gpt2.model.config import GPTConfig
from gpt2.model.gpt import GPT
from gpt2.checkpoint import save_checkpoint, load_checkpoint, checkpoint_name


def _tiny_model():
    config = GPTConfig(vocab_size=50, context_length=8, embed_dim=16, num_layers=2, num_heads=2)
    return GPT(config), config


def test_save_then_load_reproduces_every_weight(tmp_path):
    model, config = _tiny_model()
    path = save_checkpoint(model, config, tmp_path / "m.pt")
    reloaded = load_checkpoint(path)
    for (name_a, p_a), (name_b, p_b) in zip(model.named_parameters(), reloaded.named_parameters()):
        assert name_a == name_b
        assert torch.equal(p_a, p_b)


def test_reloaded_config_matches(tmp_path):
    model, config = _tiny_model()
    path = save_checkpoint(model, config, tmp_path / "m.pt")
    reloaded = load_checkpoint(path)
    assert reloaded.config.num_layers == config.num_layers
    assert reloaded.config.embed_dim == config.embed_dim
    assert reloaded.config.vocab_size == config.vocab_size


def test_save_never_overwrites(tmp_path):
    model, config = _tiny_model()
    first = save_checkpoint(model, config, tmp_path / "m.pt")
    second = save_checkpoint(model, config, tmp_path / "m.pt")   # same target
    assert first != second              # the guard renamed the second one
    assert first.exists() and second.exists()   # the first was NOT clobbered


def test_checkpoint_name_encodes_arch_and_params():
    model, config = _tiny_model()
    name = checkpoint_name(model, config)
    assert name.startswith("gpt2_L2_D16_")   # layers + embed_dim
    assert name.endswith(".pt")
