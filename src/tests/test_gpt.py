import math
import torch
from gpt2.model.config import GPTConfig
from gpt2.model.gpt import GPT


def make_model(context_length=16):
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=26, context_length=context_length,
                    embed_dim=128, num_layers=2, num_heads=4)
    return GPT(config)


def test_forward_output_shape():
    model = make_model()
    token_ids = torch.randint(0, 26, (3, 7))          # batch 3, 7 positions
    logits, loss = model(token_ids)
    # one score per vocab word, at every position
    assert tuple(logits.shape) == (3, 7, 26)
    # no targets -> nothing to score
    assert loss is None


def test_untrained_loss_is_near_ln_vocab():
    model = make_model()
    token_ids = torch.randint(0, 26, (3, 7))
    targets = torch.randint(0, 26, (3, 7))
    _, loss = model(token_ids, targets)
    # a freshly-initialized model is ~uniform, so loss should sit near ln(vocab_size)
    assert abs(loss.item() - math.log(26)) < 0.3


def test_weight_tying():
    model = make_model()
    # the unembedding reuses the token-embedding matrix (same object in memory)
    assert model.unembed.weight is model.token_embedding.weight


def test_generate_grows_by_max_new_tokens():
    model = make_model()
    seed = torch.randint(0, 26, (1, 4))
    out = model.generate(seed, max_new_tokens=10)
    assert out.shape[1] == 4 + 10
    # the original seed is left untouched at the front
    assert torch.equal(out[:, :4], seed)
    # every produced id is a valid letter
    assert int(out.max()) < 26 and int(out.min()) >= 0


def test_generate_crops_to_context_length():
    # a seed LONGER than context_length must still work (line 48 crops it before forward)
    model = make_model(context_length=16)
    long_seed = torch.randint(0, 26, (1, 20))   # 20 > 16
    out = model.generate(long_seed, max_new_tokens=5)
    assert out.shape[1] == 25


def test_generate_temperature_and_top_k_run():
    model = make_model()
    seed = torch.randint(0, 26, (1, 4))
    out = model.generate(seed, max_new_tokens=5, temperature=0.7, top_k=3)
    assert tuple(out.shape) == (1, 9)


def test_generate_is_reproducible_with_fixed_seed():
    model = make_model()
    seed = torch.randint(0, 26, (1, 4))
    torch.manual_seed(42); a = model.generate(seed, max_new_tokens=6)
    torch.manual_seed(42); b = model.generate(seed, max_new_tokens=6)
    assert torch.equal(a, b)
