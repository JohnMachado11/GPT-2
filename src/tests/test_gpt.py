import math
import pytest
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
    # a seed LONGER than context_length must still work (generate crops it to the last
    # context_length tokens before each forward pass)
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


def test_generate_stops_at_stop_token():
    # A real model here would rarely emit any particular token, so stub the forward
    # pass to always predict `stop_id`: generation must halt on the FIRST new token
    # instead of running all 50.
    model = make_model()
    stop_id = 7

    def always_predicts_stop(token_ids, targets=None):
        logits = torch.full((*token_ids.shape, model.config.vocab_size), -1e9)
        logits[:, :, stop_id] = 0.0
        return logits, None

    model.forward = always_predicts_stop
    prompt = torch.randint(0, 26, (1, 4))

    stopped = model.generate(prompt, max_new_tokens=50, stop_at_token=stop_id)
    assert stopped.shape[1] == 5                  # 4 prompt tokens + the stop token
    assert stopped[0, -1].item() == stop_id       # the stop token is kept, not dropped

    # without stop_at_token it must run the full length
    full = model.generate(prompt, max_new_tokens=50)
    assert full.shape[1] == 54


def test_top_k_1_and_near_zero_temperature_both_mean_always_pick_the_likeliest():
    # not just "it runs" -- with top_k=1, or a temperature near 0, generation must equal greedy decoding
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=26, context_length=16, embed_dim=32, num_layers=1, num_heads=2))
    model.eval()
    prompt = torch.randint(0, 26, (1, 4))
    greedy = prompt
    for _ in range(8):
        next_id = model(greedy)[0][:, -1].argmax(-1, keepdim=True)
        greedy = torch.cat([greedy, next_id], dim=1)
    for seed in range(5):                       # whatever the random state, the answer can't change
        torch.manual_seed(seed)
        assert torch.equal(model.generate(prompt, 8, top_k=1), greedy)
        torch.manual_seed(seed)
        assert torch.equal(model.generate(prompt, 8, temperature=1e-4), greedy)


def test_every_dropout_spot_runs():
    # GPT-2 drops out in 3 places per block (attention weights, attention output, feed-forward output)
    # plus once on the embeddings
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=50, context_length=8, embed_dim=16, num_layers=2, num_heads=2, dropout=0.1))
    calls = []
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.register_forward_hook(lambda module, inputs, output: calls.append(module))
    model(torch.zeros(1, 8, dtype=torch.long))
    assert len(calls) == 1 + 3 * model.config.num_layers


def test_gpt2_weight_initialization():
    # std 0.02 everywhere, except the two layers that write into the residual stream, which are scaled
    # down by 1/sqrt(2 * num_layers) so the stream doesn't grow with depth; biases start at zero
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=1000, context_length=64, embed_dim=256, num_layers=4, num_heads=4))
    residual_scale = 0.02 / math.sqrt(2 * 4)
    for block in model.blocks:
        assert block.attn.qkv.weight.std().item() == pytest.approx(0.02, rel=0.05)
        assert block.ff.expand.weight.std().item() == pytest.approx(0.02, rel=0.05)
        assert block.attn.out_proj.weight.std().item() == pytest.approx(residual_scale, rel=0.05)
        assert block.ff.contract.weight.std().item() == pytest.approx(residual_scale, rel=0.05)
        assert torch.all(block.attn.qkv.bias == 0)
    assert model.token_embedding.weight.std().item() == pytest.approx(0.02, rel=0.05)


def test_config_refuses_an_embed_dim_that_heads_cant_split_evenly():
    with pytest.raises(ValueError, match="divide evenly"):
        GPTConfig(vocab_size=50, context_length=8, embed_dim=30, num_layers=1, num_heads=4)
