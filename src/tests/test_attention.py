import pytest
import torch
from gpt2.model.attention import CausalSelfAttention


@pytest.mark.parametrize("fast", [False, True])
def test_causal_no_future_leak(fast):
    torch.manual_seed(0)
    attn = CausalSelfAttention(embed_dim=32, num_heads=4, fast=fast)

    x = torch.randn(1, 6, 32)               # 1 sequence, 6 positions, 32 numbers each
    out = attn(x)

    # change ONLY the last position's input, then rerun
    x_changed = x.clone()
    x_changed[:, -1, :] = torch.randn(32)
    out_changed = attn(x_changed)

    # positions BEFORE the last must be identical — they can't see the future
    assert torch.allclose(out[:, :-1, :], out_changed[:, :-1, :], atol=1e-6)
    # the last position MUST change — it saw its own new input
    assert not torch.allclose(out[:, -1, :], out_changed[:, -1, :])


def test_attention_matches_pytorchs_reference_implementation():
    # the same math as PyTorch's own scaled_dot_product_attention (softmax(QK^T / sqrt(head_dim)) V, causal)
    # -- catches a missing or wrong scale, a wrong mask, or heads split the wrong way
    torch.manual_seed(0)
    attention = CausalSelfAttention(embed_dim=32, num_heads=4)
    x = torch.randn(2, 6, 32)
    query, key, value = attention.qkv(x).split(32, dim=-1)
    heads = lambda t: t.view(2, 6, 4, 8).transpose(1, 2)
    reference = torch.nn.functional.scaled_dot_product_attention(heads(query), heads(key), heads(value), is_causal=True)
    reference = attention.out_proj(reference.transpose(1, 2).reshape(2, 6, 32))
    assert torch.allclose(attention(x), reference, atol=1e-6)


@pytest.mark.parametrize("training", [False, True])
def test_fast_attention_gives_the_same_output_and_gradients_as_the_step_by_step_one(training):
    # (in float64, so any real difference would show far above rounding)
    torch.manual_seed(0)
    slow = CausalSelfAttention(embed_dim=32, num_heads=4, context_length=16).double()
    fast = CausalSelfAttention(embed_dim=32, num_heads=4, context_length=16, fast=True).double()
    fast.load_state_dict(slow.state_dict())
    slow.train(training)
    fast.train(training)
    x = torch.randn(2, 16, 32, dtype=torch.float64)
    slow_out, fast_out = slow(x), fast(x)
    slow_out.sum().backward()
    fast_out.sum().backward()
    assert torch.allclose(slow_out, fast_out, atol=1e-12)
    assert all(torch.allclose(a.grad, b.grad, atol=1e-12) for a, b in zip(slow.parameters(), fast.parameters()))


def test_fast_attention_drops_out_while_training_and_not_after():
    torch.manual_seed(0)
    attn = CausalSelfAttention(embed_dim=32, num_heads=4, dropout=0.5, fast=True)
    x = torch.randn(1, 8, 32)
    attn.eval()
    assert torch.equal(attn(x), attn(x))                    # no dropout: the same answer every time
    attn.train()
    attn.resid_dropout.p = 0.0                              # leave only the attention-weights dropout
    assert not torch.equal(attn(x), attn(x))                # random weights dropped each time


def test_the_mask_is_lower_triangular_and_not_saved():
    attn = CausalSelfAttention(embed_dim=32, num_heads=4, context_length=16)
    assert torch.equal(attn.causal_mask, torch.tril(torch.ones(16, 16, dtype=torch.bool)))
    assert "causal_mask" not in attn.state_dict()           # so checkpoints, and GPT-2's weights, are unchanged
