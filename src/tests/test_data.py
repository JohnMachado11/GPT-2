import pytest
import torch
from gpt2.train.data import get_batch


def test_batch_has_the_right_shape():
    token_data = torch.arange(100)
    token_ids, targets = get_batch(token_data, batch_size=8, context_length=16)
    assert token_ids.shape == (8, 16)
    assert targets.shape == (8, 16)


def test_targets_are_inputs_shifted_by_one():
    # a clean 0,1,2,...,49 stream makes the shift exact and easy to assert
    token_data = torch.arange(50)
    token_ids, targets = get_batch(token_data, batch_size=5, context_length=8)
    assert torch.equal(targets, token_ids + 1)


def test_windows_stay_in_bounds():
    # the largest index a target can touch is start + context_length; it must
    # never run off the end of the stream, even for the very last valid start
    token_data = torch.arange(20)
    for _ in range(200):  # many draws to stress the random starts
        token_ids, targets = get_batch(token_data, batch_size=4, context_length=6)
        assert token_ids.min() >= 0
        assert targets.max() <= token_data[-1]


def test_respects_device_argument():
    token_data = torch.arange(40)
    token_ids, targets = get_batch(token_data, batch_size=2, context_length=4, device="cpu")
    assert token_ids.device.type == "cpu"
    assert targets.device.type == "cpu"


def test_rejects_stream_too_short_for_context():
    # a window needs context_length + 1 tokens; a shorter stream should raise a clear error,
    # not the cryptic torch.randint negative-range crash
    token_data = torch.arange(10)          # only 10 tokens
    with pytest.raises(ValueError):
        get_batch(token_data, batch_size=2, context_length=16)   # needs >= 17
