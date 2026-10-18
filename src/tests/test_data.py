import pytest
import torch
from gpt2.train.data import RandomWindows, ShardLoader, get_batch
from gpt2.train.shards import write_shard


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


def make_shards(folder):
    # counting numbers make every token's position obvious
    write_shard(folder / "train_000000.bin", list(range(0, 20)))       # 20 tokens
    write_shard(folder / "train_000001.bin", list(range(100, 115)))    # 15 tokens
    return sorted(folder.glob("train_*.bin"))


def test_random_windows_hands_out_get_batch_windows():
    windows = RandomWindows(torch.arange(100), batch_size=4, context_length=8)
    token_ids, targets = windows.next_batch()
    assert token_ids.shape == (4, 8)
    assert torch.equal(targets, token_ids + 1)


def test_shard_loader_reads_in_order_with_targets_shifted_by_one(tmp_path):
    loader = ShardLoader(make_shards(tmp_path), batch_size=2, context_length=3)
    first_ids, first_targets = loader.next_batch()
    second_ids, _ = loader.next_batch()
    assert first_ids.flatten().tolist() == [0, 1, 2, 3, 4, 5]
    assert torch.equal(first_targets, first_ids + 1)
    assert second_ids.flatten().tolist() == [6, 7, 8, 9, 10, 11]     # picks up exactly where the last stopped
    assert first_ids.dtype == torch.int64                             # uint16 on disk, int64 for the model


def test_shard_loader_moves_to_the_next_shard_then_wraps(tmp_path):
    loader = ShardLoader(make_shards(tmp_path), batch_size=2, context_length=3)
    starts = [loader.next_batch()[0][0, 0].item() for _ in range(6)]
    # shard 0 gives 3 batches (0, 6, 12); its last 2 tokens can't fill a batch -> shard 1 (100, 106) -> wrap
    assert starts == [0, 6, 12, 100, 106, 0]


def test_shard_loader_skips_shards_too_short_for_a_batch(tmp_path):
    paths = make_shards(tmp_path)
    write_shard(tmp_path / "train_000002.bin", [500, 501, 502])
    loader = ShardLoader(sorted(tmp_path.glob("train_*.bin")), batch_size=2, context_length=3)
    assert len(loader.shard_paths) == 2
    with pytest.raises(ValueError):
        ShardLoader([tmp_path / "train_000002.bin"], batch_size=2, context_length=3)


def test_shard_loader_reset_replays_the_same_batches(tmp_path):
    loader = ShardLoader(make_shards(tmp_path), batch_size=2, context_length=3)
    first = loader.next_batch()[0]
    loader.next_batch()
    loader.reset()
    assert torch.equal(loader.next_batch()[0], first)


def test_shard_loader_resumes_exactly_where_it_stopped(tmp_path):
    paths = make_shards(tmp_path)
    straight = ShardLoader(paths, batch_size=2, context_length=3)
    expected = [straight.next_batch()[0] for _ in range(5)]

    before = ShardLoader(paths, batch_size=2, context_length=3)
    got = [before.next_batch()[0] for _ in range(2)]
    after = ShardLoader(paths, batch_size=2, context_length=3)        # a fresh loader, as after a restart
    after.load_state(before.state())
    got += [after.next_batch()[0] for _ in range(3)]

    assert all(torch.equal(a, b) for a, b in zip(expected, got))


def test_shard_loader_reads_shards_in_name_order_whatever_order_it_is_given(tmp_path):
    # a saved position like {"shard": 0} must mean the same file on every run,
    # even if glob hands the paths over in a different order
    paths = make_shards(tmp_path)
    forwards = ShardLoader(paths, batch_size=2, context_length=3)
    backwards = ShardLoader(list(reversed(paths)), batch_size=2, context_length=3)
    assert torch.equal(forwards.next_batch()[0], backwards.next_batch()[0])
