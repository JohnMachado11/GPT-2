import importlib.util
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoTokenizer

from gpt2.train.shards import read_shard, write_shard


# src/run isn't a package, so load prepare_data.py straight from its file
_path = Path(__file__).parents[1] / "run" / "prepare_data.py"
_spec = importlib.util.spec_from_file_location("prepare_data", _path)
prepare_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_data)

ENDOFTEXT = 50256


def read_all_shards(folder):
    # validation first, then training shards in order -- the order they were written
    names = sorted(p.name for p in Path(folder).glob("*.bin"))
    names = [n for n in names if n.startswith("val")] + [n for n in names if n.startswith("train")]
    return names, np.concatenate([read_shard(Path(folder) / n) for n in names]).tolist()


def test_shard_round_trip(tmp_path):
    ids = [0, 464, 3981, ENDOFTEXT, 1547]
    write_shard(tmp_path / "s.bin", ids)
    assert read_shard(tmp_path / "s.bin").tolist() == ids
    assert (tmp_path / "s.bin").stat().st_size == 2 * len(ids)      # 2 bytes per token, no header


def test_shard_rejects_ids_too_big_for_uint16(tmp_path):
    with pytest.raises(ValueError):
        write_shard(tmp_path / "s.bin", [70_000])


def test_writer_spills_across_shards_without_losing_tokens(tmp_path, capsys):
    writer = prepare_data.ShardWriter(tmp_path, shard_size=10)
    writer.add(list(range(1, 8)))       # 7 tokens: fits in shard 0
    writer.add(list(range(8, 26)))      # 18 more: fills shard 0, fills shard 1, 5 spill into shard 2
    writer.write()                      # the last, partly filled shard
    writer.write()                      # nothing left: must not create an empty file

    names, tokens = read_all_shards(tmp_path)
    assert names == ["val_000000.bin", "train_000000.bin", "train_000001.bin"]
    assert tokens == list(range(1, 26))


def test_prepare_marks_every_document_and_keeps_literal_endoftext_as_text(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("The Martians landed on Horsell Common.")
    (docs / "b.txt").write_text("A page that mentions <|endoftext|> in passing, and keeps going.")
    (docs / "c.txt").write_text("The end of the third document.")

    out = tmp_path / "shards"
    prepare_data.prepare(prepare_data.documents_from_folder(docs), out, shard_size=16)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    expected = []
    for f in sorted(docs.glob("*.txt")):
        expected += tokenizer.encode(f.read_text(), split_special_tokens=True) + [ENDOFTEXT]

    _, tokens = read_all_shards(out)
    assert tokens == expected
    assert tokens.count(ENDOFTEXT) == 3     # one per document -- none from the literal text in b.txt


def test_prepare_stops_exactly_at_the_token_budget(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("The Martians landed on Horsell Common, and nobody believed it at first. " * 5)

    out = tmp_path / "shards"
    writer = prepare_data.prepare(prepare_data.documents_from_folder(docs), out, shard_size=16, max_tokens=30)

    _, tokens = read_all_shards(out)
    assert len(tokens) == 30
    assert writer.total_tokens == 30


def test_writer_refuses_a_folder_that_already_has_shards(tmp_path):
    write_shard(tmp_path / "train_000000.bin", [1, 2, 3])
    with pytest.raises(FileExistsError):
        prepare_data.ShardWriter(tmp_path, shard_size=10)
    assert read_shard(tmp_path / "train_000000.bin").tolist() == [1, 2, 3]     # existing shard untouched
