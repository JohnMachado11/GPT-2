import json

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from gpt2.train.shards import (
    PrepRefused, ShardWriter, documents_from_folder, load_tokens, prepare, read_shard, shard_paths, text_shards,
    write_shard,
)


ENDOFTEXT = 50256
TOKENIZER = AutoTokenizer.from_pretrained("gpt2")


def read_set(folder, name):
    # every shard of one set ("train" or "val"), in order, joined into one list
    return np.concatenate([read_shard(path) for path in shard_paths(folder, name)]).tolist()


def split_each_90_10(texts):
    # the train and val tokens the 90/10 split of each document should give -- the rule train_on_text.py
    # has always used, with <|endoftext|> after every piece
    expected_train, expected_val = [], []
    for text in texts:
        tokens = TOKENIZER.encode(text, split_special_tokens=True)
        split = int(0.9 * len(tokens))
        expected_train += tokens[:split] + [ENDOFTEXT]
        expected_val += tokens[split:] + [ENDOFTEXT]
    return expected_train, expected_val


def write_documents(folder, texts):
    folder.mkdir()
    for i, text in enumerate(texts):
        (folder / f"{i:03d}.txt").write_text(text)
    return folder


def test_shard_round_trip(tmp_path):
    ids = [0, 464, 3981, ENDOFTEXT, 1547]
    write_shard(tmp_path / "s.bin", ids)
    assert read_shard(tmp_path / "s.bin").tolist() == ids
    assert (tmp_path / "s.bin").stat().st_size == 2 * len(ids)      # 2 bytes per token, no header


def test_shard_rejects_ids_too_big_for_uint16(tmp_path):
    with pytest.raises(ValueError):
        write_shard(tmp_path / "s.bin", [2 ** 16])             # one past the largest uint16


def test_shard_rejects_negative_ids(tmp_path):
    # uint16 can't hold -1: it would silently become 65,535
    with pytest.raises(ValueError):
        write_shard(tmp_path / "s.bin", [5, -1])


def test_shard_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError):
        write_shard(tmp_path / "s.bin", [])


def test_writer_spills_across_shards_without_losing_tokens(tmp_path):
    writer = ShardWriter(tmp_path, "train", shard_size=10)
    writer.add(list(range(1, 8)))       # 7 tokens: fits in the first shard
    writer.add(list(range(8, 26)))      # 18 more: fills shard 0, fills shard 1, 5 spill into shard 2
    writer.write()                      # the last, partly filled shard
    writer.write()                      # nothing left: must not create an empty file

    assert [path.name for path in shard_paths(tmp_path, "train")] == [
        "train_000000.bin", "train_000001.bin", "train_000002.bin"]
    assert read_set(tmp_path, "train") == list(range(1, 26))


def test_writer_refuses_a_folder_that_already_has_shards_of_its_set(tmp_path):
    write_shard(tmp_path / "train_000000.bin", [1, 2, 3])
    with pytest.raises(PrepRefused):
        ShardWriter(tmp_path, "train", shard_size=10)
    assert read_shard(tmp_path / "train_000000.bin").tolist() == [1, 2, 3]     # existing shard untouched
    ShardWriter(tmp_path, "val", shard_size=10)                                 # the other set is fine


def test_long_documents_send_the_last_10_percent_of_each_one_to_val(tmp_path):
    # the rule train_on_text.py has always used: split EACH document 90/10, so every document shows up
    # in both sets, with <|endoftext|> after every piece
    texts = ["The Martians landed on Horsell Common, and nobody believed it at first. " * 6,
             "A second book, much shorter.",
             "And a third one, of middling length, about the heat-ray and the river. " * 3]
    folder = write_documents(tmp_path / "docs", texts)
    prepare(documents_from_folder(folder), TOKENIZER, tmp_path / "shards", train_fraction=0.9, whole_documents=False)
    assert (read_set(tmp_path / "shards", "train"), read_set(tmp_path / "shards", "val")) == split_each_90_10(texts)


def test_short_documents_send_every_nth_one_to_val_whole(tmp_path):
    texts = [f"Web page number {i} talks about topic {i * 7}." for i in range(25)]
    folder = write_documents(tmp_path / "docs", texts)
    prepare(documents_from_folder(folder), TOKENIZER, tmp_path / "shards", train_fraction=0.9, whole_documents=True)

    pieces = [TOKENIZER.encode(text) + [ENDOFTEXT] for text in texts]
    assert read_set(tmp_path / "shards", "val") == pieces[0] + pieces[10] + pieces[20]     # every 10th, whole
    assert read_set(tmp_path / "shards", "train") == sum((p for i, p in enumerate(pieces) if i % 10), [])


def test_99_percent_means_every_100th_document(tmp_path):
    # 1 / (1 - 0.99) is 99.99999999999991 in floating point: cut down to a whole number, it would send
    # every 99th page to val
    texts = [f"Page {i}." for i in range(201)]
    folder = write_documents(tmp_path / "docs", texts)
    prepare(documents_from_folder(folder), TOKENIZER, tmp_path / "shards", train_fraction=0.99, whole_documents=True)
    pieces = [TOKENIZER.encode(text) + [ENDOFTEXT] for text in texts]
    assert read_set(tmp_path / "shards", "val") == pieces[0] + pieces[100] + pieces[200]


def test_the_split_point_is_cut_down_never_rounded_up(tmp_path):
    # today's rule is int(0.9 * length) -- a document whose 90% mark falls past a half (e.g. 13.5 of 15
    # tokens) shows the difference: rounding would move one token from val to train
    text = next(t for t in ("The Martians landed. " * n for n in range(1, 40))
                if (0.9 * len(TOKENIZER.encode(t))) % 1 >= 0.5)
    folder = write_documents(tmp_path / "docs", [text])
    prepare(documents_from_folder(folder), TOKENIZER, tmp_path / "shards", train_fraction=0.9, whole_documents=False)
    length = len(TOKENIZER.encode(text))
    assert len(read_set(tmp_path / "shards", "val")) == length - int(0.9 * length) + 1     # +1: <|endoftext|>


def test_a_literal_endoftext_in_the_text_stays_text(tmp_path):
    texts = ["A page that mentions <|endoftext|> in passing, and keeps going.", "Another page."]
    folder = write_documents(tmp_path / "docs", texts)
    prepare(documents_from_folder(folder), TOKENIZER, tmp_path / "shards", train_fraction=0.5, whole_documents=True)
    all_tokens = read_set(tmp_path / "shards", "val") + read_set(tmp_path / "shards", "train")
    assert all_tokens.count(ENDOFTEXT) == 2         # one per document -- none from the literal text


def test_prepare_stops_exactly_at_the_token_budget(tmp_path):
    folder = write_documents(tmp_path / "docs", [f"Page {i}: the Martians landed, and nobody believed it." for i in range(40)])
    train, val = prepare(documents_from_folder(folder), TOKENIZER, tmp_path / "shards", train_fraction=0.9,
                         whole_documents=True, shard_size=16, max_tokens=100)
    assert train.total_tokens + val.total_tokens == 100
    assert len(read_set(tmp_path / "shards", "train")) + len(read_set(tmp_path / "shards", "val")) == 100


def test_prepare_stops_reading_documents_once_the_budget_is_reached(tmp_path):
    # with FineWeb, reading on would stream all 10B tokens just to throw them away
    read = []
    def documents():
        for i in range(10_000):
            read.append(i)
            yield f"Page {i}: the Martians landed, and nobody believed it."
    prepare(documents(), TOKENIZER, tmp_path / "shards", train_fraction=0.9, whole_documents=True, max_tokens=100)
    assert len(read) < 1_100          # ~10 pages hold 100 tokens; the tokenizer reads ahead 1,000 at a time


def test_info_json_records_the_split(tmp_path):
    folder = write_documents(tmp_path / "docs", ["One long document about the Martians. " * 20])
    train, val = prepare(documents_from_folder(folder), TOKENIZER, tmp_path / "shards", train_fraction=0.9,
                         whole_documents=False, source="my books")
    info = json.loads((tmp_path / "shards" / "info.json").read_text())
    assert info == {"source": "my books", "train_fraction": 0.9, "whole_documents": False,
                    "shard_size": 100_000_000, "val_shard_size": 100_000_000, "max_tokens": None,
                    "train_tokens": train.total_tokens, "val_tokens": val.total_tokens,
                    "train_shards": 1, "val_shards": 1}


def test_a_split_that_leaves_a_set_empty_is_refused_and_not_marked_finished(tmp_path):
    # one document, split whole: it goes to val, and nothing is left to train on
    folder = write_documents(tmp_path / "docs", ["The only page."])
    with pytest.raises(PrepRefused, match="both need some"):
        prepare(documents_from_folder(folder), TOKENIZER, tmp_path / "shards", train_fraction=0.9, whole_documents=True)
    assert not (tmp_path / "shards" / "info.json").exists()


@pytest.mark.parametrize("train_fraction", [0, 1, 1.5, -0.1])
def test_a_train_fraction_outside_0_to_1_is_refused(tmp_path, train_fraction):
    with pytest.raises(PrepRefused, match="between 0 and 1"):
        prepare(iter(["text"]), TOKENIZER, tmp_path / "shards", train_fraction=train_fraction, whole_documents=False)


def test_load_tokens_joins_a_sets_shards_in_order_as_one_tensor(tmp_path):
    for i, ids in enumerate([[1, 2, 3], [4, 5], [6]]):
        write_shard(tmp_path / f"train_{i:06d}.bin", ids)
    write_shard(tmp_path / "val_000000.bin", [99])                     # another set: must be left out
    tokens = load_tokens(tmp_path, "train")
    assert tokens.tolist() == [1, 2, 3, 4, 5, 6]
    assert tokens.dtype == torch.int64                                 # what the model's embedding needs


def test_load_tokens_says_how_to_fix_a_missing_set(tmp_path):
    with pytest.raises(FileNotFoundError, match="delete the folder to rebuild"):
        load_tokens(tmp_path, "train")


def test_load_tokens_ignores_a_stray_copy_of_a_shard(tmp_path):
    write_shard(tmp_path / "train_000000.bin", [1, 2, 3])
    write_shard(tmp_path / "train_000000 copy.bin", [1, 2, 3])        # what Finder's "Duplicate" makes
    assert load_tokens(tmp_path, "train").tolist() == [1, 2, 3]


def test_with_whole_documents_a_fraction_that_isnt_1_in_n_is_refused(tmp_path):
    # 0.97 would really mean "every 33rd" -- 3.03%, not 3% -- so it's refused rather than quietly changed
    with pytest.raises(PrepRefused, match="1 - 1/N"):
        prepare(iter(["text"]), TOKENIZER, tmp_path / "shards", train_fraction=0.97, whole_documents=True)
    prepare(iter([f"Page {i}." for i in range(40)]), TOKENIZER, tmp_path / "ok", train_fraction=0.95,
            whole_documents=True)                                    # 1 in 20: fine


@pytest.mark.parametrize("bad", [dict(max_tokens=0), dict(max_tokens=-5), dict(shard_size=0), dict(val_shard_size=0)])
def test_a_budget_or_shard_size_below_1_is_refused(tmp_path, bad):
    with pytest.raises(PrepRefused, match="at least 1"):
        prepare(iter(["Some text."]), TOKENIZER, tmp_path / "shards", train_fraction=0.9, whole_documents=False, **bad)


def test_the_budget_is_exact_with_long_documents_too(tmp_path):
    folder = write_documents(tmp_path / "docs", ["The Martians landed on Horsell Common. " * 30] * 3)
    train, val = prepare(documents_from_folder(folder), TOKENIZER, tmp_path / "shards", train_fraction=0.9,
                         whole_documents=False, max_tokens=250)
    assert train.total_tokens + val.total_tokens == 250


def test_an_interrupted_rebuild_never_looks_finished(tmp_path):
    # a finished build, then its shards deleted and a rebuild cut off partway (a Colab disconnect):
    # the old info.json must not be left behind to make the half-built folder look finished
    out = tmp_path / "shards"
    prepare(iter(["One long document about the Martians. " * 20]), TOKENIZER, out, train_fraction=0.9,
            whole_documents=False)
    for shard in out.glob("*.bin"):
        shard.unlink()
    def documents_then_disconnect():
        yield "A first page."
        raise KeyboardInterrupt("disconnected")
    with pytest.raises(KeyboardInterrupt):
        prepare(documents_then_disconnect(), TOKENIZER, out, train_fraction=0.9, whole_documents=False)
    assert not (out / "info.json").exists()


def test_a_refused_build_keeps_a_finished_folders_info_json(tmp_path):
    # the old info.json is removed only AFTER the existing-shards check passes -- a refusal must leave
    # a finished folder exactly as it was
    out = tmp_path / "shards"
    prepare(iter(["One long document about the Martians. " * 20]), TOKENIZER, out, train_fraction=0.9,
            whole_documents=False)
    before = (out / "info.json").read_text()
    with pytest.raises(PrepRefused):
        prepare(iter(["Other text."]), TOKENIZER, out, train_fraction=0.9, whole_documents=False)
    assert (out / "info.json").read_text() == before


def test_a_file_that_isnt_utf8_is_named_in_the_error(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "latin1.txt").write_bytes("caf\xe9".encode("latin-1"))
    with pytest.raises(PrepRefused, match="latin1.txt isn't UTF-8"):
        list(documents_from_folder(folder))


def test_1_in_a_million_is_a_valid_whole_document_split(tmp_path):
    prepare(iter([f"Page {i}." for i in range(3)]), TOKENIZER, tmp_path / "shards", train_fraction=0.999999,
            whole_documents=True)


class CountingTokenizer:
    # the real tokenizer, recording how many documents each call is given
    eos_token_id = ENDOFTEXT

    def __init__(self):
        self.documents_per_call = []

    def __call__(self, texts, **options):
        self.documents_per_call.append(len(texts))
        return TOKENIZER(texts, **options)


def test_long_documents_are_tokenized_one_at_a_time_short_ones_in_batches(tmp_path):
    # tokenizing many big files in ONE call holds them all in memory at once
    texts = [f"Document {i} about the Martians." for i in range(25)]
    long_documents, short_documents = CountingTokenizer(), CountingTokenizer()
    prepare(iter(texts), long_documents, tmp_path / "long", train_fraction=0.9, whole_documents=False)
    prepare(iter(texts), short_documents, tmp_path / "short", train_fraction=0.9, whole_documents=True)
    assert long_documents.documents_per_call == [1] * 25
    assert short_documents.documents_per_call == [25]


# ---------- text_shards: build once, reuse until the text or the split changes ----------

def build(tmp_path, capsys, train_fraction=0.9):
    train, val = text_shards(tmp_path / "docs", TOKENIZER, tmp_path / "shards", train_fraction)
    built = "tokenizing" in capsys.readouterr().out
    return train, val, built


def test_text_shards_builds_once_then_reuses(tmp_path, capsys):
    texts = ["The Martians landed on Horsell Common. " * 10, "A second, shorter story."]
    write_documents(tmp_path / "docs", texts)
    expected_train, expected_val = split_each_90_10(texts)

    train, val, built = build(tmp_path, capsys)
    assert built and train.tolist() == expected_train and val.tolist() == expected_val
    train, val, built = build(tmp_path, capsys)
    assert not built and train.tolist() == expected_train and val.tolist() == expected_val


@pytest.mark.parametrize("change", ["edit a file", "add a file", "remove a file", "rename a file",
                                    "change train_fraction"])
def test_text_shards_rebuilds_when_the_text_or_the_split_changes(tmp_path, capsys, change):
    docs = write_documents(tmp_path / "docs", ["The first story. " * 10, "The second story. " * 10])
    build(tmp_path, capsys)
    train_fraction = 0.9
    if change == "edit a file":
        (docs / "000.txt").write_text("The first story, edited. " * 10)
    elif change == "add a file":
        (docs / "002.txt").write_text("A third story. " * 10)
    elif change == "remove a file":
        (docs / "001.txt").unlink()
    elif change == "rename a file":
        (docs / "001.txt").rename(docs / "000b.txt")
    else:
        train_fraction = 0.85               # (not 1 - 1/N: only fine because text splits each document)
    assert build(tmp_path, capsys, train_fraction)[2]


@pytest.mark.parametrize("damage", ["damaged info.json", "info.json that isn't an object", "info.json missing a key",
                                    "info.json missing a token count",
                                    "a shard deleted", "a shard cut short", "an empty extra shard",
                                    "a shard one byte too long"])
def test_text_shards_rebuilds_rather_than_trusting_a_folder_that_doesnt_match(tmp_path, capsys, damage):
    write_documents(tmp_path / "docs", ["The first story. " * 10, "The second story. " * 10])
    expected = build(tmp_path, capsys)[:2]
    info_path = tmp_path / "shards" / "info.json"
    info = json.loads(info_path.read_text())
    if damage == "damaged info.json":
        info_path.write_text('{"source": {"000.txt"')                # cut off mid-write
    elif damage == "info.json that isn't an object":
        info_path.write_text("[]")
    elif damage == "info.json missing a key":
        del info["train_fraction"]
        info_path.write_text(json.dumps(info))
    elif damage == "info.json missing a token count":
        del info["train_tokens"]
        info_path.write_text(json.dumps(info))
    elif damage == "a shard deleted":
        (tmp_path / "shards" / "val_000000.bin").unlink()
    elif damage == "a shard cut short":
        write_shard(tmp_path / "shards" / "train_000000.bin", [1, 2, 3])
    elif damage == "an empty extra shard":
        (tmp_path / "shards" / "train_000001.bin").touch()           # same token total, but it can't be read
    else:
        with open(tmp_path / "shards" / "train_000000.bin", "ab") as shard:
            shard.write(b"x")                                       # half a token: it can't be read
    train, val, built = build(tmp_path, capsys)
    assert built
    assert train.tolist() == expected[0].tolist() and val.tolist() == expected[1].tolist()


def test_text_shards_never_deletes_a_whole_document_build_like_finewebs(tmp_path, capsys):
    # pointing train_on_text at shards/fineweb by mistake must not wipe hours of FineWeb prep
    write_documents(tmp_path / "docs", ["The first story. " * 10])
    out = tmp_path / "fineweb"
    prepare(iter([f"Page {i}." for i in range(30)]), TOKENIZER, out, train_fraction=0.9, whole_documents=True)
    before = sorted(p.name for p in out.iterdir())
    with pytest.raises(PrepRefused, match="whole documents"):
        text_shards(tmp_path / "docs", TOKENIZER, out, 0.9)
    assert sorted(p.name for p in out.iterdir()) == before


def test_text_shards_ignores_a_stray_copy_without_rebuilding(tmp_path, capsys):
    write_documents(tmp_path / "docs", ["The first story. " * 10])
    expected_train = build(tmp_path, capsys)[0]
    write_shard(tmp_path / "shards" / "train_000000 copy.bin", [1, 2, 3])
    train, _, built = build(tmp_path, capsys)
    assert not built and train.tolist() == expected_train.tolist()


def test_an_invalid_train_fraction_is_refused_before_a_good_build_is_cleared(tmp_path, capsys):
    write_documents(tmp_path / "docs", ["The first story. " * 10])
    build(tmp_path, capsys)
    with pytest.raises(PrepRefused, match="between 0 and 1"):
        text_shards(tmp_path / "docs", TOKENIZER, tmp_path / "shards", 9)
    assert not build(tmp_path, capsys)[2]                            # the good build is still there


def test_text_shards_says_so_when_there_is_no_text(tmp_path):
    (tmp_path / "docs").mkdir()
    with pytest.raises(FileNotFoundError, match="add some text"):
        text_shards(tmp_path / "docs", TOKENIZER, tmp_path / "shards", 0.9)
