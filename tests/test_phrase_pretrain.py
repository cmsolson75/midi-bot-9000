import json

import numpy as np
import pytest
from test_pipeline import make_dataset

from jazzbot.midi import PreprocessConfig, load_tokenizer, make_tokenizer
from jazzbot.phrase_pretrain import prepare_phrase_pretraining
from jazzbot.prepare import prepare


def test_alignment_preserves_notes_splits_sampling_weights_and_targets(tmp_path):
    raw, source, output = tmp_path / "raw", tmp_path / "source", tmp_path / "aligned"
    make_dataset(raw)
    prepare(
        raw, source, split_mode="official", cfg=PreprocessConfig(min_notes=4), chord_compatible=True
    )
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["records"]:
        record["sampling_weight"] = 0.7
    manifest_path.write_text(json.dumps(manifest))
    target = tmp_path / "phrase-tokenizer.json"
    tokenizer = make_tokenizer(use_chords=True, use_phrases=True)
    tokenizer.save(target)
    before = load_tokenizer(source / "tokenizer.json")
    prepare_phrase_pretraining(source, output, target)
    after_manifest = json.loads((output / "manifest.json").read_text())
    assert after_manifest["records"] == manifest["records"]
    assert after_manifest["tokens"] == manifest["tokens"]
    assert after_manifest["preprocessing"] == manifest["preprocessing"]
    assert after_manifest["phrase_boundaries"] is False
    assert (output / "tokenizer.json").read_bytes() == target.read_bytes()
    for split in ("train", "val", "test"):
        old = np.fromfile(source / f"{split}.bin", dtype=np.uint16)
        new = np.fromfile(output / f"{split}.bin", dtype=np.uint16)
        assert [before[int(i)] for i in old] == [tokenizer[int(i)] for i in new]
        assert tokenizer["PhraseStart_None"] not in new
    # Reusing the exact inputs is safe. Changed input/output snapshots are rejected.
    prepare_phrase_pretraining(source, output, target)
    with (output / "train.bin").open("ab") as handle:
        handle.write(b"\x00\x00")
    with pytest.raises(ValueError, match="tokens have changed"):
        prepare_phrase_pretraining(source, output, target)
    with pytest.raises(ValueError, match="Target must add only"):
        prepare_phrase_pretraining(source, tmp_path / "bad", source / "tokenizer.json")


def test_alignment_rejects_unknown_tokens_without_publishing_output(tmp_path):
    raw, source = tmp_path / "raw", tmp_path / "source"
    make_dataset(raw)
    prepare(
        raw, source, split_mode="official", cfg=PreprocessConfig(min_notes=4), chord_compatible=True
    )
    path = source / "train.bin"
    values = np.fromfile(path, dtype=np.uint16)
    values[0] = 65535
    values.tofile(path)
    target = tmp_path / "phrase-tokenizer.json"
    make_tokenizer(use_chords=True, use_phrases=True).save(target)
    output = tmp_path / "aligned"
    with pytest.raises(ValueError, match="unknown token"):
        prepare_phrase_pretraining(source, output, target)
    assert not output.exists()
