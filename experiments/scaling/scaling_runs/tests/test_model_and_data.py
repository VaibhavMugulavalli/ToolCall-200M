from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scaling.config import load_config
from scaling.data import CorpusExhausted, PackedTokenCorpus
from scaling.model import ToolCallLanguageModel
from scaling.utils import count_parameters


def test_model_parameter_counts() -> None:
    expected = {
        "m13/configs/low_120m.json": 12_913_920,
        "m30/configs/low_50m.json": 29_990_784,
        "m60/configs/low_25m.json": 60_439_040,
    }
    for relative, count in expected.items():
        model = ToolCallLanguageModel(load_config(ROOT / relative).model)
        assert count_parameters(model)[0] == count


def test_corpus_crosses_shard_boundaries_without_wrapping(tmp_path: Path) -> None:
    root = tmp_path / "corpus" / "shards"
    root.mkdir(parents=True)
    np.arange(6, dtype="<u2").tofile(root / "shard_00000.bin")
    np.arange(6, 12, dtype="<u2").tofile(root / "shard_00001.bin")
    corpus = PackedTokenCorpus(root.parent)
    assert corpus.read(4, 6).tolist() == [4, 5, 6, 7, 8, 9]
    try:
        corpus.read(10, 3)
    except CorpusExhausted:
        pass
    else:
        raise AssertionError("Corpus must not wrap around at end of data")


def test_model_forward() -> None:
    config = load_config(ROOT / "m13/configs/debug_1m.json").model
    model = ToolCallLanguageModel(config)
    tokens = torch.randint(0, config.vocab_size, (2, 16))
    output = model(tokens, labels=tokens)
    assert output["logits"].shape == (2, 16, config.vocab_size)
    assert torch.isfinite(output["loss"])
