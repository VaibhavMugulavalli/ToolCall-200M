from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.final_pretraining.config import (
    TokenPlan,
    estimate_parameter_count,
    load_config,
)
from training.final_pretraining.data import ShardedTokenCorpus, rank_batch_start


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "configs/toolcall_200m_4b.json"


class ConfigAndDataTests(unittest.TestCase):
    def test_frozen_parameter_count_and_token_plan(self) -> None:
        config = load_config(CONFIG_PATH)
        self.assertEqual(estimate_parameter_count(config.model), 200_184_576)
        plan = TokenPlan.from_corpus(
            config, config.data.expected_train_source_tokens
        )
        self.assertEqual(plan.prediction_tokens_per_step, 262_144)
        self.assertEqual(plan.source_tokens_per_step, 262_272)
        self.assertEqual(plan.optimizer_steps, 15_281)
        self.assertEqual(plan.total_prediction_tokens, 4_005_822_464)
        self.assertEqual(plan.total_source_tokens, 4_007_778_432)
        self.assertEqual(plan.unused_source_tokens, 56_640)
        self.assertEqual(plan.warmup_steps, 306)

    def test_cross_shard_read_and_rank_disjoint_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.arange(0, 7, dtype=np.uint16).tofile(root / "train_shard_00000.bin")
            np.arange(7, 15, dtype=np.uint16).tofile(root / "train_shard_00001.bin")
            corpus = ShardedTokenCorpus(root, ("train_shard_*.bin",))
            self.assertEqual(corpus.total_tokens, 15)
            self.assertEqual(corpus.read(5, 6).tolist(), [5, 6, 7, 8, 9, 10])

            rank_zero = rank_batch_start(0, 0, local_batch_size=2, sequence_length=2)
            rank_one = rank_batch_start(0, 1, local_batch_size=2, sequence_length=2)
            self.assertEqual(rank_zero, 0)
            self.assertEqual(rank_one, 6)
            self.assertEqual(
                rank_batch_start(12, 0, local_batch_size=2, sequence_length=2), 12
            )


if __name__ == "__main__":
    unittest.main()
