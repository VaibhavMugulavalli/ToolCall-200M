#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scaling.config import load_config
from scaling.model import ToolCallLanguageModel
from scaling.utils import count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model construction")
    parser.add_argument("--config", required=True)
    parser.add_argument("--forward-check", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model = ToolCallLanguageModel(config.model)
    total, non_embedding = count_parameters(model)
    result = {
        "total_parameters": total,
        "non_embedding_parameters": non_embedding,
        "embedding_parameters": total - non_embedding,
        "tokens_per_optimizer_step": config.tokens_per_optimizer_step,
    }
    print(json.dumps(result, indent=2))

    expected_by_family = {
        "m13_": 12_913_920,
        "m30_": 29_990_784,
        "m60_": 60_439_040,
    }
    expected = next(
        (
            parameter_count
            for prefix, parameter_count in expected_by_family.items()
            if config.run_name.startswith(prefix)
        ),
        None,
    )
    if expected is not None and total != expected:
        raise RuntimeError(
            f"{config.run_name} should have {expected:,} parameters; found {total:,}"
        )
    if args.forward_check:
        model.eval()
        tokens = torch.randint(
            0, config.model.vocab_size, (2, 32), dtype=torch.long
        )
        with torch.no_grad():
            output = model(tokens, labels=tokens)
        print(
            json.dumps(
                {
                    "forward_loss": float(output["loss"].item()),
                    "logits_shape": list(output["logits"].shape),
                },
                indent=2,
            )
        )
    print("Model check passed.")


if __name__ == "__main__":
    main()
