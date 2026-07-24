#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scaling.config import ExperimentConfig, load_config


MATRIX = {
    "m13": {
        "expected_parameters": 12_913_920,
        "configs": {
            "low": ("m13/configs/low_120m.json", 120_000_000),
            "medium": ("m13/configs/medium_230m.json", 230_000_000),
            "high": ("m13/configs/high_460m.json", 460_000_000),
        },
    },
    "m30": {
        "expected_parameters": 29_990_784,
        "configs": {
            "low": ("m30/configs/low_50m.json", 50_000_000),
            "medium": ("m30/configs/medium_100m.json", 100_000_000),
            "high": ("m30/configs/high_200m.json", 200_000_000),
        },
    },
    "m60": {
        "expected_parameters": 60_439_040,
        "configs": {
            "low": ("m60/configs/low_25m.json", 25_000_000),
            "medium": ("m60/configs/medium_50m.json", 50_000_000),
            "high": ("m60/configs/high_100m.json", 100_000_000),
        },
    },
}


def parameter_formula(config: ExperimentConfig) -> int:
    model = config.model
    if not model.tie_embeddings or model.mlp_ratio != 4:
        raise RuntimeError("Matrix checker assumes tied embeddings and MLP ratio 4")
    embedding = model.vocab_size * model.hidden_size
    blocks = model.num_layers * (
        12 * model.hidden_size**2 + 2 * model.hidden_size
    )
    final_norm = model.hidden_size
    return embedding + blocks + final_norm


def main() -> None:
    tier_compute: dict[str, list[float]] = {tier: [] for tier in ("low", "medium", "high")}
    rows: list[tuple[str, str, int, int, float]] = []

    for family, definition in MATRIX.items():
        expected_parameters = definition["expected_parameters"]
        for tier, (relative_path, expected_tokens) in definition["configs"].items():
            config = load_config(PROJECT_ROOT / relative_path)
            parameters = parameter_formula(config)
            if parameters != expected_parameters:
                raise RuntimeError(
                    f"{relative_path}: expected {expected_parameters:,} parameters, "
                    f"formula produced {parameters:,}"
                )
            if config.training.target_tokens != expected_tokens:
                raise RuntimeError(
                    f"{relative_path}: expected {expected_tokens:,} target tokens"
                )
            if config.tokens_per_optimizer_step != 16_384:
                raise RuntimeError(
                    f"{relative_path}: optimizer batch must be 16,384 tokens"
                )
            nd = parameters * expected_tokens
            tier_compute[tier].append(nd)
            rows.append((family, tier, parameters, expected_tokens, nd))

    for tier, values in tier_compute.items():
        relative_spread = (max(values) - min(values)) / (sum(values) / len(values))
        if relative_spread > 0.05:
            raise RuntimeError(
                f"{tier} tier is not approximately isoFLOP: spread={relative_spread:.2%}"
            )

    print("family  tier      parameters   tokens       N×D")
    print("------  --------  -----------  -----------  ------------")
    for family, tier, parameters, tokens, nd in rows:
        print(
            f"{family:<6}  {tier:<8}  {parameters:>11,}  {tokens:>11,}  {nd:>12.3e}"
        )
    print("\nExperiment matrix check passed.")


if __name__ == "__main__":
    main()

