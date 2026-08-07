#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

from .config import load_config


PROJECT_ROOT = Path(__file__).resolve().parent
CASES = {
    "m30": (
        PROJECT_ROOT / "configs/m30_standard_470m.json",
        29_990_784,
        "gelu",
        6,
    ),
    "m60": (
        PROJECT_ROOT / "configs/m60_swiglu_gqa_470m.json",
        60_439_040,
        "swiglu",
        4,
    ),
}


def parameter_count(config) -> int:
    model = config.model
    hidden = model.hidden_size
    head_dimension = hidden // model.num_heads
    kv_dimension = model.resolved_num_kv_heads * head_dimension
    attention = hidden * hidden + 2 * hidden * kv_dimension + hidden * hidden
    feed_forward_projection_count = 3 if model.activation == "swiglu" else 2
    feed_forward = (
        feed_forward_projection_count
        * hidden
        * model.resolved_intermediate_size
    )
    norms = 2 * hidden
    blocks = model.num_layers * (attention + feed_forward + norms)
    embeddings = model.vocab_size * hidden
    if not model.tie_embeddings:
        embeddings *= 2
    return embeddings + blocks + hidden


def main() -> None:
    for name, (path, expected_parameters, activation, kv_heads) in CASES.items():
        config = load_config(path)
        parameters = parameter_count(config)
        if parameters != expected_parameters:
            raise RuntimeError(
                f"{name}: expected {expected_parameters:,} parameters, "
                f"found {parameters:,}"
            )
        if config.model.activation != activation:
            raise RuntimeError(f"{name}: unexpected activation")
        if config.model.resolved_num_kv_heads != kv_heads:
            raise RuntimeError(f"{name}: unexpected KV-head count")
        if config.tokens_per_optimizer_step != 16_384:
            raise RuntimeError(f"{name}: global optimizer batch changed")

        steps = math.ceil(
            config.training.target_tokens / config.tokens_per_optimizer_step
        )
        prediction_tokens = steps * config.tokens_per_optimizer_step
        source_tokens = steps * (
            config.training.micro_batch_size
            * (config.model.sequence_length + 1)
            * config.training.gradient_accumulation_steps
        )
        if source_tokens > 470_000_000:
            raise RuntimeError(
                f"{name}: requires {source_tokens:,} source tokens, "
                "which exceeds the 470M-token shard"
            )
        print(
            f"PASS {name}: parameters={parameters:,} "
            f"configured_target={config.training.target_tokens:,} "
            f"actual_predictions={prediction_tokens:,} "
            f"source={source_tokens:,} activation={activation} "
            f"query_heads={config.model.num_heads} kv_heads={kv_heads}"
        )
    print("Extended-run configuration validation passed.")


if __name__ == "__main__":
    main()

