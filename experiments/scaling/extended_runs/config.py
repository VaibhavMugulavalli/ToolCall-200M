from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .bootstrap import ensure_scaling_runs_importable

ensure_scaling_runs_importable()

from scaling.config import DataConfig, TrainingConfig


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    sequence_length: int
    num_layers: int
    num_heads: int
    hidden_size: int
    mlp_ratio: int = 4
    intermediate_size: int | None = None
    activation: str = "gelu"
    num_kv_heads: int | None = None
    rope_base: float = 10_000.0
    rms_norm_epsilon: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        config = cls(**values)
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if (config.hidden_size // config.num_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if config.vocab_size <= 0 or config.sequence_length <= 0:
            raise ValueError("vocab_size and sequence_length must be positive")
        if config.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if config.intermediate_size is not None and config.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive when specified")
        if config.activation not in {"gelu", "swiglu"}:
            raise ValueError("activation must be gelu or swiglu")
        if config.resolved_num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if config.num_heads % config.resolved_num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        return config

    @property
    def resolved_intermediate_size(self) -> int:
        if self.intermediate_size is not None:
            return self.intermediate_size
        return self.hidden_size * self.mlp_ratio

    @property
    def resolved_num_kv_heads(self) -> int:
        if self.num_kv_heads is not None:
            return self.num_kv_heads
        return self.num_heads


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        run_name = values.get("run_name", "").strip()
        if not run_name or any(char in run_name for char in "/\\"):
            raise ValueError("run_name must be a non-empty directory-safe name")
        return cls(
            run_name=run_name,
            model=ModelConfig.from_dict(values["model"]),
            data=DataConfig.from_dict(values["data"]),
            training=TrainingConfig.from_dict(values["training"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def tokens_per_micro_batch(self) -> int:
        return self.training.micro_batch_size * self.model.sequence_length

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.tokens_per_micro_batch * self.training.gradient_accumulation_steps


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return ExperimentConfig.from_dict(json.load(handle))


def resolve_project_path(project_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()

