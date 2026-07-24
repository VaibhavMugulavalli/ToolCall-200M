from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    sequence_length: int
    num_layers: int
    num_heads: int
    hidden_size: int
    mlp_ratio: int = 4
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
        return config


@dataclass(frozen=True)
class DataConfig:
    train_dir: str
    validation_dir: str
    structured_validation_dir: str | None = None
    token_dtype: str = "uint16"

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "DataConfig":
        config = cls(**values)
        if config.token_dtype != "uint16":
            raise ValueError("This experiment currently supports uint16 shards only")
        return config


@dataclass(frozen=True)
class TrainingConfig:
    target_tokens: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_learning_rate: float
    minimum_learning_rate: float
    warmup_fraction: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    gradient_clip: float
    log_every_steps: int
    evaluate_every_steps: int
    validation_batches: int
    final_validation_batches: int
    checkpoint_every_steps: int
    keep_last_checkpoints: int
    seed: int
    precision: str = "fp16"
    compile_model: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        config = cls(**values)
        positive_fields = (
            "target_tokens",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "log_every_steps",
            "evaluate_every_steps",
            "validation_batches",
            "final_validation_batches",
            "checkpoint_every_steps",
            "keep_last_checkpoints",
        )
        for field in positive_fields:
            if getattr(config, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if not 0.0 <= config.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if config.minimum_learning_rate > config.max_learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed max_learning_rate")
        if config.precision not in {"fp16", "bf16", "fp32"}:
            raise ValueError("precision must be fp16, bf16, or fp32")
        return config


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

