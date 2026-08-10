from __future__ import annotations

import hashlib
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
    num_kv_heads: int
    hidden_size: int
    intermediate_size: int
    activation: str
    rope_base: float
    rms_norm_epsilon: float
    dropout: float
    tie_embeddings: bool
    gradient_checkpointing: bool
    expected_parameter_count: int

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        config = cls(**values)
        if config.hidden_size % config.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if (config.hidden_size // config.num_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if config.num_heads % config.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if config.activation != "swiglu":
            raise ValueError("The frozen final architecture requires SwiGLU")
        if config.dropout != 0.0:
            raise ValueError("The frozen final architecture uses dropout=0")
        actual = estimate_parameter_count(config)
        if actual != config.expected_parameter_count:
            raise ValueError(
                "Model parameter estimate does not match expected_parameter_count: "
                f"{actual:,} != {config.expected_parameter_count:,}"
            )
        return config


def estimate_parameter_count(config: ModelConfig) -> int:
    """Exact count for the bias-free RMSNorm/GQA/SwiGLU implementation."""

    embedding = config.vocab_size * config.hidden_size
    output_head = 0 if config.tie_embeddings else embedding
    head_dimension = config.hidden_size // config.num_heads
    kv_dimension = config.num_kv_heads * head_dimension
    attention = (
        config.hidden_size * config.hidden_size
        + 2 * config.hidden_size * kv_dimension
        + config.hidden_size * config.hidden_size
    )
    feed_forward = 3 * config.hidden_size * config.intermediate_size
    norms = 2 * config.hidden_size
    blocks = config.num_layers * (attention + feed_forward + norms)
    final_norm = config.hidden_size
    return embedding + output_head + blocks + final_norm


@dataclass(frozen=True)
class DataConfig:
    train_patterns: tuple[str, ...]
    validation_general_patterns: tuple[str, ...]
    validation_structured_patterns: tuple[str, ...]
    tokenizer_patterns: tuple[str, ...]
    token_dtype: str
    expected_train_source_tokens: int
    expected_validation_general_tokens: int
    expected_validation_structured_tokens: int

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "DataConfig":
        normalized = dict(values)
        for key in (
            "train_patterns",
            "validation_general_patterns",
            "validation_structured_patterns",
            "tokenizer_patterns",
        ):
            normalized[key] = tuple(normalized[key])
        config = cls(**normalized)
        if config.token_dtype != "uint16":
            raise ValueError("Final pretraining requires raw uint16 token shards")
        if min(
            config.expected_train_source_tokens,
            config.expected_validation_general_tokens,
            config.expected_validation_structured_tokens,
        ) <= 0:
            raise ValueError("Expected split token counts must be positive")
        return config


@dataclass(frozen=True)
class TrainingConfig:
    expected_world_size: int
    micro_batch_size_per_gpu: int
    gradient_accumulation_steps: int
    minimum_prediction_tokens: int
    max_learning_rate: float
    minimum_learning_rate: float
    warmup_fraction: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    gradient_clip: float
    seed: int
    log_every_steps: int
    evaluate_every_steps: int
    validation_batches_per_rank: int
    final_validation_batches_per_rank: int
    local_checkpoint_interval_minutes: float
    hub_sync_interval_minutes: float
    keep_last_local_checkpoints: int
    max_session_minutes: float

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        config = cls(**values)
        positive = (
            "expected_world_size",
            "micro_batch_size_per_gpu",
            "gradient_accumulation_steps",
            "minimum_prediction_tokens",
            "log_every_steps",
            "evaluate_every_steps",
            "validation_batches_per_rank",
            "final_validation_batches_per_rank",
            "local_checkpoint_interval_minutes",
            "hub_sync_interval_minutes",
            "keep_last_local_checkpoints",
            "max_session_minutes",
        )
        for key in positive:
            if getattr(config, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if config.expected_world_size != 2:
            raise ValueError("This frozen run is designed for exactly two GPUs")
        if not 0.0 < config.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in (0, 1)")
        if config.minimum_learning_rate > config.max_learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed max_learning_rate")
        if config.hub_sync_interval_minutes < config.local_checkpoint_interval_minutes:
            raise ValueError("Hub sync cannot be more frequent than local checkpoints")
        return config


@dataclass(frozen=True)
class TrackingConfig:
    wandb_project: str
    wandb_run_name: str
    hub_checkpoint_path: str

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrackingConfig":
        config = cls(**values)
        if not config.wandb_project or not config.hub_checkpoint_path:
            raise ValueError("Tracking names must not be empty")
        return config


@dataclass(frozen=True)
class FinalRunConfig:
    run_name: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    tracking: TrackingConfig

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FinalRunConfig":
        run_name = str(values.get("run_name", "")).strip()
        if not run_name or any(character in run_name for character in "/\\"):
            raise ValueError("run_name must be a non-empty, directory-safe name")
        return cls(
            run_name=run_name,
            model=ModelConfig.from_dict(values["model"]),
            data=DataConfig.from_dict(values["data"]),
            training=TrainingConfig.from_dict(values["training"]),
            tracking=TrackingConfig.from_dict(values["tracking"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TokenPlan:
    world_size: int
    sequence_length: int
    micro_batch_size_per_gpu: int
    gradient_accumulation_steps: int
    optimizer_steps: int
    prediction_tokens_per_step: int
    source_tokens_per_step: int
    total_prediction_tokens: int
    total_source_tokens: int
    unused_source_tokens: int
    warmup_steps: int

    @classmethod
    def from_corpus(
        cls, config: FinalRunConfig, total_train_source_tokens: int
    ) -> "TokenPlan":
        training = config.training
        sequence_length = config.model.sequence_length
        global_sequences = (
            training.expected_world_size
            * training.micro_batch_size_per_gpu
            * training.gradient_accumulation_steps
        )
        prediction_tokens_per_step = global_sequences * sequence_length
        source_tokens_per_step = global_sequences * (sequence_length + 1)
        optimizer_steps = total_train_source_tokens // source_tokens_per_step
        if optimizer_steps < 1:
            raise ValueError("Training corpus cannot provide one optimizer step")
        total_prediction_tokens = optimizer_steps * prediction_tokens_per_step
        total_source_tokens = optimizer_steps * source_tokens_per_step
        if total_prediction_tokens < training.minimum_prediction_tokens:
            raise ValueError(
                "The corpus cannot supply the required prediction-token budget: "
                f"{total_prediction_tokens:,} < {training.minimum_prediction_tokens:,}"
            )
        warmup_steps = max(1, round(optimizer_steps * training.warmup_fraction))
        return cls(
            world_size=training.expected_world_size,
            sequence_length=sequence_length,
            micro_batch_size_per_gpu=training.micro_batch_size_per_gpu,
            gradient_accumulation_steps=training.gradient_accumulation_steps,
            optimizer_steps=optimizer_steps,
            prediction_tokens_per_step=prediction_tokens_per_step,
            source_tokens_per_step=source_tokens_per_step,
            total_prediction_tokens=total_prediction_tokens,
            total_source_tokens=total_source_tokens,
            unused_source_tokens=total_train_source_tokens - total_source_tokens,
            warmup_steps=warmup_steps,
        )


def load_config(path: str | Path) -> FinalRunConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return FinalRunConfig.from_dict(json.load(handle))
