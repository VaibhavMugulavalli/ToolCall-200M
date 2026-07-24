from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from scaling.checkpointing import (
    CheckpointManager,
    capture_rng_state,
    restore_rng_state,
)
from scaling.config import ExperimentConfig, resolve_project_path
from scaling.data import PackedTokenCorpus
from scaling.metrics import JsonlMetricsLogger
from scaling.model import ToolCallLanguageModel
from scaling.utils import (
    atomic_write_json,
    count_parameters,
    safe_perplexity,
    set_seed,
    system_information,
    utc_now,
)


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        project_root: Path,
        runs_root: Path | None = None,
        device_name: str = "auto",
        train_dir_override: str | None = None,
        validation_dir_override: str | None = None,
        structured_validation_dir_override: str | None = None,
    ) -> None:
        self.config = config
        self.project_root = project_root.resolve()
        self.device = self._select_device(device_name)
        self._validate_precision()
        set_seed(config.training.seed)

        if train_dir_override or validation_dir_override or structured_validation_dir_override:
            data = replace(
                config.data,
                train_dir=train_dir_override or config.data.train_dir,
                validation_dir=validation_dir_override or config.data.validation_dir,
                structured_validation_dir=(
                    structured_validation_dir_override
                    if structured_validation_dir_override is not None
                    else config.data.structured_validation_dir
                ),
            )
            self.config = replace(config, data=data)

        self.runs_root = (runs_root or (self.project_root / "runs")).resolve()
        self.run_dir = self.runs_root / self.config.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = JsonlMetricsLogger(self.run_dir / "metrics.jsonl")
        self.checkpoints = CheckpointManager(
            self.run_dir, self.config.training.keep_last_checkpoints
        )

        train_path = resolve_project_path(self.project_root, self.config.data.train_dir)
        validation_path = resolve_project_path(
            self.project_root, self.config.data.validation_dir
        )
        structured_path = resolve_project_path(
            self.project_root, self.config.data.structured_validation_dir
        )
        assert train_path is not None and validation_path is not None
        self.train_corpus = PackedTokenCorpus(train_path, self.config.data.token_dtype)
        self.validation_corpus = PackedTokenCorpus(
            validation_path, self.config.data.token_dtype
        )
        self.structured_validation_corpus = (
            PackedTokenCorpus(structured_path, self.config.data.token_dtype)
            if structured_path is not None
            else None
        )

        self.raw_model = ToolCallLanguageModel(self.config.model).to(self.device)
        self.parameter_count, self.non_embedding_parameter_count = count_parameters(
            self.raw_model
        )
        self.optimizer = self._build_optimizer()
        self.model: torch.nn.Module = self.raw_model
        if self.config.training.compile_model:
            if not hasattr(torch, "compile"):
                raise RuntimeError("compile_model requires PyTorch 2 or newer")
            self.model = torch.compile(self.raw_model)

        scaler_enabled = (
            self.device.type == "cuda" and self.config.training.precision == "fp16"
        )
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.max_steps = math.ceil(
            self.config.training.target_tokens
            / self.config.tokens_per_optimizer_step
        )
        self.warmup_steps = max(
            1, round(self.max_steps * self.config.training.warmup_fraction)
        )
        self.step = 0
        self.tokens_seen = 0
        self.data_cursor = 0
        self.best_validation_loss = math.inf
        self.loss_ema: float | None = None
        self.started_at = utc_now()
        self.total_training_seconds = 0.0
        self.tensorboard: SummaryWriter | None = None
        self._preflight()
        atomic_write_json(self.run_dir / "config.json", self.config.to_dict())

    @staticmethod
    def _select_device(device_name: str) -> torch.device:
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device

    def _validate_precision(self) -> None:
        precision = self.config.training.precision
        if precision == "fp16" and self.device.type != "cuda":
            raise RuntimeError("fp16 training requires CUDA; use fp32 for CPU debug")
        if precision == "bf16" and self.device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("The selected CUDA device does not support bf16")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for parameter in self.raw_model.parameters():
            if not parameter.requires_grad:
                continue
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        groups = [
            {"params": decay, "weight_decay": self.config.training.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        options: dict[str, Any] = {
            "lr": self.config.training.max_learning_rate,
            "betas": (
                self.config.training.adam_beta1,
                self.config.training.adam_beta2,
            ),
            "eps": self.config.training.adam_epsilon,
        }
        if self.device.type == "cuda":
            options["fused"] = True
        try:
            return torch.optim.AdamW(groups, **options)
        except (TypeError, RuntimeError):
            options.pop("fused", None)
            return torch.optim.AdamW(groups, **options)

    def _preflight(self) -> None:
        source_tokens_per_step = (
            self.config.training.micro_batch_size
            * (self.config.model.sequence_length + 1)
            * self.config.training.gradient_accumulation_steps
        )
        required_source_tokens = self.max_steps * source_tokens_per_step
        if required_source_tokens > self.train_corpus.total_tokens:
            raise RuntimeError(
                "Training corpus is too small for a no-repeat run. "
                f"Need {required_source_tokens:,} source tokens but found "
                f"{self.train_corpus.total_tokens:,}. The extra tokens above the target "
                "come from the one-token causal shift in every packed sequence."
            )
        if self.validation_corpus.maximum_batches(
            self.config.training.micro_batch_size,
            self.config.model.sequence_length,
        ) < 1:
            raise RuntimeError("Validation corpus is too small for one batch")
        if self.structured_validation_corpus is not None and (
            self.structured_validation_corpus.maximum_batches(
                self.config.training.micro_batch_size,
                self.config.model.sequence_length,
            )
            < 1
        ):
            raise RuntimeError("Structured validation corpus is too small for one batch")

    def describe(self) -> dict[str, Any]:
        return {
            "run_name": self.config.run_name,
            "device": str(self.device),
            "parameter_count": self.parameter_count,
            "non_embedding_parameter_count": self.non_embedding_parameter_count,
            "target_tokens": self.config.training.target_tokens,
            "tokens_per_optimizer_step": self.config.tokens_per_optimizer_step,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "train_corpus": self.train_corpus.describe().__dict__,
            "validation_corpus": self.validation_corpus.describe().__dict__,
        }

    def _autocast_context(self):
        precision = self.config.training.precision
        enabled = precision in {"fp16", "bf16"}
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(
            device_type=self.device.type,
            dtype=dtype,
            enabled=enabled,
        )

    def _learning_rate(self, step: int) -> float:
        training = self.config.training
        if step <= self.warmup_steps:
            return training.max_learning_rate * step / self.warmup_steps
        if self.max_steps == self.warmup_steps:
            return training.minimum_learning_rate
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
        return training.minimum_learning_rate + coefficient * (
            training.max_learning_rate - training.minimum_learning_rate
        )

    def _set_learning_rate(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def _checkpoint_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "config": self.config.to_dict(),
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "data_cursor": self.data_cursor,
            "best_validation_loss": self.best_validation_loss,
            "loss_ema": self.loss_ema,
            "total_training_seconds": self.total_training_seconds,
            "train_corpus_fingerprint": self.train_corpus.fingerprint(),
            "validation_corpus_fingerprint": self.validation_corpus.fingerprint(),
            "structured_validation_corpus_fingerprint": (
                self.structured_validation_corpus.fingerprint()
                if self.structured_validation_corpus is not None
                else None
            ),
            "rng_state": capture_rng_state(),
        }

    def save_checkpoint(self, label: str | None = None) -> Path:
        return self.checkpoints.save(self._checkpoint_state(), self.step, label)

    def resume(self, checkpoint: str | Path | None) -> Path | None:
        if checkpoint is None or str(checkpoint).lower() == "none":
            return None
        if str(checkpoint).lower() == "auto":
            path = self.checkpoints.latest()
            if path is None:
                return None
        else:
            path = Path(checkpoint).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint does not exist: {path}")

        state = torch.load(path, map_location=self.device, weights_only=False)
        saved_model_config = state.get("config", {}).get("model")
        if saved_model_config != self.config.to_dict()["model"]:
            raise RuntimeError("Checkpoint model configuration does not match this run")
        saved_training_config = state.get("config", {}).get("training")
        if saved_training_config != self.config.to_dict()["training"]:
            raise RuntimeError("Checkpoint training configuration does not match this run")
        if state.get("train_corpus_fingerprint") != self.train_corpus.fingerprint():
            raise RuntimeError("Training corpus changed since this checkpoint was written")
        if (
            state.get("validation_corpus_fingerprint")
            != self.validation_corpus.fingerprint()
        ):
            raise RuntimeError("Validation corpus changed since this checkpoint was written")
        structured_fingerprint = (
            self.structured_validation_corpus.fingerprint()
            if self.structured_validation_corpus is not None
            else None
        )
        if state.get("structured_validation_corpus_fingerprint") != structured_fingerprint:
            raise RuntimeError(
                "Structured validation corpus changed since this checkpoint was written"
            )
        self.raw_model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state.get("scaler", {}))
        self.step = int(state["step"])
        self.tokens_seen = int(state["tokens_seen"])
        self.data_cursor = int(state["data_cursor"])
        self.best_validation_loss = float(state.get("best_validation_loss", math.inf))
        self.loss_ema = state.get("loss_ema")
        self.total_training_seconds = float(state.get("total_training_seconds", 0.0))
        restore_rng_state(state["rng_state"])
        self.metrics.log(
            {
                "type": "event",
                "event": "resumed",
                "checkpoint": str(path),
                "step": self.step,
                "tokens_seen": self.tokens_seen,
            }
        )
        return path

    @torch.no_grad()
    def evaluate(
        self,
        corpus: PackedTokenCorpus,
        split: str,
        requested_batches: int,
    ) -> dict[str, Any]:
        self.model.eval()
        batch_size = self.config.training.micro_batch_size
        sequence_length = self.config.model.sequence_length
        batch_count = min(
            requested_batches, corpus.maximum_batches(batch_size, sequence_length)
        )
        total_loss = 0.0
        cursor = 0
        for _ in range(batch_count):
            inputs, targets, cursor = corpus.batch(
                cursor, batch_size, sequence_length, self.device
            )
            with self._autocast_context():
                loss = self.model(inputs, labels=targets)["loss"]
            total_loss += float(loss.item())
        mean_loss = total_loss / batch_count
        record = {
            "type": "validation",
            "split": split,
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "loss": mean_loss,
            "perplexity": safe_perplexity(mean_loss),
            "batches": batch_count,
            "tokens_evaluated": batch_count * batch_size * sequence_length,
        }
        self.metrics.log(record)
        if self.tensorboard is not None:
            prefix = f"validation/{split}"
            self.tensorboard.add_scalar(f"{prefix}_loss", mean_loss, self.step)
            self.tensorboard.add_scalar(
                f"{prefix}_perplexity", record["perplexity"], self.step
            )
        self.model.train()
        return record

    def _log_tensorboard_train(self, record: dict[str, Any]) -> None:
        if self.tensorboard is None:
            return
        step = int(record["step"])
        fields = {
            "train/loss": "loss",
            "train/loss_ema": "loss_ema",
            "train/learning_rate": "learning_rate",
            "train/gradient_norm": "gradient_norm",
            "performance/tokens_per_second": "tokens_per_second",
            "performance/step_seconds": "step_seconds",
            "memory/allocated_gib": "gpu_memory_allocated_gib",
            "memory/reserved_gib": "gpu_memory_reserved_gib",
            "progress/tokens_seen": "tokens_seen",
        }
        for tag, key in fields.items():
            self.tensorboard.add_scalar(tag, record[key], step)

    def _gpu_memory(self) -> tuple[float, float]:
        if self.device.type != "cuda":
            return 0.0, 0.0
        gibibyte = 1024**3
        return (
            torch.cuda.memory_allocated(self.device) / gibibyte,
            torch.cuda.memory_reserved(self.device) / gibibyte,
        )

    def _write_summary(
        self,
        status: str,
        last_train_loss: float | None = None,
        validation: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        summary = {
            "run_name": self.config.run_name,
            "status": status,
            "started_at": self.started_at,
            "updated_at": utc_now(),
            "step": self.step,
            "max_steps": self.max_steps,
            "tokens_seen": self.tokens_seen,
            "target_tokens": self.config.training.target_tokens,
            "progress_fraction": min(
                self.tokens_seen / self.config.training.target_tokens, 1.0
            ),
            "parameter_count": self.parameter_count,
            "non_embedding_parameter_count": self.non_embedding_parameter_count,
            "estimated_training_flops": 6 * self.parameter_count * self.tokens_seen,
            "last_train_loss": last_train_loss,
            "loss_ema": self.loss_ema,
            "best_validation_loss": (
                None
                if not math.isfinite(self.best_validation_loss)
                else self.best_validation_loss
            ),
            "last_validation": validation,
            "training_seconds": self.total_training_seconds,
            "error": error,
        }
        atomic_write_json(self.run_dir / "summary.json", summary)

    def train(self) -> dict[str, Any]:
        training = self.config.training
        self.tensorboard = SummaryWriter(
            log_dir=str(self.run_dir / "tensorboard"),
            purge_step=self.step if self.step > 0 else None,
            flush_secs=30,
        )
        self.metrics.log(
            {
                "type": "system",
                **system_information(self.device, self.project_root),
                **self.describe(),
            }
        )
        self._write_summary("running")
        self.model.train()
        last_loss: float | None = None
        last_validation: dict[str, Any] | None = None

        try:
            while self.step < self.max_steps:
                step_started = time.perf_counter()
                next_step = self.step + 1
                learning_rate = self._learning_rate(next_step)
                self._set_learning_rate(learning_rate)
                self.optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0

                for _ in range(training.gradient_accumulation_steps):
                    inputs, targets, self.data_cursor = self.train_corpus.batch(
                        self.data_cursor,
                        training.micro_batch_size,
                        self.config.model.sequence_length,
                        self.device,
                    )
                    with self._autocast_context():
                        loss = self.model(inputs, labels=targets)["loss"]
                        scaled_loss = loss / training.gradient_accumulation_steps
                    self.scaler.scale(scaled_loss).backward()
                    accumulated_loss += float(loss.item())
                    self.tokens_seen += int(inputs.numel())

                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.raw_model.parameters(), training.gradient_clip
                )
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_step_skipped = self.scaler.get_scale() < scale_before

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                step_seconds = time.perf_counter() - step_started
                self.total_training_seconds += step_seconds
                self.step = next_step
                last_loss = accumulated_loss / training.gradient_accumulation_steps
                self.loss_ema = (
                    last_loss
                    if self.loss_ema is None
                    else 0.98 * self.loss_ema + 0.02 * last_loss
                )

                if not math.isfinite(last_loss):
                    raise FloatingPointError(f"Non-finite training loss: {last_loss}")

                should_log = self.step == 1 or self.step % training.log_every_steps == 0
                if should_log:
                    allocated_gib, reserved_gib = self._gpu_memory()
                    record = {
                        "type": "train",
                        "step": self.step,
                        "max_steps": self.max_steps,
                        "tokens_seen": self.tokens_seen,
                        "target_tokens": training.target_tokens,
                        "loss": last_loss,
                        "loss_ema": self.loss_ema,
                        "learning_rate": learning_rate,
                        "gradient_norm": float(gradient_norm.item()),
                        "tokens_per_second": (
                            self.config.tokens_per_optimizer_step / step_seconds
                        ),
                        "step_seconds": step_seconds,
                        "gpu_memory_allocated_gib": allocated_gib,
                        "gpu_memory_reserved_gib": reserved_gib,
                        "optimizer_step_skipped": optimizer_step_skipped,
                    }
                    self.metrics.log(record)
                    self._log_tensorboard_train(record)
                    self._write_summary("running", last_loss, last_validation)
                    print(
                        f"step={self.step:,}/{self.max_steps:,} "
                        f"tokens={self.tokens_seen:,} loss={last_loss:.4f} "
                        f"lr={learning_rate:.2e} tok/s={record['tokens_per_second']:,.0f}",
                        flush=True,
                    )

                if self.step % training.evaluate_every_steps == 0:
                    last_validation = self.evaluate(
                        self.validation_corpus,
                        "general",
                        training.validation_batches,
                    )
                    self.best_validation_loss = min(
                        self.best_validation_loss, last_validation["loss"]
                    )
                    if self.structured_validation_corpus is not None:
                        self.evaluate(
                            self.structured_validation_corpus,
                            "structured",
                            training.validation_batches,
                        )
                    self._write_summary("running", last_loss, last_validation)

                if self.step % training.checkpoint_every_steps == 0:
                    checkpoint = self.save_checkpoint()
                    self.metrics.log(
                        {
                            "type": "event",
                            "event": "checkpoint_saved",
                            "checkpoint": str(checkpoint),
                            "step": self.step,
                            "tokens_seen": self.tokens_seen,
                        }
                    )

            last_validation = self.evaluate(
                self.validation_corpus,
                "general_final",
                training.final_validation_batches,
            )
            self.best_validation_loss = min(
                self.best_validation_loss, last_validation["loss"]
            )
            if self.structured_validation_corpus is not None:
                self.evaluate(
                    self.structured_validation_corpus,
                    "structured_final",
                    training.final_validation_batches,
                )
            final_checkpoint = self.save_checkpoint(label="final")
            self.metrics.log(
                {
                    "type": "event",
                    "event": "completed",
                    "checkpoint": str(final_checkpoint),
                    "step": self.step,
                    "tokens_seen": self.tokens_seen,
                }
            )
            self._write_summary("completed", last_loss, last_validation)
            return self.describe()
        except KeyboardInterrupt:
            checkpoint = self.save_checkpoint()
            self.metrics.log(
                {
                    "type": "event",
                    "event": "interrupted",
                    "checkpoint": str(checkpoint),
                    "step": self.step,
                    "tokens_seen": self.tokens_seen,
                }
            )
            self._write_summary("interrupted", last_loss, last_validation)
            raise
        except Exception as error:
            self._write_summary("failed", last_loss, last_validation, repr(error))
            self.metrics.log(
                {
                    "type": "event",
                    "event": "failed",
                    "step": self.step,
                    "tokens_seen": self.tokens_seen,
                    "error": repr(error),
                }
            )
            raise
        finally:
            if self.tensorboard is not None:
                self.tensorboard.flush()
                self.tensorboard.close()
