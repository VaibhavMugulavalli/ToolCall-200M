from __future__ import annotations

import json
import math
import os
import random
import shutil
import signal
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from .config import FinalRunConfig, TokenPlan
from .data import ShardedTokenCorpus, discover_one, rank_batch_start
from .hub import download_latest_checkpoint, upload_latest_checkpoint, validate_hub_repo
from .model import FinalToolCallLanguageModel


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def safe_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def cosine_learning_rate(config: FinalRunConfig, plan: TokenPlan, step: int) -> float:
    training = config.training
    if step <= plan.warmup_steps:
        return training.max_learning_rate * step / plan.warmup_steps
    progress = (step - plan.warmup_steps) / (plan.optimizer_steps - plan.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return training.minimum_learning_rate + coefficient * (
        training.max_learning_rate - training.minimum_learning_rate
    )


class DistributedTrainer:
    def __init__(
        self,
        *,
        config: FinalRunConfig,
        config_path: Path,
        deepspeed_config_path: Path,
        data_root: Path,
        run_directory: Path,
        hub_repo_id: str | None,
        hub_token: str | None,
        resume_mode: str,
        allow_no_wandb: bool,
        qualification: bool,
    ) -> None:
        self.config = config
        self.config_path = config_path.resolve()
        self.deepspeed_config_path = deepspeed_config_path.resolve()
        self.data_root = data_root.resolve()
        self.run_directory = run_directory.resolve()
        self.checkpoint_root = self.run_directory / "checkpoints"
        self.hub_repo_id = hub_repo_id
        self.hub_token = hub_token
        self.resume_mode = resume_mode
        self.allow_no_wandb = allow_no_wandb
        self.qualification = qualification
        self.stop_requested = False
        self.started_at = utc_timestamp()
        self.session_started = time.monotonic()
        self.total_training_seconds = 0.0
        self.loss_ema: float | None = None
        self.best_general_loss = math.inf
        self.best_structured_loss = math.inf
        self.optimizer_step = 0
        self.global_source_cursor = 0
        self.prediction_tokens_seen = 0
        self.wandb_run_id: str | None = None
        self.last_hub_sync_step = 0
        self.last_checkpoint_monotonic = self.session_started
        self.last_hub_sync_monotonic = self.session_started

        self._initialize_distributed()
        self._validate_runtime_inputs()
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)

        self.train_corpus = ShardedTokenCorpus(
            self.data_root, self.config.data.train_patterns, self.config.data.token_dtype
        )
        self.general_corpus = ShardedTokenCorpus(
            self.data_root,
            self.config.data.validation_general_patterns,
            self.config.data.token_dtype,
        )
        self.structured_corpus = ShardedTokenCorpus(
            self.data_root,
            self.config.data.validation_structured_patterns,
            self.config.data.token_dtype,
        )
        self.tokenizer_path = discover_one(
            self.data_root, self.config.data.tokenizer_patterns, "SentencePiece model"
        )
        self._validate_dataset()
        self.plan = TokenPlan.from_corpus(self.config, self.train_corpus.total_tokens)

        self._seed_for_model_initialization()
        self.raw_model = FinalToolCallLanguageModel(self.config.model)
        parameter_count = sum(parameter.numel() for parameter in self.raw_model.parameters())
        if parameter_count != self.config.model.expected_parameter_count:
            raise RuntimeError(
                f"Instantiated model has {parameter_count:,} parameters; expected "
                f"{self.config.model.expected_parameter_count:,}"
            )
        self.parameter_count = parameter_count
        optimizer = self._build_optimizer(self.raw_model)
        deepspeed_config = self._resolved_deepspeed_config()

        import deepspeed

        self.engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.raw_model,
            optimizer=optimizer,
            config=deepspeed_config,
            dist_init_required=False,
        )
        self.device = self.engine.device
        self._prepare_resume_checkpoint()
        self._load_or_initialize_state()
        self._install_signal_handlers()
        self.tensorboard: SummaryWriter | None = None
        self.wandb_run: Any | None = None
        self._initialize_tracking()
        self._write_static_metadata(deepspeed_config)

    def _initialize_distributed(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Final pretraining requires CUDA")
        if torch.cuda.device_count() < self.config.training.expected_world_size:
            raise RuntimeError(
                f"Expected {self.config.training.expected_world_size} visible GPUs, found "
                f"{torch.cuda.device_count()}"
            )
        if torch.cuda.device_count() >= 2:
            try:
                peer_access = torch.cuda.can_device_access_peer(0, 1)
            except RuntimeError:
                peer_access = False
            if not peer_access:
                os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")

        import deepspeed

        deepspeed.init_distributed(dist_backend="nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        torch.cuda.set_device(self.local_rank)
        if self.world_size != self.config.training.expected_world_size:
            raise RuntimeError(
                f"Launch with exactly {self.config.training.expected_world_size} ranks; "
                f"got {self.world_size}"
            )

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def _validate_runtime_inputs(self) -> None:
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Kaggle dataset root does not exist: {self.data_root}")
        if self.resume_mode not in {"auto", "none", "required"}:
            raise ValueError("resume_mode must be auto, none, or required")
        if not self.qualification:
            if not self.hub_repo_id or not self.hub_token:
                raise RuntimeError(
                    "Real training requires HF_REPO_ID and the HF_TOKEN Kaggle secret"
                )
            error: list[str | None] = [None]
            if self.is_main:
                try:
                    validate_hub_repo(self.hub_repo_id, self.hub_token)
                except Exception as exception:
                    error[0] = f"{type(exception).__name__}: {exception}"
            dist.broadcast_object_list(error, src=0)
            if error[0] is not None:
                raise RuntimeError(f"Cannot access Hugging Face repo: {error[0]}")
        if not self.qualification and not self.allow_no_wandb:
            if not os.environ.get("WANDB_API_KEY"):
                raise RuntimeError(
                    "WANDB_API_KEY is missing. Add the Kaggle secret so the long run "
                    "always has live graphs."
                )

    def _validate_dataset(self) -> None:
        expected = {
            "train": self.config.data.expected_train_source_tokens,
            "validation_general": self.config.data.expected_validation_general_tokens,
            "validation_structured": self.config.data.expected_validation_structured_tokens,
        }
        actual = {
            "train": self.train_corpus.total_tokens,
            "validation_general": self.general_corpus.total_tokens,
            "validation_structured": self.structured_corpus.total_tokens,
        }
        if actual != expected:
            raise RuntimeError(
                "Dataset token counts differ from the frozen run configuration: "
                f"expected={expected}, actual={actual}"
            )
        import sentencepiece as spm

        processor = spm.SentencePieceProcessor(model_file=str(self.tokenizer_path))
        if processor.vocab_size() != self.config.model.vocab_size:
            raise RuntimeError(
                f"Tokenizer vocab is {processor.vocab_size():,}; expected "
                f"{self.config.model.vocab_size:,}"
            )

    def _seed_for_model_initialization(self) -> None:
        seed = self.config.training.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for parameter in model.parameters():
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
        try:
            return torch.optim.AdamW(groups, fused=True, **options)
        except (TypeError, RuntimeError):
            return torch.optim.AdamW(groups, **options)

    def _resolved_deepspeed_config(self) -> dict[str, Any]:
        with self.deepspeed_config_path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        training = self.config.training
        values["train_micro_batch_size_per_gpu"] = training.micro_batch_size_per_gpu
        values["gradient_accumulation_steps"] = training.gradient_accumulation_steps
        values["train_batch_size"] = (
            training.micro_batch_size_per_gpu
            * training.gradient_accumulation_steps
            * training.expected_world_size
        )
        values["gradient_clipping"] = training.gradient_clip
        return values

    def _prepare_resume_checkpoint(self) -> None:
        local_exists = (self.checkpoint_root / "latest").is_file()
        if self.resume_mode == "none":
            if local_exists:
                raise RuntimeError("--resume none was used but local checkpoint state exists")
            return
        if not local_exists and not self.qualification:
            payload: list[str | None] = [None, None]
            if self.is_main:
                try:
                    downloaded = download_latest_checkpoint(
                        self.hub_repo_id or "",
                        self.hub_token or "",
                        self.config.tracking.hub_checkpoint_path,
                        self.run_directory / "hub-download",
                    )
                    if downloaded is not None:
                        self.checkpoint_root = downloaded
                    payload[0] = str(self.checkpoint_root)
                except Exception as exception:
                    payload[1] = f"{type(exception).__name__}: {exception}"
            dist.broadcast_object_list(payload, src=0)
            if payload[1] is not None:
                raise RuntimeError(f"Hugging Face checkpoint download failed: {payload[1]}")
            self.checkpoint_root = Path(payload[0] or self.checkpoint_root)
        dist.barrier()
        if self.resume_mode == "required" and not (self.checkpoint_root / "latest").is_file():
            raise RuntimeError("A resume checkpoint was required, but none exists on the Hub")

    def _load_or_initialize_state(self) -> None:
        latest = self.checkpoint_root / "latest"
        if not latest.is_file():
            self.wandb_run_id = uuid.uuid4().hex[:12] if self.is_main else None
            payload: list[str | None] = [self.wandb_run_id]
            dist.broadcast_object_list(payload, src=0)
            self.wandb_run_id = payload[0]
            return

        load_path, client_state = self.engine.load_checkpoint(
            str(self.checkpoint_root),
            tag=None,
            load_optimizer_states=True,
            load_lr_scheduler_states=False,
            load_module_strict=True,
        )
        if load_path is None or not isinstance(client_state, dict):
            raise RuntimeError("DeepSpeed did not return a valid checkpoint client_state")
        self._validate_client_state(client_state)
        self.optimizer_step = int(client_state["optimizer_step"])
        self.global_source_cursor = int(client_state["global_source_cursor"])
        self.prediction_tokens_seen = int(client_state["prediction_tokens_seen"])
        self.total_training_seconds = float(client_state.get("total_training_seconds", 0.0))
        self.loss_ema = client_state.get("loss_ema")
        self.best_general_loss = float(client_state.get("best_general_loss", math.inf))
        self.best_structured_loss = float(client_state.get("best_structured_loss", math.inf))
        self.wandb_run_id = str(client_state["wandb_run_id"])
        self.last_hub_sync_step = int(client_state.get("last_hub_sync_step", 0))
        self._restore_rng_state(client_state.get("rng_states"))
        if self.is_main:
            print(f"Resumed from {load_path} at optimizer step {self.optimizer_step:,}")

    def _validate_client_state(self, state: dict[str, Any]) -> None:
        checks = {
            "config_fingerprint": self.config.fingerprint(),
            "train_fingerprint": self.train_corpus.fingerprint(),
            "general_fingerprint": self.general_corpus.fingerprint(),
            "structured_fingerprint": self.structured_corpus.fingerprint(),
            "world_size": self.world_size,
        }
        for key, expected in checks.items():
            if state.get(key) != expected:
                raise RuntimeError(
                    f"Checkpoint {key} mismatch: {state.get(key)!r} != {expected!r}"
                )
        step = int(state["optimizer_step"])
        if int(state["prediction_tokens_seen"]) != step * self.plan.prediction_tokens_per_step:
            raise RuntimeError("Checkpoint prediction-token counter is inconsistent")
        if int(state["global_source_cursor"]) != step * self.plan.source_tokens_per_step:
            raise RuntimeError("Checkpoint source-token cursor is inconsistent")

    def _capture_rng_states(self) -> list[dict[str, Any]]:
        local = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(self.device),
        }
        gathered: list[dict[str, Any] | None] = [None] * self.world_size
        dist.all_gather_object(gathered, local)
        return [item for item in gathered if item is not None]

    def _restore_rng_state(self, states: Any) -> None:
        if not isinstance(states, list) or len(states) != self.world_size:
            raise RuntimeError("Checkpoint does not contain per-rank RNG states")
        state = states[self.rank]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        torch.cuda.set_rng_state(state["cuda"].cpu(), self.device)

    def _initialize_tracking(self) -> None:
        error: list[str | None] = [None]
        if self.is_main:
            try:
                self.tensorboard = SummaryWriter(
                    log_dir=str(self.run_directory / "tensorboard"),
                    purge_step=self.optimizer_step if self.optimizer_step else None,
                    flush_secs=30,
                )
                use_wandb = not self.qualification and not (
                    self.allow_no_wandb and not os.environ.get("WANDB_API_KEY")
                )
                if use_wandb:
                    import wandb

                    self.wandb_run = wandb.init(
                        project=os.environ.get(
                            "WANDB_PROJECT", self.config.tracking.wandb_project
                        ),
                        entity=os.environ.get("WANDB_ENTITY") or None,
                        id=self.wandb_run_id,
                        resume="allow",
                        name=self.config.tracking.wandb_run_name,
                        config=self.config.to_dict(),
                        tags=[
                            "pretraining",
                            "200m",
                            "2xt4",
                            "zero2",
                            "from-scratch",
                        ],
                    )
                    self.wandb_run.define_metric("prediction_tokens_seen")
                    self.wandb_run.define_metric(
                        "train/*", step_metric="prediction_tokens_seen"
                    )
                    self.wandb_run.define_metric(
                        "validation/*", step_metric="prediction_tokens_seen"
                    )
                    self.wandb_run.define_metric(
                        "performance/*", step_metric="prediction_tokens_seen"
                    )
                    self.wandb_run.define_metric(
                        "memory/*", step_metric="prediction_tokens_seen"
                    )
                    print(f"Live graphs: {self.wandb_run.url}")
            except Exception as exception:
                error[0] = f"{type(exception).__name__}: {exception}"
        dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(f"Metrics initialization failed: {error[0]}")

    def _write_static_metadata(self, deepspeed_config: dict[str, Any]) -> None:
        if not self.is_main:
            return
        atomic_write_json(self.run_directory / "config.json", self.config.to_dict())
        atomic_write_json(
            self.run_directory / "model_config.json", asdict(self.config.model)
        )
        (self.run_directory / "wandb_run_id.txt").write_text(
            f"{self.wandb_run_id}\n", encoding="utf-8"
        )
        manifest = {
            "version": 1,
            "created_at": self.started_at,
            "config_fingerprint": self.config.fingerprint(),
            "parameter_count": self.parameter_count,
            "token_plan": asdict(self.plan),
            "data": {
                "train": asdict(self.train_corpus.describe()),
                "validation_general": asdict(self.general_corpus.describe()),
                "validation_structured": asdict(self.structured_corpus.describe()),
                "tokenizer": self.tokenizer_path.relative_to(self.data_root).as_posix(),
            },
            "deepspeed": deepspeed_config,
            "gpu_names": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
        atomic_write_json(self.run_directory / "manifest.json", manifest)
        self._write_summary("initialized")
        print(json.dumps(manifest, indent=2))

    def _install_signal_handlers(self) -> None:
        def request_stop(signum: int, _frame: Any) -> None:
            self.stop_requested = True
            if self.is_main:
                print(f"Received signal {signum}; checkpointing after this optimizer step")

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

    def _set_learning_rate(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def _distributed_mean(self, value: float, count: int) -> float:
        tensor = torch.tensor([value, float(count)], dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float((tensor[0] / tensor[1]).item())

    def _distributed_memory(self) -> tuple[float, float, float]:
        gib = 1024**3
        local = torch.tensor(
            [
                torch.cuda.memory_allocated(self.device) / gib,
                torch.cuda.memory_reserved(self.device) / gib,
                torch.cuda.max_memory_reserved(self.device) / gib,
            ],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(local, op=dist.ReduceOp.MAX)
        return tuple(float(value) for value in local.tolist())  # type: ignore[return-value]

    def _training_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        training = self.config.training
        start = rank_batch_start(
            self.global_source_cursor,
            self.rank,
            training.micro_batch_size_per_gpu,
            self.config.model.sequence_length,
        )
        batch = self.train_corpus.batch(
            start,
            training.micro_batch_size_per_gpu,
            self.config.model.sequence_length,
            self.device,
        )
        local_source_tokens = training.micro_batch_size_per_gpu * (
            self.config.model.sequence_length + 1
        )
        self.global_source_cursor += local_source_tokens * self.world_size
        return batch

    @torch.no_grad()
    def evaluate(
        self,
        corpus: ShardedTokenCorpus,
        split: str,
        batches_per_rank: int,
    ) -> dict[str, Any]:
        self.engine.eval()
        batch_size = self.config.training.micro_batch_size_per_gpu
        sequence_length = self.config.model.sequence_length
        local_source_tokens = batch_size * (sequence_length + 1)
        global_source_tokens = local_source_tokens * self.world_size
        batch_count = min(batches_per_rank, corpus.total_tokens // global_source_tokens)
        if batch_count < 1:
            raise RuntimeError(f"Validation split {split!r} cannot provide one global batch")
        local_loss_sum = 0.0
        for batch_index in range(batch_count):
            start = rank_batch_start(
                batch_index * global_source_tokens,
                self.rank,
                batch_size,
                sequence_length,
            )
            inputs, targets = corpus.batch(start, batch_size, sequence_length, self.device)
            loss = self.engine(inputs, labels=targets)["loss"]
            local_loss_sum += float(loss.item())
        mean_loss = self._distributed_mean(local_loss_sum, batch_count)
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"Non-finite {split} validation loss: {mean_loss}")
        record = {
            "type": "validation",
            "split": split,
            "optimizer_step": self.optimizer_step,
            "prediction_tokens_seen": self.prediction_tokens_seen,
            "loss": mean_loss,
            "perplexity": safe_perplexity(mean_loss),
            "tokens_evaluated": (
                batch_count * self.world_size * batch_size * sequence_length
            ),
        }
        if self.is_main:
            append_jsonl(self.run_directory / "metrics.jsonl", record)
            assert self.tensorboard is not None
            self.tensorboard.add_scalar(
                f"validation/{split}_loss", mean_loss, self.optimizer_step
            )
            self.tensorboard.add_scalar(
                f"validation/{split}_perplexity", record["perplexity"], self.optimizer_step
            )
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "prediction_tokens_seen": self.prediction_tokens_seen,
                        f"validation/{split}_loss": mean_loss,
                        f"validation/{split}_perplexity": record["perplexity"],
                    }
                )
        self.engine.train()
        return record

    def _client_state(self, rng_states: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "version": 1,
            "config_fingerprint": self.config.fingerprint(),
            "train_fingerprint": self.train_corpus.fingerprint(),
            "general_fingerprint": self.general_corpus.fingerprint(),
            "structured_fingerprint": self.structured_corpus.fingerprint(),
            "world_size": self.world_size,
            "optimizer_step": self.optimizer_step,
            "global_source_cursor": self.global_source_cursor,
            "prediction_tokens_seen": self.prediction_tokens_seen,
            "total_training_seconds": self.total_training_seconds,
            "loss_ema": self.loss_ema,
            "best_general_loss": self.best_general_loss,
            "best_structured_loss": self.best_structured_loss,
            "wandb_run_id": self.wandb_run_id,
            "last_hub_sync_step": self.last_hub_sync_step,
            "rng_states": rng_states,
        }

    def save_checkpoint(self, reason: str, sync_to_hub: bool) -> str:
        if sync_to_hub:
            self.last_hub_sync_step = self.optimizer_step
        rng_states = self._capture_rng_states()
        tag = f"global_step{self.optimizer_step:08d}"
        client_state = self._client_state(rng_states)
        self.engine.save_checkpoint(
            str(self.checkpoint_root),
            tag=tag,
            client_state=client_state,
            save_latest=True,
            exclude_frozen_parameters=False,
        )
        dist.barrier()
        if self.is_main:
            self._remove_old_local_checkpoints(tag)
            event = {
                "type": "event",
                "event": "checkpoint_saved",
                "reason": reason,
                "tag": tag,
                "optimizer_step": self.optimizer_step,
                "prediction_tokens_seen": self.prediction_tokens_seen,
                "sync_to_hub": sync_to_hub,
                "timestamp": utc_timestamp(),
            }
            append_jsonl(self.run_directory / "metrics.jsonl", event)
            self._write_summary("checkpointing")
        dist.barrier()

        upload_error: list[str | None] = [None]
        if sync_to_hub and self.is_main:
            try:
                url = upload_latest_checkpoint(
                    repo_id=self.hub_repo_id or "",
                    token=self.hub_token or "",
                    checkpoint_root=self.checkpoint_root,
                    checkpoint_tag=tag,
                    checkpoint_path=self.config.tracking.hub_checkpoint_path,
                    run_directory=self.run_directory,
                    artifact_files={
                        "model_config.json": self.run_directory / "model_config.json",
                        "run_config.json": self.config_path,
                        "deepspeed_zero2.json": self.deepspeed_config_path,
                        "toolcall_spm_32k.model": self.tokenizer_path,
                    },
                    commit_message=(
                        f"Resume checkpoint at step {self.optimizer_step:,} "
                        f"({self.prediction_tokens_seen:,} tokens)"
                    ),
                )
                append_jsonl(
                    self.run_directory / "metrics.jsonl",
                    {
                        "type": "event",
                        "event": "hub_sync_complete",
                        "tag": tag,
                        "url": url,
                        "optimizer_step": self.optimizer_step,
                        "timestamp": utc_timestamp(),
                    },
                )
                print(f"Checkpoint synced to Hugging Face: {url}")
            except Exception as error:  # propagate the rank-0 network failure to all ranks
                upload_error[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(upload_error, src=0)
        if upload_error[0] is not None:
            raise RuntimeError(f"Hugging Face checkpoint sync failed: {upload_error[0]}")
        self.last_checkpoint_monotonic = time.monotonic()
        if sync_to_hub:
            self.last_hub_sync_monotonic = self.last_checkpoint_monotonic
        return tag

    def _remove_old_local_checkpoints(self, current_tag: str) -> None:
        keep = self.config.training.keep_last_local_checkpoints
        directories = sorted(
            path
            for path in self.checkpoint_root.glob("global_step*")
            if path.is_dir() and path.name != current_tag
        )
        for path in directories[: max(0, len(directories) - (keep - 1))]:
            shutil.rmtree(path)

    def _write_summary(
        self,
        status: str,
        last_loss: float | None = None,
        stop_reason: str | None = None,
    ) -> None:
        if not self.is_main:
            return
        summary = {
            "run_name": self.config.run_name,
            "status": status,
            "started_at": self.started_at,
            "updated_at": utc_timestamp(),
            "optimizer_step": self.optimizer_step,
            "optimizer_steps_total": self.plan.optimizer_steps,
            "prediction_tokens_seen": self.prediction_tokens_seen,
            "prediction_tokens_total": self.plan.total_prediction_tokens,
            "global_source_cursor": self.global_source_cursor,
            "progress_fraction": self.prediction_tokens_seen
            / self.plan.total_prediction_tokens,
            "last_loss": last_loss,
            "loss_ema": self.loss_ema,
            "best_general_loss": None
            if not math.isfinite(self.best_general_loss)
            else self.best_general_loss,
            "best_structured_loss": None
            if not math.isfinite(self.best_structured_loss)
            else self.best_structured_loss,
            "total_training_seconds": self.total_training_seconds,
            "last_hub_sync_step": self.last_hub_sync_step,
            "wandb_run_id": self.wandb_run_id,
            "stop_reason": stop_reason,
        }
        atomic_write_json(self.run_directory / "summary.json", summary)

    def _log_train_step(
        self,
        loss: float,
        learning_rate: float,
        step_seconds: float,
    ) -> None:
        allocated, reserved, peak_reserved = self._distributed_memory()
        tokens_per_second = self.plan.prediction_tokens_per_step / step_seconds
        remaining_tokens = self.plan.total_prediction_tokens - self.prediction_tokens_seen
        remaining_hours = remaining_tokens / max(tokens_per_second, 1.0) / 3600.0
        overflow = bool(getattr(self.optimizer, "overflow", False))
        loss_scale = getattr(self.optimizer, "cur_scale", None)
        record = {
            "type": "train",
            "optimizer_step": self.optimizer_step,
            "optimizer_steps_total": self.plan.optimizer_steps,
            "prediction_tokens_seen": self.prediction_tokens_seen,
            "prediction_tokens_total": self.plan.total_prediction_tokens,
            "global_source_cursor": self.global_source_cursor,
            "loss": loss,
            "loss_ema": self.loss_ema,
            "learning_rate": learning_rate,
            "tokens_per_second": tokens_per_second,
            "step_seconds": step_seconds,
            "estimated_remaining_hours": remaining_hours,
            "gpu_memory_allocated_gib_max": allocated,
            "gpu_memory_reserved_gib_max": reserved,
            "gpu_peak_reserved_gib_max": peak_reserved,
            "fp16_overflow": overflow,
            "loss_scale": float(loss_scale) if loss_scale is not None else None,
        }
        if self.is_main:
            append_jsonl(self.run_directory / "metrics.jsonl", record)
            assert self.tensorboard is not None
            tensorboard_fields = {
                "train/loss": loss,
                "train/loss_ema": self.loss_ema,
                "train/learning_rate": learning_rate,
                "performance/tokens_per_second": tokens_per_second,
                "performance/step_seconds": step_seconds,
                "performance/estimated_remaining_hours": remaining_hours,
                "memory/allocated_gib_max": allocated,
                "memory/reserved_gib_max": reserved,
                "memory/peak_reserved_gib_max": peak_reserved,
                "progress/prediction_tokens_seen": self.prediction_tokens_seen,
            }
            for name, value in tensorboard_fields.items():
                if value is not None and math.isfinite(float(value)):
                    self.tensorboard.add_scalar(name, value, self.optimizer_step)
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "prediction_tokens_seen": self.prediction_tokens_seen,
                        "train/loss": loss,
                        "train/loss_ema": self.loss_ema,
                        "train/learning_rate": learning_rate,
                        "performance/tokens_per_second": tokens_per_second,
                        "performance/step_seconds": step_seconds,
                        "performance/estimated_remaining_hours": remaining_hours,
                        "memory/allocated_gib_max": allocated,
                        "memory/reserved_gib_max": reserved,
                        "memory/peak_reserved_gib_max": peak_reserved,
                        "train/fp16_overflow": int(overflow),
                    }
                )
            self._write_summary("running", loss)
            print(
                f"step={self.optimizer_step:,}/{self.plan.optimizer_steps:,} "
                f"tokens={self.prediction_tokens_seen:,} loss={loss:.4f} "
                f"lr={learning_rate:.2e} tok/s={tokens_per_second:,.0f} "
                f"peak={peak_reserved:.2f}GiB eta={remaining_hours:.1f}h",
                flush=True,
            )

    def _session_stop_reason(self) -> str | None:
        local_code = 2 if self.stop_requested else 0
        elapsed_minutes = (time.monotonic() - self.session_started) / 60.0
        if elapsed_minutes >= self.config.training.max_session_minutes:
            local_code = max(local_code, 1)
        code = torch.tensor(local_code, dtype=torch.int32, device=self.device)
        dist.all_reduce(code, op=dist.ReduceOp.MAX)
        resolved = int(code.item())
        if resolved == 2:
            return "signal"
        if resolved == 1:
            return "session_time_limit"
        return None

    def train(self, qualification_steps: int | None = None) -> dict[str, Any]:
        if qualification_steps is not None and qualification_steps <= 0:
            raise ValueError("qualification_steps must be positive")
        if self.optimizer_step >= self.plan.optimizer_steps:
            if self.is_main:
                print("The full token plan is already complete.")
            return self._finalize_completed_run()

        target_step = self.plan.optimizer_steps
        if qualification_steps is not None:
            target_step = min(target_step, self.optimizer_step + qualification_steps)
        self.engine.train()
        last_loss: float | None = None

        while self.optimizer_step < target_step:
            stop_reason = None if self.qualification else self._session_stop_reason()
            if stop_reason is not None:
                self.save_checkpoint(stop_reason, sync_to_hub=True)
                self._write_summary("paused", last_loss, stop_reason)
                if self.wandb_run is not None:
                    self.wandb_run.finish(exit_code=0)
                return self.describe()

            step_started = time.perf_counter()
            next_step = self.optimizer_step + 1
            learning_rate = cosine_learning_rate(self.config, self.plan, next_step)
            self._set_learning_rate(learning_rate)
            local_loss_sum = 0.0
            for _ in range(self.config.training.gradient_accumulation_steps):
                inputs, targets = self._training_batch()
                loss = self.engine(inputs, labels=targets)["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Rank {self.rank} produced non-finite loss at step {next_step}"
                    )
                self.engine.backward(loss)
                self.engine.step()
                local_loss_sum += float(loss.detach().item())

            torch.cuda.synchronize(self.device)
            step_seconds = time.perf_counter() - step_started
            self.total_training_seconds += step_seconds
            self.optimizer_step = next_step
            self.prediction_tokens_seen = (
                self.optimizer_step * self.plan.prediction_tokens_per_step
            )
            expected_cursor = self.optimizer_step * self.plan.source_tokens_per_step
            if self.global_source_cursor != expected_cursor:
                raise RuntimeError("Distributed source cursor drifted from the token plan")
            mean_loss = self._distributed_mean(
                local_loss_sum, self.config.training.gradient_accumulation_steps
            )
            last_loss = mean_loss
            self.loss_ema = (
                mean_loss
                if self.loss_ema is None
                else 0.98 * self.loss_ema + 0.02 * mean_loss
            )
            should_log = (
                self.optimizer_step == 1
                or self.optimizer_step % self.config.training.log_every_steps == 0
            )
            if should_log:
                self._log_train_step(mean_loss, learning_rate, step_seconds)

            if (
                not self.qualification
                and self.optimizer_step % self.config.training.evaluate_every_steps == 0
            ):
                general = self.evaluate(
                    self.general_corpus,
                    "general",
                    self.config.training.validation_batches_per_rank,
                )
                structured = self.evaluate(
                    self.structured_corpus,
                    "structured",
                    self.config.training.validation_batches_per_rank,
                )
                self.best_general_loss = min(self.best_general_loss, general["loss"])
                self.best_structured_loss = min(
                    self.best_structured_loss, structured["loss"]
                )

            now = time.monotonic()
            local_due = (
                now - self.last_checkpoint_monotonic
                >= 60.0 * self.config.training.local_checkpoint_interval_minutes
            )
            hub_due = (
                now - self.last_hub_sync_monotonic
                >= 60.0 * self.config.training.hub_sync_interval_minutes
            )
            if not self.qualification and (local_due or hub_due):
                self.save_checkpoint(
                    "periodic_hub" if hub_due else "periodic_local",
                    sync_to_hub=hub_due,
                )

        if self.qualification:
            tag = self.save_checkpoint("qualification", sync_to_hub=False)
            _, state = self.engine.load_checkpoint(
                str(self.checkpoint_root), tag=tag, load_module_strict=True
            )
            self._validate_client_state(state)
            if self.is_main:
                print(
                    "QUALIFICATION PASSED: 2xGPU forward/backward, ZeRO-2, "
                    "rank-disjoint data, and local checkpoint reload are healthy."
                )
            return self.describe()
        return self._finalize_completed_run()

    def _finalize_completed_run(self) -> dict[str, Any]:
        general = self.evaluate(
            self.general_corpus,
            "general_final",
            self.config.training.final_validation_batches_per_rank,
        )
        structured = self.evaluate(
            self.structured_corpus,
            "structured_final",
            self.config.training.final_validation_batches_per_rank,
        )
        self.best_general_loss = min(self.best_general_loss, general["loss"])
        self.best_structured_loss = min(self.best_structured_loss, structured["loss"])
        self.save_checkpoint("completed", sync_to_hub=not self.qualification)
        self._write_summary("completed")
        if self.wandb_run is not None:
            self.wandb_run.finish(exit_code=0)
        if self.tensorboard is not None:
            self.tensorboard.close()
        if self.is_main:
            print(
                f"PRETRAINING COMPLETE: {self.prediction_tokens_seen:,} prediction "
                f"tokens across {self.optimizer_step:,} optimizer steps."
            )
        return self.describe()

    def describe(self) -> dict[str, Any]:
        return {
            "run_name": self.config.run_name,
            "rank": self.rank,
            "world_size": self.world_size,
            "parameter_count": self.parameter_count,
            "optimizer_step": self.optimizer_step,
            "prediction_tokens_seen": self.prediction_tokens_seen,
            "token_plan": asdict(self.plan),
            "checkpoint_root": str(self.checkpoint_root),
            "hub_repo_id": self.hub_repo_id,
            "wandb_run_id": self.wandb_run_id,
        }
