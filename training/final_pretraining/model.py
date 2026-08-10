from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from experiments.scaling.extended_runs.config import ModelConfig as CoreModelConfig
from experiments.scaling.extended_runs.model import ToolCallLanguageModel

from .config import ModelConfig


def to_core_config(config: ModelConfig) -> CoreModelConfig:
    return CoreModelConfig.from_dict(
        {
            "vocab_size": config.vocab_size,
            "sequence_length": config.sequence_length,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "activation": config.activation,
            "rope_base": config.rope_base,
            "rms_norm_epsilon": config.rms_norm_epsilon,
            "dropout": config.dropout,
            "tie_embeddings": config.tie_embeddings,
        }
    )


class FinalToolCallLanguageModel(ToolCallLanguageModel):
    """Frozen 200M architecture with optional per-block activation recomputation."""

    def __init__(self, final_config: ModelConfig) -> None:
        super().__init__(to_core_config(final_config))
        self.final_config = final_config

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.sequence_length:
            raise ValueError("input sequence exceeds configured sequence_length")

        hidden = self.embedding_dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            if self.training and self.final_config.gradient_checkpointing:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        if labels is None:
            return {"logits": logits}
        return {
            "loss": F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
        }


def load_exported_model(
    export_directory: str | Path,
    map_location: str | torch.device = "cpu",
) -> FinalToolCallLanguageModel:
    """Load a consolidated final/milestone export without DeepSpeed."""

    export_directory = Path(export_directory)
    with (export_directory / "model_config.json").open("r", encoding="utf-8") as handle:
        config = ModelConfig.from_dict(json.load(handle))
    model = FinalToolCallLanguageModel(config)

    safetensors_path = export_directory / "model.safetensors"
    pytorch_path = export_directory / "pytorch_model.bin"
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_path), device=str(map_location))
    elif pytorch_path.is_file():
        state = torch.load(pytorch_path, map_location=map_location, weights_only=True)
    else:
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin found in {export_directory}"
        )
    model.load_state_dict(state, strict=True)
    model.to(map_location)
    model.eval()
    return model
