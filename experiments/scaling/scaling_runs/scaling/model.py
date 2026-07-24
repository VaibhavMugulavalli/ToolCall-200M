from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaling.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().pow(2).mean(dim=-1, keepdim=True) + self.epsilon
        )
        return (normalized * self.weight.float()).to(dtype=inputs.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dimension: int, base: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            base
            ** (
                torch.arange(0, head_dimension, 2, dtype=torch.float32)
                / head_dimension
            )
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def cos_sin(
        self, sequence_length: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inverse_frequency.to(device))
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cos = angles.cos().to(dtype=dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin


def rotate_half(inputs: torch.Tensor) -> torch.Tensor:
    first, second = inputs.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dimension = config.hidden_size // config.num_heads
        self.dropout = config.dropout
        self.qkv_projection = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=False
        )
        self.output_projection = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.rotary = RotaryEmbedding(self.head_dimension, config.rope_base)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = inputs.shape
        qkv = self.qkv_projection(inputs)
        qkv = qkv.view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dimension,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        cos, sin = self.rotary.cos_sin(
            sequence_length, inputs.device, query.dtype
        )
        query = (query * cos) + (rotate_half(query) * sin)
        key = (key * cos) + (rotate_half(key) * sin)

        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, hidden_size
        )
        return self.output_projection(attention)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        intermediate_size = config.hidden_size * config.mlp_ratio
        self.input_projection = nn.Linear(
            config.hidden_size, intermediate_size, bias=False
        )
        self.output_projection = nn.Linear(
            intermediate_size, config.hidden_size, bias=False
        )
        self.dropout = config.dropout

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.input_projection(inputs), approximate="tanh")
        hidden = self.output_projection(hidden)
        return F.dropout(hidden, p=self.dropout, training=self.training)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(
            config.hidden_size, config.rms_norm_epsilon
        )
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = RMSNorm(
            config.hidden_size, config.rms_norm_epsilon
        )
        self.feed_forward = FeedForward(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        inputs = inputs + self.feed_forward(self.feed_forward_norm(inputs))
        return inputs


class ToolCallLanguageModel(nn.Module):
    """Bias-free RoPE/RMSNorm causal Transformer used by the scaling family."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)
        self._scale_residual_projections()

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        scale = 0.02 / math.sqrt(2.0 * self.config.num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.output_projection.weight, 0.0, scale)
            nn.init.normal_(block.feed_forward.output_projection.weight, 0.0, scale)

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
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)

        output = {"logits": logits}
        if labels is not None:
            output["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
        return output

