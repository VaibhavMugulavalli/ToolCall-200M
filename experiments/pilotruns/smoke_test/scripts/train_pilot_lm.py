#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    dropout: float = 0.0
    rope_base: float = 10_000.0


def load_run_config(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


class PackedShardDataset:
    def __init__(self, data_dir: Path, seq_len: int):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.shard_paths = sorted((data_dir / "shards").glob("shard_*.bin"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard_*.bin files found in {data_dir / 'shards'}")
        self.shards = [np.memmap(path, dtype=np.uint16, mode="r") for path in self.shard_paths]
        self.valid_shards = [arr for arr in self.shards if len(arr) > seq_len + 1]
        if not self.valid_shards:
            raise RuntimeError("No shard is large enough for the requested seq_len.")
        self.total_tokens = sum(len(arr) for arr in self.shards)

    def get_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for _ in range(batch_size):
            arr = random.choice(self.valid_shards)
            start = random.randint(0, len(arr) - self.seq_len - 1)
            chunk = np.asarray(arr[start:start + self.seq_len + 1], dtype=np.int64)
            xs.append(torch.from_numpy(chunk[:-1].copy()))
            ys.append(torch.from_numpy(chunk[1:].copy()))
        return torch.stack(xs).to(device), torch.stack(ys).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def precompute_rope(seq_len: int, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]


def apply_rope(x, cos, sin):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0
        assert config.num_heads % config.num_kv_heads == 0
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.repeat_factor = config.num_heads // config.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = config.dropout

    def forward(self, x, cos, sin):
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos[:, :, :t], sin[:, :, :t])
        k = apply_rope(k, cos[:, :, :t], sin[:, :, :t])
        if self.repeat_factor > 1:
            k = k.repeat_interleave(self.repeat_factor, dim=1)
            v = v.repeat_interleave(self.repeat_factor, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        return self.o_proj(y.transpose(1, 2).contiguous().view(b, t, c))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size)
        self.self_attn = CausalSelfAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_norm(x), cos, sin)
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class PilotLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        _, t = input_ids.shape
        x = self.tok_embeddings(input_ids)
        cos, sin = precompute_rope(t, self.config.hidden_size // self.config.num_heads, self.config.rope_base, input_ids.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def generate(model, sp, prompt: str, device: torch.device, max_new_tokens: int = 120):
    model.eval()
    ids = sp.encode(prompt, out_type=int) or [sp.bos_id() if sp.bos_id() >= 0 else 0]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids[:, -model.config.seq_len:])
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if int(next_id.item()) == sp.eos_id():
            break
    model.train()
    return sp.decode(input_ids[0].tolist())


def resolve_arg(config: dict, cli_value, key: str, default=None):
    return cli_value if cli_value is not None else config.get(key, default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_config = load_run_config(args.config)
    model_config_dict = run_config.get("model", {})

    data_dir = Path(resolve_arg(run_config, args.data_dir, "data_dir"))
    tokenizer_path = Path(resolve_arg(run_config, args.tokenizer, "tokenizer"))
    out_dir = Path(resolve_arg(run_config, args.out_dir, "out_dir", "runs/pilot"))

    seq_len = int(resolve_arg(run_config, args.seq_len, "seq_len", 512))
    batch_size = int(resolve_arg(run_config, args.batch_size, "batch_size", 8))
    grad_accum = int(resolve_arg(run_config, args.grad_accum, "grad_accum", 4))
    max_steps = int(resolve_arg(run_config, args.max_steps, "max_steps", 1000))
    eval_interval = int(resolve_arg(run_config, args.eval_interval, "eval_interval", 100))
    save_interval = int(resolve_arg(run_config, args.save_interval, "save_interval", 500))
    lr = float(resolve_arg(run_config, args.lr, "learning_rate", 3e-4))

    hidden_size = int(args.hidden_size or model_config_dict.get("hidden_size", 256))
    layers = int(args.layers or model_config_dict.get("num_layers", 6))
    heads = int(args.heads or model_config_dict.get("num_heads", 4))
    kv_heads = int(args.kv_heads or model_config_dict.get("num_kv_heads", 2))
    intermediate_size = int(args.intermediate_size or model_config_dict.get("intermediate_size", 768))
    dropout = float(args.dropout if args.dropout is not None else model_config_dict.get("dropout", 0.0))

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    dataset = PackedShardDataset(data_dir, seq_len=seq_len)
    model_cfg = ModelConfig(sp.vocab_size(), seq_len, hidden_size, layers, heads, kv_heads, intermediate_size, dropout)
    model = PilotLM(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    effective_batch_tokens = batch_size * seq_len * grad_accum

    metadata = {
        "run_config_path": str(args.config) if args.config else None,
        "data_dir": str(data_dir),
        "tokenizer": str(tokenizer_path),
        "out_dir": str(out_dir),
        "model_config": asdict(model_cfg),
        "parameters": count_parameters(model),
        "dataset_tokens": dataset.total_tokens,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch_tokens": effective_batch_tokens,
        "max_steps": max_steps,
        "learning_rate": lr,
        "device": str(device),
        "seed": args.seed,
    }

    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

    log_path = out_dir / "train_log.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    for step in range(1, max_steps + 1):
        step_loss = 0.0
        for _ in range(grad_accum):
            x, y = dataset.get_batch(batch_size, device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=amp_dtype):
                _, loss = model(x, labels=y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            step_loss += float(loss.detach().cpu()) * grad_accum

        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % eval_interval == 0:
            elapsed = time.time() - start_time
            tokens_seen = step * effective_batch_tokens
            tokens_per_second = tokens_seen / max(elapsed, 1e-9)
            record = {
                "step": step,
                "loss": step_loss,
                "grad_norm": grad_norm,
                "tokens_seen": tokens_seen,
                "tokens_per_second": tokens_per_second,
                "elapsed_seconds": elapsed,
            }
            print(f"step={step:05d} loss={step_loss:.4f} grad_norm={grad_norm:.3f} tokens={tokens_seen:,} tok/s={tokens_per_second:,.0f}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            prompt = "<|user|>\nFind unread emails from HR.\n<|tool_schema|>\n{\"name\":\"email.search_emails\"}\n<|tool_call|>\n"
            try:
                sample = generate(model, sp, prompt, device=device, max_new_tokens=80)
                (out_dir / f"sample_step_{step:06d}.txt").write_text(sample, encoding="utf-8")
                print("\n--- sample preview ---")
                print(sample[:1000])
                print("----------------------\n")
            except Exception as exc:
                print("Generation failed:", repr(exc))

        if step % save_interval == 0 or step == max_steps:
            ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": asdict(model_cfg), "metadata": metadata, "step": step}
            ckpt_path = out_dir / f"ckpt_step_{step:06d}.pt"
            torch.save(ckpt, ckpt_path)
            print("Saved checkpoint:", ckpt_path)

    print("Training complete.")


if __name__ == "__main__":
    main()
