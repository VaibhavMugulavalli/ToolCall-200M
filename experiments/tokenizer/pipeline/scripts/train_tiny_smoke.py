#!/usr/bin/env python3
"""
Tiny Llama-style smoke trainer for ToolCall-200M token shards.
This validates shards, dataloader, causal LM loss, checkpointing and generation.
It is not the final 200M trainer.
"""
from __future__ import annotations
import argparse, json, random, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

class PackedShardDataset:
    def __init__(self, data_dir: Path, seq_len: int):
        self.seq_len = seq_len
        self.shard_paths = sorted((data_dir / "shards").glob("shard_*.bin"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard_*.bin files found in {data_dir / 'shards'}")
        self.shards = [np.memmap(path, dtype=np.uint16, mode="r") for path in self.shard_paths]
        self.valid_shards = [arr for arr in self.shards if len(arr) > seq_len + 1]
        if not self.valid_shards:
            raise RuntimeError("No shard is large enough for requested seq_len.")
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

@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int = 512
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 4
    num_kv_heads: int = 2
    intermediate_size: int = 768
    dropout: float = 0.0
    rope_base: float = 10000.0

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

def precompute_rope(seq_len: int, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]

def apply_rope(x, cos, sin):
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.hidden_size % cfg.num_heads == 0 and cfg.num_heads % cfg.num_kv_heads == 0
        self.num_heads, self.num_kv_heads = cfg.num_heads, cfg.num_kv_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.repeat_factor = cfg.num_heads // cfg.num_kv_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.dropout = cfg.dropout
    def forward(self, x, cos, sin):
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos[:, :, :t], sin[:, :, :t]); k = apply_rope(k, cos[:, :, :t], sin[:, :, :t])
        if self.repeat_factor > 1:
            k = k.repeat_interleave(self.repeat_factor, dim=1); v = v.repeat_interleave(self.repeat_factor, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        return self.o_proj(y.transpose(1, 2).contiguous().view(b, t, c))

class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
    def forward(self, x): return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__(); self.input_norm = RMSNorm(cfg.hidden_size); self.self_attn = CausalSelfAttention(cfg); self.post_attn_norm = RMSNorm(cfg.hidden_size); self.mlp = SwiGLU(cfg)
    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_norm(x), cos, sin)
        return x + self.mlp(self.post_attn_norm(x))

class TinyToolCallLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__(); self.config = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear): nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding): nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, input_ids, labels=None):
        b, t = input_ids.shape; device = input_ids.device
        x = self.tok_embeddings(input_ids)
        cos, sin = precompute_rope(t, self.config.hidden_size // self.config.num_heads, self.config.rope_base, device)
        for block in self.blocks: x = block(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss

@torch.no_grad()
def generate(model, sp, prompt: str, device: torch.device, max_new_tokens: int = 80):
    model.eval(); ids = sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids[:, -model.config.seq_len:])
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if int(next_id.item()) == sp.eos_id(): break
    model.train(); return sp.decode(input_ids[0].tolist())

def count_parameters(model): return sum(p.numel() for p in model.parameters())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True); parser.add_argument("--tokenizer", type=Path, required=True); parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=512); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000); parser.add_argument("--eval-interval", type=int, default=100); parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--hidden-size", type=int, default=256); parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4); parser.add_argument("--kv-heads", type=int, default=2); parser.add_argument("--intermediate-size", type=int, default=768); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer)); dataset = PackedShardDataset(args.data_dir, seq_len=args.seq_len)
    cfg = ModelConfig(vocab_size=sp.vocab_size(), seq_len=args.seq_len, hidden_size=args.hidden_size, num_layers=args.layers, num_heads=args.heads, num_kv_heads=args.kv_heads, intermediate_size=args.intermediate_size)
    model = TinyToolCallLM(cfg).to(device)
    print("Model config:", asdict(cfg)); print(f"Parameters: {count_parameters(model):,}"); print(f"Dataset tokens: {dataset.total_tokens:,}"); print("Device:", device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    log_path = args.out_dir / "train_log.jsonl"; start_time = time.time(); model.train(); opt.zero_grad(set_to_none=True)
    for step in range(1, args.max_steps + 1):
        step_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = dataset.get_batch(args.batch_size, device)
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
                _, loss = model(x, labels=y); loss = loss / args.grad_accum
            scaler.scale(loss).backward(); step_loss += float(loss.detach().cpu()) * args.grad_accum
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if step % args.eval_interval == 0 or step == 1:
            elapsed = time.time() - start_time; toks = step * args.batch_size * args.seq_len * args.grad_accum; tps = toks / max(elapsed, 1e-9)
            print(f"step={step:05d} loss={step_loss:.4f} tokens={toks:,} tok/s={tps:,.0f}")
            with log_path.open("a", encoding="utf-8") as f: f.write(json.dumps({"step": step, "loss": step_loss, "tokens_seen": toks, "tokens_per_sec": tps, "elapsed_sec": elapsed}) + "\n")
            try:
                print("\n--- sample ---"); print(generate(model, sp, "<|user|>\nFind unread emails from HR.\n<|tool_schema|>\n", device=device)[:1000]); print("--------------\n")
            except Exception as exc: print("Generation failed:", repr(exc))
        if step % args.save_interval == 0 or step == args.max_steps:
            ckpt_path = args.out_dir / f"ckpt_step_{step:06d}.pt"
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": asdict(cfg), "step": step}, ckpt_path)
            print("Saved checkpoint:", ckpt_path)
if __name__ == "__main__": main()
