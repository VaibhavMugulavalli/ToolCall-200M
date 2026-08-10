# ToolCall-200M final pretraining

This package runs the frozen **24-layer, untied-head, 200,184,576-parameter**
ToolCall model over the complete 4B-token Kaggle corpus on exactly two T4 GPUs.
It is intended for repeated scheduled Kaggle sessions, not one uninterrupted job.

## Frozen run

| Setting | Value |
|---|---:|
| Layers / width | 24 / 768 |
| Query / KV heads | 12 / 4 |
| SwiGLU intermediate size | 2,048 |
| Vocabulary | 32,000 |
| Input/output embeddings | Untied |
| Parameters | 200,184,576 |
| Context | 2,048 for the entire corpus |
| Precision | FP16 with dynamic loss scaling |
| Parallelism | DeepSpeed ZeRO-2, no CPU offload |
| Micro-batch | 2 sequences per GPU |
| Accumulation | 32 micro-batches |
| Global prediction tokens/step | 262,144 |
| Peak / minimum LR | 3e-4 / 3e-5 |
| Schedule | 2% warmup, then cosine decay |
| Activation checkpointing | Enabled |

The actual train split contains 4,007,835,072 source tokens. Causal packing needs
one extra source token per sequence. The no-repeat plan therefore performs 15,281
complete optimizer steps, predicts **4,005,822,464 tokens**, consumes
4,007,778,432 source tokens, and leaves only 56,640 source tokens that cannot form
another global step. This uses the complete 4B prediction-token budget without
wrapping or silently reusing data.

Training all tokens at context 2,048 is the most representative choice for this
model's schema-reading objective and matches `docs/model_strategy.md`. The GPU
qualification run is mandatory because the earlier scaling runs used context 512.
The trainer does not silently fall back to a different context or batch size.

## Why ZeRO-2

ZeRO-2 partitions optimizer state and gradients between the two T4s while keeping
the model replicated. This is a useful memory reduction for AdamW without the
communication and checkpoint complexity of ZeRO-3. CPU/NVMe offload is disabled;
it would trade scarce GPU time for host transfers. Per-block activation
checkpointing controls the additional activation memory from context 2,048.

Every rank reads a different slice of one deterministic global token stream. The
checkpoint stores the global data cursor, optimizer/scaler state, per-rank random
state, exact model/config fingerprints, and the shard-name/size fingerprints.
Resume fails closed if any of these change.

## One-time Hugging Face and W&B setup

1. Sign in to Hugging Face and create a new **model** repository. A private repo is
   recommended while training. A clear name is
   `YOUR_USERNAME/ToolCall-200M-checkpoints`.
2. Create a fine-grained Hugging Face token with write access only to that model
   repo. Do not paste the token into the notebook.
3. In the Kaggle notebook, open **Add-ons -> Secrets** and add:
   - `HF_TOKEN`: the fine-grained Hugging Face token.
   - `WANDB_API_KEY`: your Weights & Biases API key.
4. Create or sign in to W&B. The trainer creates the
   `toolcall-200m-pretraining` project automatically. If you log under a W&B team,
   set `WANDB_ENTITY` in the notebook configuration cell.
5. Set `HF_REPO_ID` in the notebook to the exact repo ID from step 1.

The Hugging Face repo is the durable restart source. W&B is the live dashboard.
TensorBoard events and JSONL metrics are also uploaded with each Hub snapshot, so
the graphs are not locked to W&B.

Use a repository dedicated to rolling checkpoints. Replacing `resume/` removes
old files from the current revision, but large LFS objects can remain in Git
history and count toward storage. Hugging Face provides `super_squash_history()`
for periodic compaction, but it permanently deletes rollback history. The trainer
does not call it automatically. Decide on that policy only after the first real
checkpoint establishes its actual size and upload time.

## Kaggle workflow

Open `notebooks/toolcall_200m_kaggle.ipynb` in Kaggle, attach the
`vaibhavmugulavalli/toolcall-4b-pretraining-corpus-v1` dataset, enable Internet,
and select the **GPU T4 x2** accelerator.

Run it interactively once with:

```python
RUN_MODE = "qualify"
```

Qualification checks the exact split sizes and tokenizer, instantiates the exact
200M model, runs two full distributed optimizer steps, reports throughput and peak
VRAM, saves a ZeRO-2 checkpoint, and reloads it. It does not upload the disposable
weights. Do not schedule the 4B run until it prints `QUALIFICATION PASSED`.

Then change the same cell to:

```python
RUN_MODE = "train"
```

Save a Kaggle notebook version and schedule it. Each invocation:

1. downloads `resume/latest` from the configured Hugging Face repo, if present;
2. restores model, optimizer, loss scaler, cursor, counters, and W&B run ID;
3. trains until either completion or 675 minutes of trainer wall time;
4. saves and uploads a clean checkpoint before the 12-hour Kaggle cutoff; and
5. exits successfully so the next scheduled invocation can resume it.

Local checkpoints are written hourly. A durable Hub snapshot is uploaded every
five hours and again at the session boundary. The Hub update is one commit that
atomically replaces `resume/` and `run/`; an interrupted upload therefore does not
partially replace the previous usable revision.

## Watching the graphs

The training cell prints the direct W&B run URL. Its live charts include:

- training loss and exponential moving average;
- general and structured validation loss/perplexity;
- learning rate and FP16 overflows;
- global tokens/second and estimated hours remaining;
- allocated, reserved, and peak-reserved VRAM; and
- prediction tokens seen.

For a local fallback after a session, point TensorBoard at the downloaded
`run/tensorboard` directory from the Hugging Face repo:

```bash
tensorboard --logdir run/tensorboard
```

## Expected duration

The 4B-token run is not a three-session job. Actual duration must be taken from the
qualification throughput, but these bounds are useful for scheduling:

| Measured global throughput | Training hours | 11.25-hour sessions |
|---:|---:|---:|
| 2,000 tok/s | 556 h | about 50 |
| 5,000 tok/s | 223 h | about 20 |
| 8,000 tok/s | 139 h | about 13 |
| 10,000 tok/s | 111 h | about 10 |

Checkpoint upload, validation, package installation, and Kaggle queue/quota time
are additional. The W&B ETA uses the measured end-to-end optimizer-step
throughput and becomes more useful after the first few logged steps.

## Manual commands

Read-only data and token-plan validation:

```bash
python -m training.final_pretraining.preflight \
  --data-root /kaggle/input/toolcall-4b-pretraining-corpus-v1
```

Disposable two-step distributed qualification:

```bash
deepspeed --num_gpus 2 -m training.final_pretraining.train \
  --data-root /kaggle/input/toolcall-4b-pretraining-corpus-v1 \
  --run-dir /kaggle/working/toolcall-200m-qualification \
  --qualification-steps 2
```

Real train/resume session:

```bash
export HF_REPO_ID=YOUR_USERNAME/ToolCall-200M-checkpoints
deepspeed --num_gpus 2 -m training.final_pretraining.train \
  --data-root /kaggle/input/toolcall-4b-pretraining-corpus-v1 \
  --run-dir /kaggle/working/toolcall-200m-run \
  --resume auto
```

`--resume auto` starts from random initialization only when `resume/latest` is
absent. Use `--resume required` after the first successful session if you want a
missing Hub checkpoint to be a hard error.

## Loading and exporting checkpoints

Training resume is automatic; no manual file selection is needed. DeepSpeed
checkpoints remain sharded because that is the reliable format for restoring the
optimizer on two GPUs.

To create a normal, consolidated safetensors model after a milestone or the final
step:

```bash
python -m training.final_pretraining.export_latest \
  --hub-repo-id YOUR_USERNAME/ToolCall-200M-checkpoints \
  --output-dir /kaggle/working/toolcall-200m-export \
  --upload
```

Load that export without DeepSpeed:

```python
from training.final_pretraining.model import load_exported_model

model = load_exported_model("/kaggle/working/toolcall-200m-export", "cuda")
```

Do not change the architecture, batch geometry, sequence length, shard set, or
optimization config after the first checkpoint. Those values are deliberately
fingerprinted so an incompatible continuation cannot corrupt the run.
