# ToolCall extended scaling runs

This folder is a sibling of `scaling_runs/` and contains two fixed-purpose,
full-data experiments:

| Notebook | Run | Architecture | Parameters |
|---|---|---|---:|
| `m30_470m_colab.ipynb` | `m30_standard_470m_seed42` | Original M30, GELU and full multi-head attention | 29,990,784 |
| `m60_swiglu_gqa_470m_colab.ipynb` | `m60_swiglu_gqa_470m_seed42` | M60, SwiGLU and 2:1 grouped-query attention | 60,439,040 |

Both models start from random weights. Neither run resumes from a shorter
scaling experiment because the cosine learning-rate schedule is defined over
the complete long-token run.

## Required parent layout

```text
scaling/
├── scaling_runs/
│   ├── scaling/
│   ├── scripts/
│   └── requirements.txt
└── extended_runs/
    ├── configs/
    ├── data/
    ├── runs/
    ├── m30_470m_colab.ipynb
    └── m60_swiglu_gqa_470m_colab.ipynb
```

The complete Colab ZIP already has this layout. The folder-only ZIP contains
exactly one top-level `extended_runs/` directory. Extract that directory inside
your existing `experiments/scaling/` directory, next to `scaling_runs/`.

## Data placement

The code ZIP does not embed the large generated dataset. Upload your existing
`scaling_470m.zip`, `scaling_470m.tar.gz`, or `scaling_470m.tgz` when prompted
by either notebook. It is automatically extracted to:

```text
extended_runs/data/scaling_470m/
├── COMPLETE
├── bundle_manifest.json
├── tokenizer/
├── train/
├── validation_general/
└── validation_structured/
```

The tokenizer model is already part of a valid generated `scaling_470m` bundle.
Training reads the packed token shards directly, so there is no separate
tokenizer upload step.

## Exact token target

The bundle contains 470,000,000 training source tokens. Each packed sequence
uses 513 source tokens to produce 512 next-token predictions. Both configs
therefore request 469,000,000 prediction tokens.

After rounding to complete optimizer steps, each run produces 469,008,384
predictions, consumes 469,924,416 source tokens, and leaves 75,584 source
tokens unused. The run names use `470m` to identify the complete source-data
bundle.

## Architecture details

### M30 standard

- 10 Transformer layers
- Hidden size 384
- 6 query heads and 6 KV heads
- GELU MLP with intermediate size 1,536
- RoPE and RMSNorm
- 29,990,784 parameters

This is architecturally identical to the earlier M30 family and isolates the
effect of longer training.

### M60 SwiGLU + GQA

- 14 Transformer layers
- Hidden size 512
- 8 query heads and 4 KV heads
- 2:1 grouped-query attention
- SwiGLU with intermediate size 1,536
- RoPE and RMSNorm
- 60,439,040 parameters

The GQA projection savings offset the additional SwiGLU gate projection, so
this modern variant has exactly the same parameter count as the standard M60
baseline.

## Colab workflow

Each notebook is fixed to one run and never mounts Google Drive.

For the first session:

1. Upload `ToolCall-Extended-Runs-Colab-Complete.zip`.
2. Upload the `scaling_470m` archive.
3. Leave `UPLOAD_RESUME_BUNDLE = False`.
4. Run all cells.
5. Training pauses after 165 minutes unless it finishes first.
6. Run the export cell and download the generated resume ZIP.

For every later session:

1. Open a fresh T4 Colab runtime and run the same notebook.
2. Upload the same complete project ZIP and the same data archive.
3. Set `UPLOAD_RESUME_BUNDLE = True`.
4. Upload the latest matching resume ZIP.
5. Continue through the preflight, training, and export cells.

The resume ZIP includes the latest model weights, optimizer state, AMP scaler,
global step, tokens seen, data cursor, validation state, RNG states, metrics,
summary, and TensorBoard events. The data and project archives are deliberately
not duplicated inside every resume ZIP.

## Manual commands

Run these from the parent `scaling/` directory.

Validate both configs without importing PyTorch:

```bash
python -m extended_runs.validate
```

Preflight M30 against the extracted full dataset:

```bash
python -m extended_runs.train \
  --run m30 \
  --device cuda \
  --resume auto \
  --preflight-only
```

Train M30 for at most 165 minutes in the current process:

```bash
python -m extended_runs.train \
  --run m30 \
  --device cuda \
  --resume auto \
  --max-session-minutes 165
```

Train the M60 SwiGLU + GQA variant:

```bash
python -m extended_runs.train \
  --run m60 \
  --device cuda \
  --resume auto \
  --max-session-minutes 165
```

Create a resume bundle:

```bash
python -m extended_runs.resume_bundle pack \
  --run-name m60_swiglu_gqa_470m_seed42
```

Restore it:

```bash
python -m extended_runs.resume_bundle restore \
  /content/m60_swiglu_gqa_470m_seed42_resume.zip \
  --expected-run m60_swiglu_gqa_470m_seed42
```

## Important rules

- Start both long schedules from random initialization.
- Resume only the exact same run name and configuration.
- Use the same `scaling_470m` bundle in every session.
- Do not change the token target or learning-rate schedule after training starts.
- Do not use the M30-200M checkpoint for M30-470M.
- Do not use a standard M60 checkpoint for the SwiGLU + GQA run.
- Keep the newest resume ZIP until `summary.json` reports `completed`.

