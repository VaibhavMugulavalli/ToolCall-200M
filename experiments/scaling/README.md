# ToolCall scaling pilot — complete Colab workspace

This folder is the complete, separated workspace for the nine-point scaling
pilot. The missing `scaling_470m` dataset is **generated once before training**;
training never downloads or tokenizes raw text.

```text
experiments/scaling/
├── README.md
├── DATA_SPLIT.md    # base/CPT/SFT/preference stage boundaries
├── scaling_data/     # CPU/High-RAM Colab: build the reusable data bundle
└── scaling_runs/     # T4 Colab: train M13, M30, M60 and inspect dashboards
```

## Correct order

1. Run `scaling_data/notebooks/generate_scaling_data_colab.ipynb` in a CPU or
   High-RAM Colab runtime. Supply the already-frozen
   `toolcall_spm_32k.model`. The included notebook writes locally under
   `/content` and exports a tarball; it does not require Google Drive.
2. Run `scaling_runs/m13/m13_colab.ipynb` with a T4. It copies the bundle from
   Drive to fast `/content` storage, verifies it, performs a short
   qualification, then exposes separate cells for the 120M, 230M, and 460M
   runs.
3. Repeat with the M30 and M60 notebooks. Each scientific point starts from a
   fresh seeded initialization. `--resume auto` only resumes an interrupted
   run with the same name.
4. Use TensorBoard inside each Colab notebook for live loss, validation loss,
   learning rate, gradient norm, throughput, step time, GPU memory, and token
   progress. The bundled static dashboard can later be exported to GitHub Pages
   or Vercel.

The generated train bundle is exactly 470M `uint16` tokens (~940 MB), plus 5M
general-validation and 1M structured-diagnostic tokens. Keep about 2 GB free in
Google Drive and about 1.2 GB free in the active Colab runtime.

## Environment checkpoints

| Stage | Run where | Expected output | Stop/checkpoint condition |
| --- | --- | --- | --- |
| Source check | CPU Colab | One `PASS` per configured source | All five source streams return non-empty text |
| Data build | CPU/High-RAM Colab | `scaling_470m/COMPLETE` and manifests | All three exact token counts verify |
| Training qualification | T4 Colab | 1M-token debug checkpoint and metrics | Forward/backward/checkpoint succeeds, no OOM |
| Scientific point | T4 Colab | checkpoint, JSONL metrics, TensorBoard events, summary | final general validation loss exists |

The token-balanced mixture in `scaling_data/configs/scaling_470m.json` follows
the ToolCall base-pretraining split: 45% FineWeb-Edu, 20% raw structured
OpenAPI JSON/YAML/schema views, 15% derived API/tool documentation, 10%
permissive-license CodeParrot Clean, and 10% synthetic task/action text. The
resolved Hugging Face and APIs-guru commit SHAs are recorded in
`bundle_manifest.json`. Salesforce xLAM is intentionally reserved for the SFT
stage and is not mixed into this base scaling corpus. The additional 1M
structured validation set is diagnostic only and excluded from optimization
and the scaling-law fit.
