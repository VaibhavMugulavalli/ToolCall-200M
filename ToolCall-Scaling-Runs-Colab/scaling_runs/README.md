# Scaling runs — local-only Google Colab T4

This directory is self-contained training code for all M13, M30, and M60
scaling points. It does not depend on the tokenizer-preprocessing repository.
Its only external input is the one `scaling_470m` bundle produced by the
sibling `scaling_data` directory.

No notebook mounts or reads Google Drive. Project files, the data bundle,
TensorBoard logs, metrics, and checkpoints all live under Colab's `/content`.
Because that storage is temporary, each scientific-run cell creates and
downloads a ZIP containing the full run and its checkpoints.

## Structure

```text
scaling_runs/
├── requirements.txt
├── m13/
│   ├── m13_colab.ipynb
│   └── configs/{debug_1m,low_120m,medium_230m,high_460m}.json
├── m30/
│   ├── m30_colab.ipynb
│   └── configs/{debug_1m,low_50m,medium_100m,high_200m}.json
├── m60/
│   ├── m60_colab.ipynb
│   └── configs/{debug_1m,low_25m,medium_50m,high_100m}.json
├── scaling/             # model, dataloader, trainer, resume, metrics
├── scripts/             # checks, train, dashboards, archive, analysis
├── dashboard/           # static browser dashboard
├── data/
│   └── scaling_470m/    # upload/extract the generated data bundle here
├── runs/                # local checkpoints, metrics, logs, and summaries
└── results/
```

The required tokenizer path is:

```text
scaling_runs/data/scaling_470m/tokenizer/toolcall_spm_32k.model
```

The data generator already includes this tokenizer in `scaling_470m`. If your
bundle does not contain it, place the frozen model at `/content/tokenizer.model`
or `/content/toolcall_spm_32k.model`; the notebooks copy and rename it to the
required path.

## Experiment matrix

| Family | Exact parameters | Low | Medium | High | Colab batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| M13 | 12,913,920 | 120M | 230M | 460M | micro 8 × accumulation 4 |
| M30 | 29,990,784 | 50M | 100M | 200M | micro 8 × accumulation 4 |
| M60 | 60,439,040 | 25M | 50M | 100M | micro 4 × accumulation 8 |

Every point uses sequence length 512 and exactly 16,384 prediction tokens per
optimizer step. Each tier is approximately iso-compute in `N × D`. All nine
points use the same frozen token order and validation sets.

## Recommended Colab workflow

Open exactly one family notebook in a T4 runtime. The notebook contains the
complete sequence:

1. Upload `ToolCall-Scaling-Runs-Colab.zip` in the first notebook cell.
2. Using Colab's Files pane, upload `scaling_470m.tar.gz`,
   `scaling_470m.tgz`, or `scaling_470m.zip` to `/content`. An already-extracted
   `/content/scaling_470m` directory is also accepted. The setup cell installs
   it at `scaling_runs/data/scaling_470m`.
3. Install dependencies and verify CUDA, models, matrix, and data.
4. Run a 1M-token synthetic qualification.
5. Start TensorBoard.
6. Run low, medium, and high in separate cells. Each cell downloads a full
   run ZIP after completion.

To resume in a new runtime, upload the matching downloaded run ZIP (for example
`m30_medium_100m_seed42.zip`) into `/content` before executing the data/setup
cell. It restores the ZIP under `scaling_runs/runs`, after which `--resume auto`
continues that exact run. Do not resume a different budget from another
budget's checkpoint.

Manual example from `scaling_runs/`:

```bash
python scripts/train.py \
  --config m13/configs/low_120m.json \
  --runs-dir runs \
  --device cuda \
  --resume auto
```

Before starting:

```bash
python scripts/verify_data_bundle.py data/scaling_470m
python scripts/check_experiment_matrix.py
python scripts/check_model.py --config m13/configs/low_120m.json --forward-check
python scripts/train.py \
  --config m13/configs/low_120m.json \
  --runs-dir runs \
  --device cuda --resume auto --preflight-only
```

## Dashboard

TensorBoard is the live Colab dashboard. The trainer writes train/EMA loss,
general and structured validation loss/perplexity, learning rate, gradient
norm, throughput, step time, CUDA memory, and tokens seen.

In a notebook:

```python
%load_ext tensorboard
%tensorboard --logdir /content/toolcall_scaling_workspace/scaling_runs/runs
```

The custom dashboard reads the same runs' JSONL logs. Run locally after
downloading or while Colab is active:

```bash
python scripts/serve_dashboard.py \
  --runs-dir runs \
  --host 0.0.0.0 --port 8000
```

For GitHub Pages/Vercel, restore all nine run directories, export static JSON,
and publish `dashboard/` as the site root:

```bash
python scripts/export_dashboard.py --runs-dir runs --output-dir dashboard/data
```

For GitHub Pages, copy `deploy/github_pages_workflow.yml` to
`.github/workflows/scaling-dashboard.yml` in the repository before pushing.

## Outputs and stop conditions

Each run creates `config.json`, `metrics.jsonl`, `summary.json`, TensorBoard
events, and `checkpoints/`. A scientific point is complete only when:

- `summary.json` says `"status": "completed"`;
- a `checkpoint_final.pt` exists;
- final general validation loss was recorded;
- no non-finite loss or CUDA OOM occurred.

After all nine points:

```bash
python scripts/analyze_scaling.py \
  --runs-dir runs \
  --output-dir results
```

This is a Chinchilla-style experimental design; these small pilot runs do not
claim to reproduce the original Chinchilla coefficients.
