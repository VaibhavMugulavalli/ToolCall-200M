# Pilot Runs — Colab Direct Version

This scaffold is designed to run in Google Colab without mounting Google Drive.

The runtime layout is:

```text
/content/experiments/
  tokenizer/
    artifacts/tokenizer/toolcall_spm_32k.model
    data/tokenized/smoke_50m/
  pilot_runs/
    configs/
    scripts/
    runs/
    results/
```

## Inputs you must upload in Colab

1. This scaffold zip:
   - `pilot_runs_scaffold_colab_direct.zip`

2. Tokenizer artifact:
   - either `toolcall_spm_32k.model` directly
   - or a zip containing `toolcall_spm_32k.model`

3. 50M smoke shard artifact:
   - a zip containing:
     - `manifest.json`
     - `manifest_shards.jsonl`
     - `shards/shard_*.bin`

No Google Drive mount is required.

## Recommended Colab flow

Open:

```text
notebooks/pilot_runs_colab_direct.ipynb
```

Then run cells in order.

## What this validates

This is not the final ToolCall-200M training run. It validates:

- input shard visibility;
- packed-token dataloader;
- causal LM loss;
- backward pass;
- mixed precision;
- checkpoint writing;
- rough tokens/sec;
- generation does not crash.

## Artifact policy

Because Colab storage is ephemeral, download only summaries or checkpoints you explicitly need.

Do not commit `runs/` or checkpoint files to Git.
