# Tokenizer and Pretraining Data Pipeline

## Purpose
This directory stores the frozen tokenizer and data preprocessing pipeline for ToolCall-200M.

## Frozen tokenizer
The SentencePiece tokenizer files define the model vocabulary contract.
Once pretraining starts, these files should not be changed unless all tokenized shards are regenerated.

## Why 1M micro shard was created
To validate tokenizer loading, native SentencePiece encoding, EOS insertion, uint16 packing, and decode inspection.

## Why 50M smoke shard was created
To validate the preprocessing pipeline at non-toy scale before GPU-heavy model training.
It checks multi-source streaming, multi-shard rollover, manifest generation, random decode inspection, and dataloader readiness.

## Artifact policy
Binary .bin token shards are not committed to GitHub.
Only scripts, configs, tokenizer files, notebooks, and manifests are tracked.

## Recommended next step
Run tiny smoke training on Colab T4 using the 50M shard.
