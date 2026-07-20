# ToolCall-200M Next Pipeline

This folder contains the next practical stage after tokenizer training.

You now have a native SentencePiece tokenizer:

```text
toolcall_spm_32k.model
toolcall_spm_32k.vocab
```

The next milestone is to build packed token shards and smoke-test training before running the real 200M pretraining job.

---

## 0. Install requirements

```bash
pip install -r requirements.txt
```

In Colab/Kaggle:

```python
!pip -q install -r requirements.txt
```

---

## 1. Put the tokenizer in your project

Expected structure:

```text
toolcall-200m/
  tokenizer/
    toolcall_spm_32k.model
    toolcall_spm_32k.vocab
  scripts/
    prepare_pretraining_shards.py
    inspect_token_shards.py
    train_tiny_smoke.py
```

---

## 2. Build a 1M-token micro shard first

Run this before creating 50M or 1B tokens.

```bash
python scripts/prepare_pretraining_shards.py \
  --tokenizer tokenizer/toolcall_spm_32k.model \
  --output-dir data/tokenized/micro_1m \
  --target-tokens 1000000 \
  --shard-tokens 1000000 \
  --include-xlam false
```

Use `--include-xlam true` only if you have accepted the gated xLAM dataset and are authenticated with Hugging Face.

---

## 3. Inspect the micro shard

```bash
python scripts/inspect_token_shards.py \
  --data-dir data/tokenized/micro_1m \
  --tokenizer tokenizer/toolcall_spm_32k.model \
  --num-samples 5
```

Check that decoded samples look like normal text/JSON/code/API docs.

---

## 4. Build a 50M-token smoke shard

```bash
python scripts/prepare_pretraining_shards.py \
  --tokenizer tokenizer/toolcall_spm_32k.model \
  --output-dir data/tokenized/smoke_50m \
  --target-tokens 50000000 \
  --shard-tokens 10000000 \
  --include-xlam false
```

This creates five 10M-token shards by default. Since tokens are `uint16`, each 10M-token shard is ~20MB.

---

## 5. Train a tiny smoke model

This does not produce the final model. It only validates:

- token shards
- dataloader
- model forward/backward
- causal LM loss
- checkpointing
- basic generation

```bash
python scripts/train_tiny_smoke.py \
  --data-dir data/tokenized/smoke_50m \
  --tokenizer tokenizer/toolcall_spm_32k.model \
  --out-dir runs/tiny_smoke \
  --seq-len 512 \
  --batch-size 8 \
  --grad-accum 4 \
  --max-steps 1000
```

Expected behavior:

- loss should decrease;
- checkpoints should save;
- generation will be bad early, but it should not crash;
- tokens/sec should be recorded.

---

## 6. Only after this: build the real Stage 1 pilot

```bash
python scripts/prepare_pretraining_shards.py \
  --tokenizer tokenizer/toolcall_spm_32k.model \
  --output-dir data/tokenized/stage1_pilot_1b \
  --target-tokens 1000000000 \
  --shard-tokens 25000000 \
  --include-xlam true
```

Then inspect:

```bash
python scripts/inspect_token_shards.py \
  --data-dir data/tokenized/stage1_pilot_1b \
  --tokenizer tokenizer/toolcall_spm_32k.model \
  --num-samples 10
```

Do not build 1B tokens before the 1M and 50M stages pass.

---

## Dataset mixture

Default mixture in `prepare_pretraining_shards.py`:

| Source | Share |
|---|---:|
| FineWeb-Edu | 45% |
| CodeParrot Clean | 10% |
| APIs.guru OpenAPI | 20% |
| xLAM Function Calling 60k | 10% |
| Synthetic ToolCall | 15% |

If xLAM is unavailable, run with `--include-xlam false`, which transfers its budget to synthetic ToolCall text.

---

## Output format

Each shard is a flat binary file:

```text
uint16 token_id, uint16 token_id, ...
```

Example structure:

```text
data/tokenized/smoke_50m/
  manifest.json
  manifest_shards.jsonl
  shards/
    shard_00000.bin
    shard_00001.bin
    ...
```

Because the vocabulary is 32k, token IDs fit safely in `uint16`.

---

## Important rule

Do not use a Hugging Face tokenizer wrapper yet.

Use native SentencePiece:

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="tokenizer/toolcall_spm_32k.model")
ids = sp.encode(text, out_type=int)
```

The model pretraining dataloader should read packed token IDs, not raw text.
