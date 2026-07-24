# Scaling data generator (run once)

## What this does

This directory creates the `scaling_470m` bundle that the training folders
expect. Its 470M-token training split follows the specified ToolCall base mix:

| Category | Share | Approx. train tokens | Source/construction |
| --- | ---: | ---: | --- |
| Clean English/general | 45% | 211.5M | `HuggingFaceFW/fineweb-edu`, `sample-10BT` |
| Structured JSON/YAML/schema text | 20% | 94M | Raw and canonical views of APIs-guru OpenAPI specs |
| API/tool docs | 15% | 70.5M | Deterministic endpoint/reference docs derived from those specs |
| Code | 10% | 47M | `codeparrot/codeparrot-clean`, filtered to configured permissive licenses |
| Task/action text | 10% | 47M | Deterministic call/clarification/no-call examples derived from OpenAPI operations |

The builder clones `APIs-guru/openapi-directory` once into
`/content/toolcall_scaling_cache`, then records its exact commit SHA. Hugging
Face dataset revisions are likewise resolved and recorded. Salesforce xLAM is
not used here because it belongs to the later SFT mix and requires accepting
its gated-dataset terms.

The source revisions are resolved to immutable commit SHAs and recorded in the
output manifest. Documents are normalized, exact-deduplicated, assigned to
train/general-validation by a stable hash, encoded with the frozen native
SentencePiece model, terminated with EOS, and packed as little-endian
`uint16` shards. The builder makes:

```text
scaling_470m/
├── COMPLETE
├── bundle_manifest.json
├── build_config.json
├── tokenizer/toolcall_spm_32k.model
├── train/                       # exactly 470,000,000 tokens
├── validation_general/          # exactly   5,000,000 tokens
└── validation_structured/       # exactly   1,000,000 diagnostic tokens
```

Every split contains `manifest.json`, `manifest_shards.jsonl`, and `shards/`.
The structured diagnostic set is deterministic template data and is never used
for optimization.

## Where to run

Use a Google Colab CPU or High-RAM runtime. A GPU does not help tokenization.
The included notebook uses local `/content` storage and does not mount Google
Drive. Download its final `scaling_470m.tar.gz` export before the runtime ends,
because `/content` is temporary.

1. Copy your frozen tokenizer to:

   `scaling_data/artifacts/tokenizer/toolcall_spm_32k.model`

2. Install and check source access. The first check also downloads the
   APIs-guru repository into local Colab cache:

   ```bash
   cd /content/<repo>/experiments/scaling/scaling_data
   python -m pip install virtualenv==20.29.3
   python -m virtualenv --no-download /content/toolcall_scaling_data_env
   /content/toolcall_scaling_data_env/bin/python -m pip --version
   /content/toolcall_scaling_data_env/bin/python -m pip install --upgrade pip setuptools wheel
   /content/toolcall_scaling_data_env/bin/python -m pip install \
     --no-cache-dir --prefer-binary --retries 10 --timeout 60 \
     -r requirements.txt
   /content/toolcall_scaling_data_env/bin/python scripts/validate_data_runtime.py
   /content/toolcall_scaling_data_env/bin/python scripts/check_sources.py
   ```

   Optionally add an `HF_TOKEN` secret in Colab for higher public Hub rate
   limits. The notebook uses an isolated, exactly pinned data environment.
   Hugging Face/PyArrow streaming runs in source-worker subprocesses rather
   than in the shard-writer process, and buffered streaming shuffle is
   disabled. The pinned source revisions, weighted token scheduler,
   content-hash split, and training-time random-window sampler preserve a
   deterministic and well-mixed experiment.

   OpenAPI YAML is parsed with safe construction but without PyYAML's implicit
   timestamp conversion. Date-looking metadata remains text, so malformed
   public metadata such as an impossible calendar date cannot abort the build.

   Each Hugging Face worker also checkpoints the streaming iterator after
   every emitted document. Transient HTTP disconnects such as
   `IncompleteRead`, `ProtocolError`, and `ChunkedEncodingError` are retried
   with bounded exponential backoff, then resumed at the next document. A
   reconnect message is expected and is not a failed build; the build fails
   only after the retry limit is exhausted or for a non-network exception.

   Colab's standard-library `venv` can leave a `bin/python` without bootstrapping
   pip. The notebook therefore uses `virtualenv`, verifies pip explicitly, and
   treats environment creation as atomic. A failed or outdated environment has
   no `READY` marker and is recreated on the next run. Pip output is
   intentionally visible so network, disk, or resolver failures are not hidden
   behind a wrapper traceback.

3. Run the included notebook, or run manually in local Colab storage:

   ```bash
   /content/toolcall_scaling_data_env/bin/python scripts/build_scaling_data.py \
     --config configs/scaling_470m.json \
     --tokenizer artifacts/tokenizer/toolcall_spm_32k.model \
     --output /content/toolcall_scaling_data/scaling_470m
   ```

4. Verify byte sizes and manifests:

   ```bash
   /content/toolcall_scaling_data_env/bin/python scripts/verify_scaling_data.py \
     /content/toolcall_scaling_data/scaling_470m
   ```

   Add `--checksums` for the slower full SHA-256 verification.

5. Decode samples:

   ```bash
   python scripts/inspect_scaling_data.py \
     /content/toolcall_scaling_data/scaling_470m \
     --split validation_general
   ```

## Safe smoke test

Before spending time on 470M tokens, test the full network/tokenizer/writer path:

```bash
python scripts/build_scaling_data.py \
  --tokenizer artifacts/tokenizer/toolcall_spm_32k.model \
  --output /content/toolcall_scaling_data/_smoke_1m \
  --limit-train-tokens 1000000
```

The full builder refuses to touch an existing output directory. `--overwrite`
deletes and rebuilds only the exact directory passed via `--output`; use it only
for an intentionally discarded incomplete build.

The Colab notebook removes only the exact known `_smoke_1m` folder on a smoke
rerun. It removes a full output only when that exact folder contains the
generator's `BUILDING` marker, and never replaces an unrecognized directory or
a completed full bundle.

Primary sources: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
[CodeParrot Clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean),
and [APIs-guru OpenAPI Directory](https://github.com/APIs-guru/openapi-directory).
