# ToolCall-200M final 4B corpus — CPU Colab workflow

This package starts the final pretraining-corpus build for ToolCall-200M. The
first notebook performs the finite-source capacity audit that must precede the
8 GB shard build.

## Locked numerical target

The final model is planned to train with sequence length 512 and a global
dual-GPU optimizer batch of 32,768 prediction tokens.

| Quantity | Exact value |
| --- | ---: |
| Requested next-token predictions | 4,000,000,000 |
| Rounded optimizer steps | 122,071 |
| Actual next-token predictions | 4,000,022,528 |
| Required stored training tokens | 4,007,835,072 |
| General validation tokens | 5,000,000 |
| Structured validation tokens | 2,000,000 |

Every 512-token prediction sequence consumes 513 stored source tokens because
of the one-token causal shift. The physical corpus therefore has to be slightly
larger than the prediction budget.

## Initial mixture caps

| Category | Cap | Nominal prediction tokens |
| --- | ---: | ---: |
| FineWeb-Edu general/technical text | 60% | 2.4B |
| Permissively licensed code | 15% | 600M |
| Raw/canonical OpenAPI structured text | 10% | 400M |
| API documentation and schemas | 10% | 400M |
| Synthetic tool actions | 5% | 200M |

The three finite OpenAPI-derived values are caps, not targets that may be filled
with duplicated documents. Stage 1 exhaustively measures their actual unique
capacity. If a finite source falls short, the provisional policy assigns 80%
of the shortfall to FineWeb-Edu and 20% to permissively licensed code. That
redistribution still requires review before Stage 2.

## Why Stage 1 is mandatory

The earlier 470M scaling corpus proved the tokenizer and source pipeline, but
its 45/20/15/10/10 mixture was intentionally structured-heavy. Scaling those
percentages directly to 4B would request 1.8B OpenAPI-derived tokens and risk
template repetition. This audit measures instead of guessing.

For each finite source it reports:

- documents emitted and accepted;
- retained unique tokens under the frozen tokenizer;
- exact duplicates and truncated documents;
- a deterministic sampled SimHash near-duplicate estimate;
- configured cap and measured shortfall;
- source commit, tokenizer digest, config digest, and license note.

Near-duplicate matching is diagnostic during Stage 1. It does not delete data.
The final deduplication policy is frozen only after reviewing the report.

## Run in CPU Colab

1. Download this package and separately open
   `notebooks/01_audit_toolcall_4b_sources_cpu_colab.ipynb` in Colab.
2. Select **Runtime → Change runtime type → CPU**.
3. Run the notebook and upload the package ZIP when prompted.
4. Mount Google Drive when prompted. CPU performs all processing; Drive only
   stores the frozen tokenizer and completed audit reports.
5. Upload `toolcall_spm_32k.model` on the first session.
6. Let all three finite-source audits complete.
7. Download:

   - `toolcall_4b_capacity_summary.json`
   - `toolcall_4b_capacity_summary.csv`

8. Stop and review those files before generating training shards.

Completed source reports are saved under:

```text
MyDrive/ToolCall-200M/toolcall_4b_corpus/capacity_audit/
```

If Colab disconnects, rerun the notebook. A completed report is skipped only
after its tokenizer hash, configuration hash, and pinned OpenAPI commit match.

## Why persistent storage is used

A 4.008B-token `uint16` training corpus is about 8.02 GB before manifests and
exports. A free CPU runtime is ephemeral and may end before a complete build.
Keeping durable stage checkpoints is therefore necessary even though the
computation itself is entirely CPU Colab.

Large Hugging Face caches and the APIs-guru clone remain under `/content` to
avoid slow random I/O on Drive. Stage 2 will write completed local shard files
to persistent storage only after checksum validation.

## Included files

```text
configs/toolcall_4b.json
notebooks/01_audit_toolcall_4b_sources_cpu_colab.ipynb
scripts/audit_source_capacity.py
scripts/merge_capacity_reports.py
scripts/check_sources.py
scripts/build_scaling_data.py
scripts/validate_data_runtime.py
requirements.txt
```

`build_scaling_data.py` is retained because the audit reuses its tested OpenAPI
parsers, normalization, deterministic generators, tokenizer rules, and pinned
source handling. Do not use the legacy 470M build command for the final corpus.

## Stage 1 stop condition

Stage 1 is complete only when all of the following exist and the summary status
is `awaiting_human_approval`:

```text
structured_openapi.capacity.json
api_tool_docs.capacity.json
synthetic_tool_actions.capacity.json
toolcall_4b_capacity_summary.json
toolcall_4b_capacity_summary.csv
```

Stage 2 will use the reviewed report to freeze exact source allocations, build
restart-safe 10M-token source shards, deterministically interleave them, and
produce the final verified Kaggle-ready corpus.
