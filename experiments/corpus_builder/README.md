# ToolCall-200M final 4B corpus — CPU Colab workflow

This package contains both stages of the final pretraining-corpus build for
ToolCall-200M. Stage 1 audits finite-source capacity. Phase 2 uses the approved
audit result to produce the approximately 8.03 GB training corpus entirely in
CPU Colab sessions.

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

## Approved Phase 2 allocation

| Source | Exact stored tokens | Final fraction |
| --- | ---: | ---: |
| FineWeb-Edu general/technical text | 2,632,512,585 | 65.6842% |
| Permissively licensed CodeParrot Clean | 658,128,146 | 16.4210% |
| Raw/canonical OpenAPI structured text | 400,783,507 | 10.0000% |
| API documentation and schemas | 197,553,288 | 4.9292% |
| Synthetic tool actions | 118,857,546 | 2.9656% |
| **Total** | **4,007,835,072** | **100%** |

Stage 1 measured a 284,764,427-token finite-source shortfall. The approved
policy assigned 80% of that shortfall to FineWeb-Edu and 20% to permissively
licensed code. `configs/toolcall_4b_approved.json` freezes the exact result and
binds it to the audit-summary, tokenizer, and source-revision hashes.

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

## Stage 1: capacity audit

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
avoid slow random I/O on Drive.

## Phase 2: build the corpus

1. Open `notebooks/02_build_toolcall_4b_corpus_cpu_colab.ipynb` in Colab.
2. Select **Runtime → Change runtime type → CPU**.
3. Run all cells and upload the latest project ZIP.
4. Mount the same Google Drive used by Stage 1.
5. If asked, upload the frozen tokenizer and the Stage 1 capacity summary.
6. Leave the build cell running. It processes as many stages as fit within the
   configured session budget and pauses at a durable checkpoint.
7. After a disconnect or clean pause, start a new CPU runtime and rerun all
   cells. Completed stages are verified and skipped.

Persistent Phase 2 state is stored under:

```text
MyDrive/ToolCall-200M/toolcall_4b_corpus/phase2/
├── staging/                         # source shards, cursors, hashes, progress
├── finalization_plan.json           # immutable deterministic interleave plan
└── final/toolcall_4b/               # Kaggle-ready corpus when COMPLETE exists
```

The finalizer moves staged shards rather than making a second 8 GB copy. Do not
manually move or delete files under `staging/`; the resume transaction and
finalization plan are authoritative.

### Recovery guarantees

- Documents are normalized and exact-deduplicated per source using the first
  128 bits of SHA-256, matching the Stage 1 policy.
- A Hugging Face row and its streaming cursor are committed together through a
  parent/worker acknowledgement handshake.
- Shard data, dedup hashes, and the cursor are copied first; `progress.json` is
  written last. After a disconnect, unreferenced ahead files are ignored.
- General validation is a deterministic 1% hash-selected FineWeb-Edu holdout
  and those documents are excluded from the FineWeb training quota.
- `COMPLETE` is written only after exact token counts, file sizes, tokenizer
  digest, and every shard checksum pass.

## Included files

```text
configs/toolcall_4b.json
configs/toolcall_4b_approved.json
artifacts/approval/toolcall_4b_capacity_summary.json
artifacts/approval/toolcall_4b_capacity_summary.csv
notebooks/01_audit_toolcall_4b_sources_cpu_colab.ipynb
notebooks/02_build_toolcall_4b_corpus_cpu_colab.ipynb
scripts/audit_source_capacity.py
scripts/merge_capacity_reports.py
scripts/phase2_build_corpus.py
scripts/hf_stream_worker.py
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

## Phase 2 completion condition

The build is ready for Kaggle only when the final checksum pass succeeds and
this marker exists:

```text
phase2/final/toolcall_4b/COMPLETE
```

Expected exact split totals are 4,007,835,072 training tokens, 5,000,000
general-validation tokens, and 2,000,000 structured-validation tokens.
