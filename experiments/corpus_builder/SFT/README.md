# ToolCall-SFT-v1 CPU Colab Builder

This project builds a deterministic, teacher-free, single-turn SFT dataset for
ToolCall-200M. It starts from the gated
`Salesforce/xlam-function-calling-60k` dataset and creates exactly 150,000
validated examples.

Project version: **1.0.3**. This release safely verifies an already exported
1.0.2 dataset. It distinguishes the forbidden top-level confidence label from
a legitimate tool argument named `confidence`, so no rows need to be rebuilt.

## Output contract

The rendered format matches the structured examples already used during
ToolCall-200M pretraining:

```text
<|user|>
...
<|tool_schema|>
...
<|assistant|>
{"arguments": ..., "decision": ..., "missing_required_fields": ..., "tool_name": ...}
```

The allowed decisions are `call`, `ask_clarification`, and `no_call`. The SFT
target deliberately omits the earlier synthetic `confidence` field because no
calibrated confidence label exists.

SFT v1 is single-turn and emits at most one tool call. `ask_clarification`
means the intended tool is known but one or more required arguments are
missing. Parallel calls, sequential calls, and ambiguous-tool clarification
are deferred to SFT v2.

Clarification examples may use a deterministic counterfactual schema. When a
source call supplies an optional argument, the builder can mark that argument
as required in the presented tool schema, remove its value, and label the
result `ask_clarification`. Provenance records this adaptation. This expands
coverage without inventing labels through a teacher model.

## Exact composition

| Category | Rows |
| --- | ---: |
| Valid calls | 60,000 |
| Missing required fields | 30,000 |
| No applicable tool | 22,500 |
| Confusable tools | 15,000 |
| Schema constraints | 15,000 |
| Injection resistance | 7,500 |
| **Total** | **150,000** |

The final split is 135,000 train, 7,500 validation, and 7,500 test. Tool-schema
families are assigned to only one split. Every derivative of a source seed
therefore stays in the same split.

## Colab use

1. Open `notebooks/build_toolcall_sft_v1_cpu_colab.ipynb` in a CPU Colab.
2. Run the cells in order.
3. Paste a Hugging Face read token when prompted. The token must have access to
   the gated xLAM dataset. It is not written to Drive.
4. Upload the frozen `toolcall_spm_32k.model` when prompted. The notebook saves
   one copy in the persistent project directory.
5. Run the 1,000-row gate, inspect its printed report, and run the full build.

Persistent output is stored under:

```text
/content/drive/MyDrive/ToolCall-SFT-v1/
```

Completed stages have markers under `state/`. Rerunning after a Colab restart
skips completed stages. Delete only the marker for a stage that you explicitly
want to rebuild.

## Final output

```text
ToolCall-SFT-v1/
├── final/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   ├── manifest.json
│   ├── statistics.json
│   ├── DATA_CARD.md
│   ├── ATTRIBUTION.md
│   └── COMPLETE
├── pilot/
│   ├── pilot_1000.jsonl
│   └── pilot_report.json
├── source/
│   └── normalized_single_call_seeds.jsonl
├── state/
└── tokenizer/
    └── toolcall_spm_32k.model
```

Each JSONL record contains the raw `request`, canonical `tools`, strict
`target`, source/transformation provenance, `prompt_text`, `target_text`, and
`full_text`. A later SFT trainer should compute loss only on `target_text`.

## Validation performed

- source JSON parsing and single-call filtering;
- called-tool existence;
- required-field checks;
- argument-name, type, and enum checks;
- strict target keys and decision invariants;
- exact and bounded near-duplicate filtering;
- schema-family-disjoint split assignment;
- exact category and split quotas;
- frozen SentencePiece length check at 2,048 tokens;
- final row, ID, hash, and split-leakage verification.

## Provenance

The source xLAM dataset contains 60,000 APIGen examples and is released under
CC BY 4.0. The exported data card and attribution file retain the source name,
license, URL, APIGen citation, source record ID, and deterministic
transformation type. Review the original dataset terms before redistribution.
