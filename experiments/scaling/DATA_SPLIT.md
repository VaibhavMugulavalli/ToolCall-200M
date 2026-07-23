# ToolCall data split and stage boundaries

The scaling pilot measures base-language-model loss and throughput. Its reusable
470M-token pool is a downscaled version of the final **base pretraining** mix:

| Base category | Pilot share | Pilot train tokens | MVP 1B equivalent |
| --- | ---: | ---: | ---: |
| Clean English/general | 45% | ~211.5M | ~450M |
| Structured JSON/YAML/Markdown/config/schema text | 20% | ~94M | ~200M |
| API/tool docs and schema descriptions | 15% | ~70.5M | ~150M |
| Code/function-signature text | 10% | ~47M | ~100M |
| Task/action/instruction-style text | 10% | ~47M | ~100M |

The pilot pool is built once and reused from the beginning for every M13, M30,
and M60 point. This keeps tokenizer, document order, source mixture, and held-out
validation constant across the scaling comparison.

## Stages deliberately not produced here

These belong in separate future workspaces and must not be silently mixed into
the scaling base corpus:

| Stage | MVP target | Key sources/purpose |
| --- | ---: | --- |
| Structured CPT | 300M | OpenAPI/schema/config data, tool traces, SDK docs, signatures |
| SFT | 30M–40M | xLAM 60k after access acceptance, transformed OpenAPI examples, synthetic and hand-designed positive/negative cases |
| DPO/ORPO | 5M | Valid-vs-invalid JSON, right-vs-wrong tool, clarification-vs-hallucination preference pairs |

SFT should reserve 5% deterministic validation; preference tuning should also
reserve 5%. Base/CPT validation should remain a fixed 0.5%–1% document-level
holdout. The scaling generator creates a fixed 5M general validation stream and
a separate 1M structured diagnostic stream. Only general validation loss is used
for the scaling fit.

Salesforce xLAM is gated and is therefore intentionally excluded from the
zero-auth base-data generator. It should be added only in the SFT workspace after
the user accepts its access and license conditions.
