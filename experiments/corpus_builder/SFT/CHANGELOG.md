# Changelog

## 1.0.3

- Fix final verification falsely treating a legitimate nested tool argument
  named `confidence` as the removed top-level confidence label.
- Parse `target_text` and require it to exactly match the structured `target`
  in canonical JSON form.
- Allow an already exported 150K dataset to be verified without regenerating
  or re-exporting any records.

## 1.0.2

- Generate clarification cases from every supplied schema field; optional
  fields are deterministically promoted to required in the presented
  counterfactual schema before their value is removed.
- Archive and rebuild only incomplete pre-1.0.2 clarification buckets while
  preserving completed work such as the 54,000-row training call bucket.
- Deduplicate distractors by tool name and exclude alternate schemas that reuse
  the target tool name.
- Compute near-duplicate fingerprints from requests, tool names, and targets so
  long schemas no longer dominate similarity filtering.
- Expand schema-constraint eligibility with value-preserving enum adaptations.
- Add version-aware Colab project replacement and streamed subprocess errors.

## 1.0.1

- Parse xLAM legacy types such as `int, optional` into valid JSON Schema types.
- Infer required legacy xLAM parameters when they have neither an optional
  marker nor a default value.
- Preserve explicit Boolean or string `required` values when present.
- Add a regression test covering xLAM's actual optionality convention.

## 1.0.0

- Initial CPU-Colab builder.
