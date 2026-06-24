# ToolCall-200M Dataset Strategy

## 1. Project Goal

**ToolCall-200M** is a ~200M parameter decoder-only language model trained from scratch for local function calling, tool routing, and structured JSON action generation.

The model is not intended to be a general chatbot. Its target capability is:

```text
natural language request + available tool schemas -> valid tool-call decision JSON
```

The model should learn to:

- Select the correct tool from a provided schema set.
- Emit valid JSON arguments.
- Detect missing required fields.
- Ask for clarification when needed.
- Avoid hallucinating unavailable tools or unsupported arguments.
- Handle single-tool, multi-tool, chained, and no-call scenarios.
- Run locally as a lightweight routing/action layer inside larger agentic systems.

---

## 2. High-Level Dataset Philosophy

A 200M model does not have enough capacity or compute budget to become a broad general assistant. The dataset must therefore be biased toward **structured language** from the beginning.

The pretraining corpus should not be mostly generic web text. Instead, the model should see a large amount of:

- JSON
- YAML
- Markdown
- API documentation
- OpenAPI schemas
- SDK documentation
- function signatures
- tool-call traces
- command/action examples
- short user requests
- task-oriented conversations

The main objective is to build a model that is unusually strong at **structured action generation**, not general knowledge.

---

## 3. Target Training Phases

| Phase | Purpose | Target Tokens | Minimum Tokens | Hardware |
|---|---|---:|---:|---|
| Stage 0 | Tokenizer/data prep | N/A | N/A | CPU / Colab |
| Stage 1 | Base pretraining pilot | 1B | 1B | Kaggle 2xT4 |
| Stage 2 | Main base continuation | +1B to +2B | +1B | Kaggle 2xT4 |
| Stage 3 | Structured continued pretraining | 500M-800M | 300M | Kaggle 2xT4 |
| Stage 4 | SFT | 50M-120M | 30M | Colab 1xT4 / Kaggle 2xT4 |
| Stage 5 | Preference alignment | 5M-15M | 5M | Colab 1xT4 |
| Stage 6 | Long-schema polish | 50M-150M | optional | Colab 1xT4 / Kaggle 2xT4 |

Recommended practical run:

```text
2B base pretraining + 600M structured CPT + 80M SFT + 10M DPO/ORPO
```

Minimum viable run:

```text
1B base pretraining + 300M structured CPT + 30M-40M SFT + 5M DPO/ORPO
```

Strong run:

```text
3B base pretraining + 800M structured CPT + 120M SFT + 15M DPO/ORPO + long-schema tune
```

---

## 4. Stage 0: Tokenizer and Data Preparation

### 4.1 Tokenizer Goal

The tokenizer must be efficient for both natural language and structured action text.

It should handle:

- English user instructions
- JSON object syntax
- tool names such as `calendar.create_event`
- snake_case and camelCase fields
- timestamps and dates
- URLs
- file paths
- API names
- Markdown and YAML
- short function signatures

### 4.2 Tokenizer Recommendation

Use a **32k BPE or SentencePiece tokenizer**.

Recommended configuration:

```yaml
vocab_size: 32000
normalization: standard unicode normalization
byte_fallback: true if using SentencePiece
special_tokens:
  - <|system|>
  - <|user|>
  - <|assistant|>
  - <|tool_schema|>
  - <|tool_call|>
  - <|tool_result|>
  - <|json|>
  - <|end|>
```

Avoid adding too many special tokens. The model should learn normal JSON directly.

### 4.3 Tokenizer Training Corpus

Target tokenizer training text size:

```text
5B-10B characters sampled from the final training mixture
```

Suggested mixture:

| Source Type | Share |
|---|---:|
| Clean web / educational English | 35% |
| JSON / YAML / Markdown / config files | 25% |
| API docs / OpenAPI specs / SDK docs | 20% |
| Synthetic tool-call examples | 10% |
| Emails / task-like / productivity text | 10% |

### 4.4 Data Format

All training data should be converted into packed token shards before GPU training.

Recommended format:

```text
data/
  tokenizer/
  raw/
  processed/
  tokenized/
    train_00000.bin
    train_00001.bin
    val_00000.bin
  manifests/
    stage1_base.jsonl
    stage2_base_continuation.jsonl
    stage3_structured_cpt.jsonl
    stage4_sft.jsonl
```

Each manifest row should track:

```json
{
  "source": "fineweb_edu_sample",
  "split": "train",
  "stage": "base_pretraining",
  "license": "source_license_here",
  "document_count": 12345,
  "estimated_tokens": 100000000,
  "processing_steps": ["clean", "dedup", "tokenize", "pack"],
  "sha256": "..."
}
```

---

## 5. Stage 1: Base Pretraining Pilot

### 5.1 Purpose

The first 1B tokens are used to validate:

- model architecture
- tokenizer quality
- loss curve
- dataloader throughput
- checkpoint/restart reliability
- basic JSON/text generation

This is not expected to produce the final model.

### 5.2 Target Size

```text
1B tokens
```

### 5.3 Dataset Mixture

| Data Type | Share | Target Tokens |
|---|---:|---:|
| Clean English / educational web text | 45% | 450M |
| JSON / YAML / Markdown / config text | 20% | 200M |
| API docs / SDK docs / OpenAPI text | 15% | 150M |
| Code snippets, mainly Python/JS/TypeScript | 10% | 100M |
| Synthetic tool/action text | 5% | 50M |
| Email/task/conversation text | 5% | 50M |

### 5.4 Candidate Datasets and Sources

#### Clean English / educational text

- FineWeb-Edu sample
- FineWeb high-quality filtered sample
- SlimPajama sample
- RedPajama-V2 filtered subset
- Zyda filtered sample

FineWeb is a 15T token web dataset, and FineWeb-Edu is a 1.3T token educational subset. These are useful because they are already filtered for language-model pretraining quality.

#### Structured text

- GitHub permissively licensed JSON/YAML/Markdown files
- public configuration repositories
- OpenAPI schema collections
- software documentation
- package README files
- API reference documentation

#### Code snippets

- permissively licensed GitHub code samples
- Python package docs
- TypeScript SDK examples
- CLI examples
- small function definitions and docstrings

### 5.5 Filtering Rules

Apply:

- language filtering for English
- exact deduplication
- near-deduplication using MinHash or SimHash
- removal of boilerplate pages
- removal of extremely short documents
- removal of generated spam
- removal of secrets/API keys/password-like strings
- benchmark decontamination for held-out eval sets

---

## 6. Stage 2: Main Base Pretraining Continuation

### 6.1 Purpose

Build the core language model capability while increasing structured-text bias.

### 6.2 Target Size

```text
+1B to +2B tokens
```

Recommended total base pretraining:

```text
2B tokens minimum, 3B tokens if throughput is good
```

### 6.3 Dataset Mixture

| Data Type | Share | +1B Tokens | +2B Tokens |
|---|---:|---:|---:|
| Clean English / educational text | 40% | 400M | 800M |
| JSON / YAML / Markdown / config text | 20% | 200M | 400M |
| API docs / OpenAPI / SDK docs | 20% | 200M | 400M |
| Code snippets | 10% | 100M | 200M |
| Synthetic tool/action text | 5% | 50M | 100M |
| Email/task/conversation text | 5% | 50M | 100M |

### 6.4 Notes

Do not remove general text entirely. The model still needs enough natural language ability to understand varied user requests.

However, avoid spending most of the token budget on low-value broad web data. The model target is function calling, not general knowledge.

---

## 7. Stage 3: Structured Continued Pretraining

### 7.1 Purpose

Bias the pretrained model toward the domain of tool use.

The model should see large volumes of:

- tool schemas
- function signatures
- API docs
- JSON schemas
- OpenAPI specs
- tool-call traces
- multi-step tool examples
- tool-result interpretation text

### 7.2 Target Size

```text
500M-800M tokens
```

Minimum:

```text
300M tokens
```

Recommended:

```text
600M tokens
```

### 7.3 Dataset Mixture

| Data Type | Share | For 600M Tokens |
|---|---:|---:|
| API docs / OpenAPI specs / SDK docs | 25% | 150M |
| Function-call traces | 25% | 150M |
| JSON / YAML / config / schema text | 20% | 120M |
| Tool-use synthetic conversations | 15% | 90M |
| Code snippets around APIs | 10% | 60M |
| General instruction text | 5% | 30M |

### 7.4 Candidate Datasets and Sources

#### Public tool/function-calling datasets

- ToolBench / ToolLLM
- Salesforce `xlam-function-calling-60k`
- APIGen-derived data
- ToolACE-style generated data if available
- BFCL-style examples for evaluation only, not train contamination
- NESTFUL for nested API-call evaluation and optional training-style inspiration

ToolBench collected 16,464 RESTful APIs across 49 categories and generated tool-use instructions and solution paths. APIGen collected 3,673 executable APIs across 21 categories and released 60,000 verified function-calling examples through Salesforce xLAM. NESTFUL contains 1,800+ executable nested API-call sequences.

#### Custom schema sources

Create synthetic schemas for:

- calendar
- email
- tasks
- notes
- files
- contacts
- browser/search
- spreadsheets
- jobs
- GitHub issues/PRs
- simple CRM
- support tickets
- local safe shell-like operations

### 7.5 Structured CPT Format

Use mixed formats, not only final SFT format.

Examples:

```text
<|tool_schema|>
{
  "name": "calendar.create_event",
  "description": "Create a calendar event",
  "parameters": {
    "title": "string",
    "start_time": "ISO-8601 datetime",
    "end_time": "ISO-8601 datetime"
  },
  "required": ["title", "start_time", "end_time"]
}
<|end|>
```

```text
User request: Remind me to submit the form tomorrow morning.
Available tools: tasks.create, reminders.create, email.draft
Correct decision: reminders.create
Required arguments: title, due_time
<|end|>
```

```json
{
  "decision": "call",
  "tool_name": "reminders.create",
  "arguments": {
    "title": "Submit the form",
    "due_time": "tomorrow morning"
  }
}
```

---

## 8. Stage 4: Supervised Fine-Tuning

### 8.1 Purpose

Teach the exact model interface.

Input:

```text
system instruction + user request + available tool schemas
```

Output:

```json
strict tool-call decision JSON
```

### 8.2 Target Size

Recommended:

```text
50M-120M tokens
```

Minimum:

```text
30M tokens
```

Practical target:

```text
80M tokens
```

### 8.3 SFT Task Distribution

| Task Type | Share |
|---|---:|
| Single function call | 25% |
| Missing required argument detection | 15% |
| Similar-tool disambiguation | 15% |
| Parallel function calls | 10% |
| Function chaining | 10% |
| Nested calls | 5% |
| No-call / out-of-scope / refusal | 10% |
| Tool result interpretation | 10% |

This distribution follows the idea that function calling is not one task. It includes function-name detection, parameter-value detection, chaining, parallel calls, nested calls, next-best-function prediction, and response generation.

### 8.4 SFT Output Schema

Use one canonical schema:

```json
{
  "decision": "call | ask_clarification | no_call",
  "tool_name": "string | null",
  "arguments": {},
  "missing_required_fields": [],
  "confidence": 0.0
}
```

For multiple calls:

```json
{
  "decision": "call",
  "calls": [
    {
      "tool_name": "calendar.create_event",
      "arguments": {}
    },
    {
      "tool_name": "email.draft",
      "arguments": {}
    }
  ],
  "missing_required_fields": [],
  "confidence": 0.0
}
```

### 8.5 SFT Data Sources

Use a blend of:

1. Public function-calling data
   - Salesforce `xlam-function-calling-60k`
   - ToolBench-derived examples
   - APIGen-style examples

2. Custom synthetic productivity tools
   - calendar
   - email
   - tasks
   - reminders
   - files
   - notes
   - browser/search
   - spreadsheet operations
   - GitHub operations

3. Teacher-distilled examples
   - generated by strong teacher models
   - verified by JSON/schema validators
   - repaired by teacher critique
   - deduplicated and balanced

4. Negative examples
   - missing required field
   - unavailable tool
   - ambiguous pronoun
   - unsupported request
   - destructive action requiring confirmation
   - invalid date/time
   - hallucinated argument examples

---

## 9. Stage 5: Preference Alignment

### 9.1 Purpose

Alignment here does not mean general helpfulness. It means better decision behavior around tools.

The model should learn to:

- ask clarification instead of hallucinating missing arguments
- choose no-call when the request is outside the available tools
- avoid destructive actions without confirmation
- prefer valid minimal JSON
- avoid verbose explanations when only a tool call is required
- prefer correct tools over semantically similar wrong tools

### 9.2 Target Size

```text
5M-15M tokens
```

Recommended:

```text
10M tokens
```

### 9.3 Method

Use **DPO** or **ORPO**.

Avoid PPO/RLHF/online RL at this stage.

### 9.4 Preference Pair Types

| Scenario | Chosen | Rejected |
|---|---|---|
| Missing required field | ask clarification | hallucinated value |
| Tool unavailable | no_call | fake tool call |
| Similar tools | correct tool | close but wrong tool |
| Date normalization | ISO/structured date | vague raw date |
| Destructive action | ask confirmation | execute immediately |
| JSON format | valid schema | invalid JSON |
| Multi-action | all required calls | partial call only |

---

## 10. Stage 6: Long-Schema / Robustness Polish

### 10.1 Purpose

Final targeted training for cases where the model sees many tools or long tool schemas.

### 10.2 Target Size

```text
50M-150M tokens
```

This stage is optional and should only be done after the model performs well on basic tool calling.

### 10.3 Data Types

| Data Type | Share |
|---|---:|
| Long schema examples with 10-30 tools | 35% |
| Similar tool names / confusing APIs | 20% |
| Multi-call and chained calls | 20% |
| Tool-result interpretation | 15% |
| Robustness perturbations | 10% |

Robustness perturbations:

- reordered tools
- paraphrased user request
- extra irrelevant context
- typo/noise in user request
- similar function names
- optional-field distractors

---

## 11. Dataset Distillation Strategy

### 11.1 Where Distillation Should Be Used

Use distillation primarily in:

- SFT data generation
- preference pair generation
- hard negative generation
- semantic verification
- data repair

Do not use expensive teacher generation for the full base pretraining corpus.

### 11.2 Teacher Model Selection

Use a teacher committee rather than a single teacher.

Suggested teachers:

| Teacher Type | Use |
|---|---|
| xLAM-style function-calling models | canonical tool-call examples |
| Qwen/DeepSeek coder-style models | API/code/schema examples |
| strong general instruction models | paraphrases, ambiguity, hard cases |
| API models if budget allows | verification and repair of difficult examples |

### 11.3 Distillation Pipeline

1. Build a tool schema pool.
2. Generate user requests for each schema.
3. Generate canonical tool-call output.
4. Validate output with JSON schema checks.
5. Execute mock tools when possible.
6. Use teacher critique for semantic repair.
7. Create rejected outputs for DPO/ORPO.
8. Deduplicate and balance by task type.
9. Keep a held-out evaluation split before any generation loops.

### 11.4 Verification Rules

Every distilled example must pass:

- JSON parse
- schema validation
- tool exists
- required fields present or correctly listed as missing
- argument type correctness
- enum correctness
- no hallucinated tool
- no unsupported field
- no train/eval leakage

---

## 12. Evaluation Strategy

### 12.1 Public Evaluation

Use public benchmarks for directional comparison:

- BFCL-style function-calling evaluation
- ToolBench held-out examples
- APIGen/xLAM held-out examples
- NESTFUL for nested API-call sequences

BFCL evaluates API-call generation according to provided specifications and includes Python, Java, JavaScript, SQL, and REST API cases.

### 12.2 Custom Evaluation Set

Create a held-out custom benchmark of **2,000-5,000 examples**.

| Category | Target Count |
|---|---:|
| Single call | 500 |
| Missing fields | 400 |
| Similar tools | 400 |
| Multi-call | 300 |
| Function chaining | 300 |
| No-call/out-of-scope | 300 |
| Destructive confirmation | 200 |
| Tool result interpretation | 300 |
| Long schema | 200 |

### 12.3 Metrics

| Metric | Meaning |
|---|---|
| JSON validity | Output is parseable JSON |
| Schema compliance | Output matches required schema |
| Tool-name accuracy | Correct tool selected |
| Argument exact match | Correct argument values |
| Argument F1 | Partial argument correctness |
| Missing-field accuracy | Correctly asks for missing info |
| No-call accuracy | Correctly refuses unavailable tools |
| Multi-call sequence accuracy | Correct ordered tool sequence |
| Hallucinated-tool rate | Calls unavailable tools |
| Latency | Local inference speed |

---

## 13. Expected Training Time

Assume approximate throughput:

| Hardware | Conservative | Realistic | Optimistic |
|---|---:|---:|---:|
| 1xT4 | 1k-2.5k tok/s | 2.5k-4k tok/s | 4k-5k tok/s |
| 2xT4 DDP | 2k-5k tok/s | 5k-8k tok/s | 8k-10k tok/s |

Planning table:

| Stage | Tokens | Hardware | Estimated Time |
|---|---:|---|---:|
| Base pilot | 1B | 2xT4 | 35-140 hrs |
| Base continuation | +1B | 2xT4 | 35-140 hrs |
| Structured CPT | 600M | 2xT4 | 21-84 hrs |
| SFT | 80M | 1xT4 | 6-22 hrs |
| DPO/ORPO | 10M | 1xT4 | 1-4 hrs |
| Long-schema polish | 80M | 1xT4 / 2xT4 | 3-22 hrs |

Conservative practical total:

```text
2B base + 600M CPT + 80M SFT + 10M alignment
= ~2.69B tokens
= roughly 100-365 GPU hours depending on actual throughput
```

At 30 Kaggle GPU-hours/week for heavy stages, expect multiple weeks of training. This project should be checkpointed and staged rather than attempted as one continuous run.

---

## 14. Recommended MVP Dataset Target

The first build should not attempt the full strong version.

MVP target:

| Stage | Tokens |
|---|---:|
| Base pretraining | 1B |
| Structured CPT | 300M |
| SFT | 40M |
| DPO/ORPO | 5M |

MVP goal:

- produce valid JSON consistently
- select correct tools on simple cases
- identify missing required fields
- avoid hallucinated tools
- run locally after quantization

MVP success criteria:

| Metric | Target |
|---|---:|
| JSON validity | >95% |
| Tool-name accuracy on simple calls | >70% |
| Missing-field detection | >60% |
| Hallucinated unavailable tool rate | <10% |
| Local inference | usable on CPU/GPU after quantization |

---

## 15. References

- FineWeb and FineWeb-Edu: `https://arxiv.org/abs/2406.17557`
- SlimPajama: `https://arxiv.org/abs/2309.10818`
- RedPajama-V2: `https://arxiv.org/abs/2411.12372`
- Zyda: `https://arxiv.org/abs/2406.01981`
- ToolLLM / ToolBench: `https://arxiv.org/abs/2307.16789`
- APIGen / xLAM Function Calling 60k: `https://arxiv.org/abs/2406.18518`
- Salesforce xLAM Function Calling 60k: `https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k`
- Granite Function Calling: `https://arxiv.org/abs/2407.00121`
- NESTFUL: `https://arxiv.org/abs/2409.03797`
- MobileLLM: `https://arxiv.org/abs/2402.14905`
- SmolLM2: `https://arxiv.org/abs/2502.02737`
