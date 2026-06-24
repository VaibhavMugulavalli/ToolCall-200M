# ToolCall-200M Model Strategy

## 1. Project Goal

**ToolCall-200M** is a compact decoder-only language model trained from scratch for function calling, tool routing, and structured JSON generation.

The model is designed to act as a local action layer inside agentic systems. It should not be evaluated as a general chatbot.

Primary target behavior:

```text
user request + available tool schemas -> valid tool-call decision JSON
```

Core capabilities:

- select correct tools
- emit schema-valid JSON
- extract arguments
- detect missing required fields
- ask clarification when needed
- handle no-call / unsupported requests
- support multi-tool and chained tool calls
- run locally with low latency after quantization

---

## 2. Design Principles

For a 200M model, architecture and data quality matter heavily. The design should prioritize:

- training stability on T4 GPUs
- high token throughput
- simple implementation
- compatibility with Hugging Face / PyTorch tooling
- efficient structured-output behavior
- easy export and quantization

Avoid experimental architecture changes in the mainline model. Kronecker embeddings, sparse pretraining, and exotic low-rank pretraining should be treated as future research, not part of the first working build.

---

## 3. Recommended Architecture

Use a Llama/MobileLLM-style decoder-only transformer.

Recommended mainline configuration:

```yaml
model_type: decoder_only_transformer
parameter_target: 180M-220M
architecture_style: llama_like_deep_thin
vocab_size: 32000
hidden_size: 768
num_hidden_layers: 24
num_attention_heads: 12
num_key_value_heads: 4
intermediate_size: 2048
activation_function: swiglu
normalization: rmsnorm
position_encoding: rope
max_position_embeddings_pretrain: 2048
max_position_embeddings_polish: 4096
attention: grouped_query_attention
embedding_tying: true
bias: false
precision: fp16_training
```

This should land close to the 200M parameter range depending on exact implementation details and whether embeddings are tied.

---

## 4. Why This Architecture

### 4.1 Deep-thin design

Sub-billion models benefit from architectures that are not simply miniature versions of large LLMs. MobileLLM-style work shows that architecture matters strongly at small scale, especially deep-thin layouts, embedding sharing, and grouped-query attention.

For ToolCall-200M, deep-thin is preferred over shallow-wide because the model needs many transformation steps for:

- schema reading
- argument extraction
- tool disambiguation
- JSON formatting
- missing-field decisions

### 4.2 Grouped-query attention

Use GQA:

```yaml
num_attention_heads: 12
num_key_value_heads: 4
```

GQA reduces KV cache and attention overhead while keeping enough query heads for expressivity. It also makes later inference more efficient.

### 4.3 SwiGLU and RMSNorm

Use modern Llama-style blocks:

- RMSNorm
- SwiGLU MLP
- RoPE
- no attention/MLP bias

This keeps the model close to well-supported open architectures.

### 4.4 Tied embeddings

Use tied input/output embeddings.

Reasons:

- reduces parameter count
- improves small-model efficiency
- keeps the embedding table from dominating the 200M budget

---

## 5. Parameter Estimate

Approximate configuration:

```yaml
vocab_size: 32000
hidden_size: 768
layers: 24
heads: 12
kv_heads: 4
intermediate_size: 2048
```

Approximate parameter allocation:

| Component | Approx Params |
|---|---:|
| Token embeddings / LM head, tied | ~24.6M |
| Attention blocks | ~65M-75M |
| MLP blocks | ~75M-90M |
| Norms and small layers | <2M |
| Total | ~170M-195M depending on implementation |

If the final parameter count is below target, increase one of:

- layers: 24 -> 26
- intermediate size: 2048 -> 2304
- hidden size: 768 -> 800, if implementation supports clean head divisibility

Preferred adjustment:

```yaml
num_hidden_layers: 26
```

Avoid increasing vocab size unnecessarily.

---

## 6. Alternative Configurations

### 6.1 Safer/faster 170M-190M version

Use this if 2xT4 throughput is poor.

```yaml
vocab_size: 32000
hidden_size: 768
num_hidden_layers: 22
num_attention_heads: 12
num_key_value_heads: 4
intermediate_size: 2048
max_position_embeddings: 2048
```

### 6.2 Main 200M version

Recommended default.

```yaml
vocab_size: 32000
hidden_size: 768
num_hidden_layers: 24
num_attention_heads: 12
num_key_value_heads: 4
intermediate_size: 2048
max_position_embeddings: 2048
```

### 6.3 Ambitious 220M-250M version

Only use if throughput and memory are comfortable.

```yaml
vocab_size: 32000
hidden_size: 768
num_hidden_layers: 28
num_attention_heads: 12
num_key_value_heads: 4
intermediate_size: 2048
max_position_embeddings: 2048
```

This version may be better, but it increases training time. Given limited GPU quota, the 24-layer model is the preferred mainline.

---

## 7. Context Length Strategy

Do not train at 4096 context from the beginning.

Recommended:

| Stage | Context Length |
|---|---:|
| Base pretraining pilot | 1024 or 2048 |
| Main base pretraining | 2048 |
| Structured CPT | 2048 |
| SFT | 2048 |
| Long-schema polish | 4096 optional |

If throughput is poor, start Stage 1 at 1024 context and move to 2048 after the model and data pipeline are validated.

For ToolCall-200M, most function-calling examples fit within 2048 tokens. Long schema support can be taught in a small final 4096-context phase.

---

## 8. Training Stack

Recommended main stack:

```yaml
framework: pytorch
trainer: custom hf/lit-gpt style trainer
parallelism: ddp
precision: fp16_amp
attention_backend: pytorch_sdpa
optimizer: adamw
scheduler: cosine_with_warmup
checkpointing: frequent_resume_safe
```

Use:

- PyTorch DDP over 2xT4 for pretraining
- pre-tokenized packed binary shards
- gradient accumulation
- PyTorch SDPA
- optional activation checkpointing
- optional fused kernels if stable

Avoid initially:

- ZeRO-3 CPU offload
- FSDP unless DDP cannot fit
- online tokenization
- 4096 context from the start
- experimental embedding layers
- sparse/low-rank pretraining tricks

---

## 9. Batch and Optimization Strategy

Think in tokens per optimizer step, not examples.

Starting point:

```yaml
sequence_length: 2048
micro_batch_size_per_gpu: 2
gpus: 2
gradient_accumulation_steps: 32
global_tokens_per_step: 262144
```

If memory allows:

```yaml
micro_batch_size_per_gpu: 4
gradient_accumulation_steps: 16
```

If memory is tight:

```yaml
micro_batch_size_per_gpu: 1
gradient_accumulation_steps: 64
activation_checkpointing: true
```

### Optimizer

Use AdamW.

Starting hyperparameters:

```yaml
optimizer: adamw
learning_rate_peak: 3.0e-4
weight_decay: 0.1
beta1: 0.9
beta2: 0.95
epsilon: 1.0e-8
grad_clip_norm: 1.0
warmup_tokens: 20M-50M
scheduler: cosine_decay
min_lr_ratio: 0.1
```

For SFT:

```yaml
learning_rate: 1.0e-5 to 5.0e-5
weight_decay: 0.0 to 0.05
epochs: token-budget based, not fixed
```

---

## 10. Training Phases

### 10.1 Stage 1: Base pretraining pilot

```yaml
tokens: 1B
context_length: 1024 or 2048
hardware: 2xT4
objective: causal_lm
```

Primary checks:

- stable loss curve
- no data stalls
- reasonable tokens/sec
- valid checkpoint reload
- basic JSON generation

### 10.2 Stage 2: Main base continuation

```yaml
tokens: +1B to +2B
context_length: 2048
hardware: 2xT4
objective: causal_lm
```

Continue only if the pilot is healthy.

### 10.3 Stage 3: Structured continued pretraining

```yaml
tokens: 500M-800M
context_length: 2048
hardware: 2xT4
objective: causal_lm
```

Data becomes heavily biased toward:

- APIs
- tools
- schemas
- JSON
- function-call traces

### 10.4 Stage 4: SFT

```yaml
tokens: 50M-120M
context_length: 2048
hardware: 1xT4 or 2xT4
objective: supervised causal_lm
```

For a 200M model, run full SFT first. LoRA is optional for quick experiments but should not replace full SFT in the mainline.

### 10.5 Stage 5: DPO/ORPO

```yaml
tokens: 5M-15M
context_length: 2048
hardware: 1xT4
objective: preference_optimization
```

Use this only after SFT.

Alignment should focus on:

- no hallucinated tools
- missing-field clarification
- valid JSON
- safe handling of destructive actions
- no-call decisions

### 10.6 Stage 6: Long-schema polish

```yaml
tokens: 50M-150M
context_length: 4096
hardware: 1xT4 or 2xT4
objective: causal_lm or sft
```

Optional. Use only if the model already performs well at 2048 context.

---

## 11. Expected Training Time

Assumed throughput:

| Hardware | Conservative | Realistic | Optimistic |
|---|---:|---:|---:|
| 1xT4 | 1k-2.5k tokens/s | 2.5k-4k tokens/s | 4k-5k tokens/s |
| 2xT4 DDP | 2k-5k tokens/s | 5k-8k tokens/s | 8k-10k tokens/s |

Estimated times:

| Stage | Tokens | Hardware | Estimated Time |
|---|---:|---|---:|
| Base pilot | 1B | 2xT4 | 35-140 hrs |
| Base continuation | +1B | 2xT4 | 35-140 hrs |
| Structured CPT | 600M | 2xT4 | 21-84 hrs |
| SFT | 80M | 1xT4 | 6-22 hrs |
| DPO/ORPO | 10M | 1xT4 | 1-4 hrs |
| Long-schema polish | 80M | 1xT4 / 2xT4 | 3-22 hrs |

The biggest speedups will come from:

- packed pre-tokenized data
- avoiding padding
- avoiding online tokenization
- using 1024/2048 context instead of 4096
- DDP instead of offload-heavy strategies
- frequent checkpointing to avoid lost Kaggle sessions

---

## 12. Inference Strategy

The final model should be deployed with constrained validation around it.

Recommended inference stack:

```text
user request + tool schemas
-> ToolCall-200M
-> JSON parser
-> schema validator
-> retry/repair if invalid
-> execute or ask clarification
```

Do not rely only on raw model generation.

Use:

- low temperature, usually 0.0-0.2
- max output length limits
- stop tokens
- JSON schema validation
- unavailable-tool checking
- required-field checking
- optional constrained decoding if available

---

## 13. Evaluation Metrics

Evaluate the model as a tool-call model, not as a chat model.

Primary metrics:

| Metric | Target |
|---|---:|
| JSON validity | >95% MVP, >98% strong |
| Schema compliance | >90% MVP |
| Simple tool-name accuracy | >70% MVP |
| Argument F1 | >65% MVP |
| Missing-field detection | >60% MVP |
| No-call accuracy | >60% MVP |
| Hallucinated unavailable tool rate | <10% MVP |
| Local inference latency | acceptable for interactive use |

Advanced metrics:

- multi-call sequence accuracy
- nested call accuracy
- long-schema degradation
- similar-tool disambiguation
- robustness under paraphrase/noise
- invalid JSON repair rate

---

## 14. Baselines

Compare ToolCall-200M against:

- regex/rule-based router
- BM25/schema keyword matching
- prompted SmolLM2-135M/360M
- prompted Qwen small model
- prompted larger local model as upper bound
- ToolCall-200M base before SFT
- ToolCall-200M after SFT
- ToolCall-200M after DPO/ORPO

The project should show that a from-scratch 200M model can beat generic tiny models on tool-call specialization after targeted pretraining and SFT.

---

## 15. Unsloth Usage

Unsloth is not recommended for full from-scratch pretraining.

Use Unsloth only if the model can be made compatible with supported architectures and only for:

- SFT experiments
- LoRA/QLoRA ablations
- DPO experiments

For the main 200M model, full SFT is feasible and preferred.

---

## 16. Kronecker Embeddings

Kronecker embeddings are not part of the mainline model.

Reason:

- limited GPU budget
- custom architecture complexity
- compatibility risk with Hugging Face/Unsloth/export
- harder debugging

They may be explored later as a research ablation for rare API identifier handling, but the first working model should use standard learned embeddings with tied LM head.

---

## 17. Recommended MVP Model Plan

Use the main 24-layer architecture:

```yaml
vocab_size: 32000
hidden_size: 768
num_hidden_layers: 24
num_attention_heads: 12
num_key_value_heads: 4
intermediate_size: 2048
activation_function: swiglu
normalization: rmsnorm
position_encoding: rope
embedding_tying: true
context_length: 2048
```

Train:

```text
1B base pretraining
300M structured CPT
40M SFT
5M DPO/ORPO
```

This is the first milestone. Only continue to the stronger 2B-3B token plan after the MVP shows useful tool-call behavior.

---

## 18. References

- MobileLLM: `https://arxiv.org/abs/2402.14905`
- SmolLM2: `https://arxiv.org/abs/2502.02737`
- Granite Function Calling: `https://arxiv.org/abs/2407.00121`
- APIGen: `https://arxiv.org/abs/2406.18518`
- ToolLLM / ToolBench: `https://arxiv.org/abs/2307.16789`
- NESTFUL: `https://arxiv.org/abs/2409.03797`
