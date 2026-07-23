# Pilot Run Summary

Run directory: `runs/tiny_smoke_50m`

## Setup

- Data dir: `../tokenizer/data/tokenized/smoke_50m`
- Tokenizer: `../tokenizer/artifacts/tokenizer/toolcall_spm_32k.model`
- Device: `cuda`
- Parameters: `12,913,920`
- Dataset tokens: `50,000,000`
- Sequence length: `512`
- Batch size: `8`
- Gradient accumulation: `4`
- Effective batch tokens: `16,384`
- Max steps: `1000`
- Learning rate: `0.0003`

## Result

- First logged step: `1`
- First loss: `41.7056`
- Last logged step: `1000`
- Last loss: `12.6615`
- Loss change: `-29.0441`
- Last tokens/sec: `65,069`

## Interpretation

- Training loop completed: `Yes`
- Loss decreased meaningfully: `Yes, from 41.7056 to 12.6615`
- CUDA OOM: `No`
- Checkpoints saved: `Yes, expected at steps 500 and 1000`
- Generation sample usable for sanity check: `Yes if sample files were generated without crashing; quality not expected yet`

## Notes

The tiny 12.9M-parameter pilot successfully validated the packed-shard dataloader, CUDA training loop, causal LM objective, mixed-precision execution, checkpoint writing, and basic generation path on Colab T4.

The final loss remains high at 12.6615, above the approximate uniform-random baseline of ln(32000) ≈ 10.37, so this run should not be interpreted as a quality result. It is a systems smoke test. Before scaling to the 200M model, run a slightly larger pilot with improved training hygiene: LR warmup, cosine decay, periodic validation loss, and checkpoint reload testing.