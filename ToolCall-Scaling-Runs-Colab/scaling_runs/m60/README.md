# M60 scaling runs

M60 is the 60,439,040-parameter member of the scaling family.

```text
layers:       14
hidden size:  512
heads:        8
head size:    64
MLP ratio:    4
vocabulary:   32,000
context:      512
```

Run `m60_colab.ipynb` on a free Google Colab T4. Execute the scientific configurations independently in this order:

1. `configs/low_25m.json`
2. `configs/medium_50m.json`
3. `configs/high_100m.json`

M60 uses micro-batch 4 with eight gradient-accumulation steps to keep the global optimizer batch at 16,384 tokens while reducing T4 memory pressure.

Do not initialize a longer run from a shorter run's checkpoint. `--resume auto` is only for resuming the same named run after interruption.

Run from the parent `scaling_runs` directory. The notebook copies the shared
data bundle into `data/scaling_470m`, verifies it, starts local TensorBoard, and
writes checkpoints to `runs/`. It downloads a full checkpoint archive after
each point so the exact run can be restored and resumed in another Colab
runtime without Google Drive.
