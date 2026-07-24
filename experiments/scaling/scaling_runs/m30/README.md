# M30 scaling runs

M30 is the 29,990,784-parameter member of the scaling family.

```text
layers:       10
hidden size:  384
heads:        6
head size:    64
MLP ratio:    4
vocabulary:   32,000
context:      512
```

Run `m30_colab.ipynb` on a free Google Colab T4. Execute the scientific configurations independently in this order:

1. `configs/low_50m.json`
2. `configs/medium_100m.json`
3. `configs/high_200m.json`

Do not initialize a longer run from a shorter run's checkpoint. `--resume auto` is only for resuming the same named run after interruption.

Run from the parent `scaling_runs` directory. The notebook copies the shared
data bundle into `data/scaling_470m`, verifies it, starts local TensorBoard, and
writes checkpoints to `runs/`. It downloads a full checkpoint archive after
each point so the exact run can be restored and resumed in another Colab
runtime without Google Drive.
