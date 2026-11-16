# On-policy training and measurements

## Training implementation and selected protocol

The on-policy adapter uses TRL 1.13.0. Generation, optimization, gradient
accumulation, and checkpoint management remain in the upstream trainer.
DWE replaces the output-head logit-gradient estimator. Apply the pinned
loss hook before using the trainer:

```bash
uv sync --locked --extra models --extra training --extra runtime
uv run --locked python scripts/patch_trl.py
uv run --locked dwe-train --config configs/train.toml --backend quantile --output-dir .local/runs/dwe
```

Inspect `configs/train.toml` for model revisions, public data sources,
optimization settings, and generation parameters. Match these settings across
methods. The model and dataset licenses apply to their respective artifacts.

Use `--backend upstream` for the full-gradient control. The architecture guide
covers estimator metadata and resuming checkpoints. Teacher and student must
share vocabulary token IDs. The geometry order is constructed from the initial
student output head and retained with the checkpoint.

## Output-head measurements

```bash
uv sync --locked --extra models --extra oracle --extra gpu
uv run --locked dwe-capture --config configs/capture.example.toml --output .local/capture.pt
uv run --locked dwe-benchmark --config configs/heads.example.toml --output .local/results/heads.json
```

The capture supplies causally aligned teacher and student hidden states and
output heads. Compare methods on identical inputs, precision, sampling budgets,
and chunk sizes. Report warmup and repetition counts with measured results.
Measure complete training updates separately when reporting end-to-end speed.

## Evaluation

Follow the [evaluation guide](evaluation.md) for pinned task adapters, dataset
coverage, generation settings, and scoring. The [KD guide](kd.md) covers
fixed-corpus distillation and its evaluation commands.
