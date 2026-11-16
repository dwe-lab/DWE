# Distillation with Errors

**Sparse Unbiased Backpropagation for Knowledge Distillation**

DWE constructs sparse logit-gradient estimates using positive/negative pairs,
a fixed output-head geometry order, and stratified sampling. The estimate
preserves the full-vocabulary gradient in expectation at fixed inputs and
parameters. The implementation supports off-policy KD and on-policy distillation.

## Installation

Python 3.12 or later is required. Dependencies are pinned in `uv.lock`.

```bash
uv sync --locked
uv run --locked dwe-benchmark --config configs/smoke.toml --output .local/results/smoke.json
```

The smoke example uses synthetic tensors. `data/smoke.jsonl` contains three
handwritten example prompts. Experiment datasets, model weights, teacher caches,
and evaluation outputs are obtained or generated separately by the user.

## Knowledge distillation

```bash
uv sync --locked --extra kd --extra evaluation
```

The [KD guide](docs/kd.md) describes packed text preparation, frozen teacher
hidden-state caching, training, and downstream evaluation. Example configurations
are in `configs/kd-small.toml`, `configs/kd-medium.toml`, and
`configs/kd-scale.toml`. Model and dataset revisions are specified in the configs.

## On-policy distillation

```bash
uv sync --locked --extra models --extra training --extra runtime
uv run --locked dwe-train --config configs/train.toml --backend quantile --output-dir .local/runs/dwe
```

Training uses TRL with a source-verified loss hook. See the
[training guide](docs/experiments.md) and [evaluation guide](docs/evaluation.md)
for setup and commands. The [architecture guide](docs/architecture.md) explains
estimator configuration, geometry, checkpointing, and controlled ablations.

## Python interface

```python
from dwe import LossConfig, distillation_loss
from dwe.geometry import principal_order

order = principal_order(student_head.weight)
loss = distillation_loss(
    student_hidden, student_head.weight,
    teacher_hidden.detach(), teacher_head.weight.detach(),
    config=LossConfig(method="quantile", pairs=8),
    order=order,
)
loss.backward()
```

Inputs must be causally aligned and masked before calling the loss. Teacher and
student heads may have different hidden widths and must share output token IDs.
Pass head biases explicitly when present. Build the geometry order once from the
initial student head and reuse it throughout training.

## Documentation

- [Theory](docs/theory.md)
- [Architecture and ablations](docs/architecture.md)
- [Off-policy KD](docs/kd.md)
- [On-policy training and measurements](docs/experiments.md)
- [Evaluation](docs/evaluation.md)

Upstream sources and pinned revisions are documented in `upstream.json` and
`patches/`. Test fixtures and smoke inputs are synthetic examples.

## License

DWE is released under the [Apache License 2.0](LICENSE). Third-party components
and source patches retain their respective licenses and attribution.
