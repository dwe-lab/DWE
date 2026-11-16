# Estimator architecture and ablations

DWE computes the full current logit gradient, samples a sparse adjoint, and uses
only its selected output rows in backward. The scalar loss remains the exact KL.
At fixed parameters and sampled prefixes, the raw gradient is unbiased.

| Component | Role | Implementation |
|---|---|---|
| Positive/negative pairs | Preserve the zero-sum structure and residual marginals | `sampling.py` |
| Fixed geometry order | Couple the two marginals through output-row projections | `geometry.py` |
| Stratified quantiles | Reduce covariance relative to iid draws of the same coupled pair | `sampling.py` |
| Sparse head backward | Gather/scatter at most 2K selected rows per position | `loss.py` |

Geometry uses a one-dimensional principal-component order from the initial student
head. PCA runs once with an isolated, fixed seed; it does not consume the training
RNG stream. Distributed workers receive the same permutation from rank zero.
The order stays fixed as the head changes. Its full-hidden-gradient error and
latency benefit are empirical questions; the unbiasedness holds for any fixed
permutation. Stratification has the covariance guarantee stated in
[theory.md](theory.md) for the same coupling.

The forward still computes full-vocabulary distributions and scans the residual.
A trainable head retains a dense gradient accumulation buffer and optimizer state.
This implementation targets output-head backward arithmetic.

## Training

```bash
python -m dwe.train --config configs/train.toml \
  --backend quantile --output-dir .local/runs/dwe
```

`quantile` selects geometry plus stratification by default. The configured `pairs`
is K, and `chunk_size` controls working memory. The example recipe retains K=4;
changing it defines a new run. It does not establish the best budget empirically.

Native checkpoints and the final saved model include `estimator.json` and, for
geometry runs, `geometry.pt`. Resume restores the saved permutation and checks
the estimator, budget, ordering, temperature, and KL direction. Missing or
incompatible state causes an error before training. Use the corresponding pinned
release for historical checkpoints without this metadata.

```bash
python -m dwe.train --config configs/train.toml \
  --backend quantile --output-dir .local/runs/dwe \
  --resume .local/runs/dwe/checkpoint-50
```

Optimizer, scheduler, data progression and RNG restoration remain in TRL. Geometry
setup time is recorded separately from update timing. The run manifest records
the permutation hash and resolved estimator settings.

## Geometry ablation

Use the same initial weights, prompts, K, precision, masks and token normalization.
Both variants retain positive/negative pairs, stratified sampling and sparse
backward. The vocabulary order is the only changed component.

| Variant | Backend / order | Configuration |
|---|---|---|
| DWE with geometry | `quantile` / `principal` | Fixed initial output-head order |
| DWE with natural order | `quantile` / `natural` | Existing vocabulary order |

The training flag `--vocabulary-order natural` selects the natural-order control.
Reuse existing natural-order training results when the initial checkpoint, K,
data, seeds, training budget, objective and runtime settings match. In particular,
geometry at K=4 versus natural order at K=8 changes two factors and cannot isolate
geometry. Match the evaluation protocol as well; compare speed under the same
hardware and timing definition.

Positive/negative pairing is supported by the unbiasedness analysis, and
stratification by the covariance result in [theory.md](theory.md). The main
efficiency comparison measures the sparse backward's computational benefit.
This experiment plan adds no separate training ablations for these components.
The `coordinate`, `independent` and `coupled` controls remain available for
correctness checks and targeted diagnostics.

Start with the fixed-state diagnostic:

```bash
python -m dwe.benchmark --config configs/ablation.example.toml \
  --output .local/results/ablation.json
```

This example requires a captured tensor file at `.local/capture.pt`. It compares
the two orders at a single fixed K (4 in the example), measuring hidden/head
gradient error, total head latency and CUDA allocation. Match K to the training
comparison; no budget sweep is included. Exact gradients are computed internally
as the error reference. Geometry construction is measured separately.

For training results, report downstream performance and total training time for
the same two variants. Local gradient errors explain the ordering's effect;
training results establish its practical value.
