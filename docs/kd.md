# Full-distribution knowledge distillation

Install with `pip install '.[kd,evaluation]'` on the training host.

The KD entry points use a fixed text stream and forward KL at temperature 1.
Training consumes packed corpus text without chat templates or generated reasoning traces.
The teacher's final BF16 hidden states are cached once; its frozen output head
reconstructs the complete target distribution during training. Every method uses
the same cache, causal mask, student initialization and optimizer.

| CLI method | Backward estimator |
|---|---|
| `exact` | Full gradient with chunked autograd and activation checkpointing |
| `coordinate` | IID importance sampling proportional to the logit-gradient magnitude |
| `topk` | Deterministic Top-\(\lvert g\rvert\) truncation, without rescaling |
| `sketch` | Shared column mask from Algorithms 1, 2 and 6 of Bakong et al. |
| `quantile` | DWE with fixed geometry, shared quantiles and stratification |

`--pairs 8` gives a nominal budget of 16 coordinate draws for the sparse methods.
Sketching selects distinct columns shared by each 256-position chunk; the other
samplers may repeat coordinates. Measurements report the actual nonzero count.
`independent`, `coupled`, and `--order natural` provide the component ablations.

The Sketching implementation follows Algorithm 6's squared column-ℓ₁ scores.
Water filling and systematic correlated sampling produce its inclusion mask;
inverse inclusion probabilities weight the selected columns. The shared mask
permits small GEMMs in backward. This is an adaptation to the output head, not
an official implementation or an established performance result.

## Data and training

Packing directly uses `PackedTokenDataset` from
[SE-KD](https://github.com/almogtavor/SE-KD3x) at commit
`08b276383a31fe5c07eb6685f9c4557b78e42880`: documents are joined with EOS,
and EOS and post-EOS targets are masked. A text hash assigns documents to disjoint
training, validation and test pools. Corpus selection and the packed order remain
fixed across methods and seeds. The implementation uses Transformers Trainer for
optimization, gradient accumulation and checkpoint recovery; it does not use
SE-KD's token or sample selection.

```bash
python -m dwe.kd.cache pack --sekd /path/to/SE-KD3x \
  --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --data-revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --output /path/to/text

python -m dwe.kd.cache teacher \
  --model /path/to/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --tokens /path/to/text/train --output /path/to/cache/train

python -m dwe.kd.train --config configs/kd-small.toml \
  --model /path/to/Qwen3-0.6B --method quantile --pairs 8 --seed 42 \
  --cache /path/to/cache/train --validation-cache /path/to/cache/validation \
  --output /path/to/run
```

Build validation and test caches with the same frozen teacher. Cache generation
resumes at the last flushed sequence. `--max-rows 4096` builds a small prefix for
a pilot; use a separate pilot directory. `--tokens` sets its training budget.
`kd-small.toml` selects Qwen3-4B-Instruct-2507 → Qwen3-0.6B; `kd-medium.toml` selects
Qwen3-8B → Qwen3-1.7B; `kd-scale.toml` selects Qwen3-14B → Qwen3-4B.

All three configurations use sequence length 512, microbatch 2, accumulation 8,
Adam8bit at 1e-5, 100 warmup updates, a constant subsequent learning rate and
student activation checkpointing. Student weights and projections use BF16;
KL, analytic adjoints and sparse gradient accumulation use FP32. Tied embeddings
remain tied, including their ordinary input-side gradient. The optimizer and
weight-gradient buffers remain dense.

Training reports actual input tokens and CUDA-synchronized wall time, including
data reads, teacher-head reconstruction and optimization. Validation, checkpoint
I/O and initialization are outside that timer. Geometry setup is recorded
separately. Training saves a model every 2,000 updates and at the final endpoint,
without running intermediate evaluations. After the compared runs finish,
evaluate the initial student and every saved model on the complete seven-task
suite, using identical settings without question subsampling. Exact held-out KL
and perplexity are evaluated separately with the full held-out teacher cache.
Model-only snapshots are retained under `snapshots/`,
with actual token counts and training times in `measurement.json`. Only the last
two full optimizer checkpoints are kept for recovery. Token budgets stop at a complete update (8192 tokens).

Pass the same `--time-budget SECONDS` to compare methods at a common time budget.
The first completed update at or beyond that budget is saved, along with its
actual elapsed time. The `tokens/` and `time/` directories retain the two model
endpoints through hard links; optimizer checkpoints support `--resume`.
Set the common time budget before the compared runs. A budget derived from
completed reference runs requires obtaining their earlier time endpoints
separately; the command does not reconstruct unsaved earlier checkpoints.

## Evaluation and local measurements

```bash
python -m dwe.kd.evaluate --model /path/to/run/tokens \
  --cache /path/to/cache/test --output /path/to/evaluation

python -m dwe.kd.capture --model /path/to/run/tokens \
  --cache /path/to/cache/validation --order /path/to/run/order.pt \
  --output /path/to/capture.pt

python -m dwe.kd.measure --capture /path/to/capture.pt --output /path/to/measurements
```

Downstream evaluation uses unmodified lm-evaluation-harness 0.4.13 tasks:
HellaSwag, PIQA, ARC-Easy, ARC-Challenge, GSM8K, LAMBADA OpenAI and MMLU.
All use zero-shot completion prompts without chat templates; GSM8K uses greedy
generation capped at 512 new tokens. This protocol does not invoke the Qwen chat
template's `enable_thinking` switch. It is a completion evaluation, not an
explicitly configured non-thinking chat evaluation.
The initial student and teacher use these same settings. Retain per-example
outputs. Report normalized accuracy for HellaSwag/PIQA/ARC, flexible extraction
accuracy for GSM8K, last-word accuracy for LAMBADA and accuracy for MMLU.

Local timing uses 30 warmups and 500 CUDA-event measurements, with interleaved
method order at 1K, 4K and 8K valid positions. Component and K ablations use 1K.
The forward includes projections, loss and estimator preparation. Incremental
allocated memory is measured in a fresh process for each main method and size.
Estimator error uses 128 independent draws, BF16 projections and FP32 gradient
products to isolate sampling error. Saved initialization geometry is reused.
These are measurements to run after GPU allocation, not reported KD results.

References: [Unbiased Approximate Vector-Jacobian Products](https://arxiv.org/abs/2602.14701),
[meProp](https://proceedings.mlr.press/v70/sun17c.html),
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
