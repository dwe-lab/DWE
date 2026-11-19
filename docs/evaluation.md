# Evaluation

`configs/evaluation.toml` registers all 14 OPD² chat benchmarks. The runner
`scripts/evaluate.py` calls the pinned original adapters; it does not
implement an alternative scorer.

Sources: OPD² `2dac53fbc80673677fb3bd1d8690392761ee3e52`,
Evalchemy `ce5cea9`. Apply the documented patches in `patches/evalchemy`.
From the OPD² checkout, also apply `patches/opd2/complete-test-coverage.patch`
from this repository with `git apply`. This restores all SciBench questions
and includes all available CodeContests tests without changing the prompts
or answer-comparison rules.

| Domain | Tasks |
|---|---|
| Mathematics | AIME24, AIME25, AMC23, HMMT, MATH500, OlympiadBench, RGMath |
| Science | GPQADiamond, SuperGPQA, SciBench |
| Code | CodeContests, CodeForces, LiveCodeBenchv5, RGAlgorithmic |

Run each task with `--mode preflight`, then `--mode generate`, then `--mode score`.
All modes take `--config`, `--task`, `--evalchemy`, `--opd2`, and `--output`.
Generation additionally requires `--checkpoint`. `--data-root` selects local
copies organized by task, with the original split/config metadata. Record the
source dataset revision, row count and parquet SHA256 in each `manifest.json`.
GPQA requires authorized access to its official gated dataset.

Generation is preserved before scoring. Scoring verifies the recorded adapter,
configuration and dataset manifest against generation. Use a CPU-only isolated
container for code scoring, with no network or model/credential mounts. The
container must provide the same evaluation dependencies and pinned sources.

Compare all methods on the same questions, prompt templates, token budget and
seeds. Repeats are separate sampled answers, reported as mean pass@1, not pass@k
and not independent training runs. Start with one complete pass over all tasks;
add generation repeats afterward. Preserve and reuse completed evaluations only
when their protocol matches. Synthetic RG questions retain their original data
seed across generation repeats.

The evaluation protocol includes distinctions that must remain visible in reports:

- SuperGPQA samples 50 examples per discipline (650 across 13 disciplines).
- RGMath and RGAlgorithmic each use five upstream tasks with 20 examples each.
  The runner checks that no constituent task was silently omitted.
- SciBench includes all 692 questions after normalizing thousands separators
  and Unicode minus signs in numeric gold answers. Preflight checks the loaded
  count before the runner starts generation.
- CodeContests uses all 165 test problems and all public, private, and generated
  tests. A problem is correct only when its solution passes every test.
- LiveCodeBench is the pinned Evalchemy v5 adapter, not a rolling latest release.

A failed task or unavailable dataset is an incomplete evaluation, never a zero
model score. Report such status separately from completed benchmark results.

## CodeForces protocol correction

The pinned adapter includes reference editorials in its prompt. Apply
`patches/evalchemy/no-editorial.patch` to exclude these solutions for every
method. On the pinned data, 254 noninteractive examples contain editorials.
Results are problem-only evaluation and should not be compared as an exact
reproduction of editorial-assisted upstream numbers.

The native adapter excludes 15 interactive problems: 468 source rows yield 453
scored problems. `expected_questions` records this benchmark protocol in the
configuration; the runner verifies generated coverage and records source and
scored counts separately. After removing editorials, its longest prompt is 1,857 tokens, within the
shared 36,864-token window and 32,768-token completion budget.

Run CodeForces scoring in the isolated Linux container with the host user's
UID/GID. The evaluation runs used the container option
`--ulimit nofile=65536:65536`. The runner uses a forkserver and closes worker
processes and pipes after scoring. It uses Evalchemy's native tests and
timeouts. Infrastructure errors fail the scoring command, and the saved
generations can be scored again.
