# Dataset path injection

Base: mlfoundations/evalchemy commit `ce5cea9`.

`dataset-paths.patch` exposes the original dataset identifier as a constructor
argument in GPQADiamond, CodeForces and LiveCodeBenchv5. Defaults remain upstream
identifiers. This permits verified local copies of the same evaluation split;
it does not change prompts, examples, answer extraction or scoring.

`source.sha256.json` records the pristine and patched adapter hashes. Apply this
patch to a separate evaluation checkout, not one used by a running experiment.
For HMMT, also use OPD²'s own `patch_hmmt_none_safe` from
`evaluation/extended_eval_setting.py` at commit
`2dac53fbc80673677fb3bd1d8690392761ee3e52`.

Apply `preprocessing-workers.patch` afterward to expose LiveCodeBench's existing
Dataset.map worker count. Its default remains upstream's CPU count;
`configs/evaluation.toml` selects four workers to bound host memory. Dataset
contents, ordering and scoring are unchanged. The second hash manifest records
this additional patch separately.

Apply `no-editorial.patch` after dataset path injection. The pinned CodeForces
adapter appends the reference editorial to the question whenever present. Remove
that block for problem-only evaluation. This is an intentional protocol change,
not a performance patch; report it for all compared methods. The separate hash
manifest verifies the exact change. Native exclusion of interactive problems
and native scoring remain unchanged.
