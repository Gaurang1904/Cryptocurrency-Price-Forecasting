# Task 7 report — final daily evaluation and portfolio handoff

## Status

Task 7 implementation, historical experiments, evidence generation,
documentation, and pre-commit verification completed. Evaluation data ends
2026-07-23; the last OOS origin is 2026-07-14. Nothing here is a current/live
forecast or a portfolio/P&L result.

Prior accepted commits through `8828fb1` were preserved.

## Code change and TDD

Added an explicit `legacy` candidate to
`experiments.feature_ablation.ablation_candidates`. It excludes exactly
`vol_regime`, `drawdown_63d`, and `volume_z21`.

RED:

```text
C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe -m unittest tests.test_feature_ablation.FeatureAblationTests.test_legacy_candidate_excludes_only_new_regime_features -v
FAIL: 'legacy' not found
```

GREEN:

```text
C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe -m unittest tests.test_feature_ablation -v
Ran 2 tests ... OK
```

`apply_patch` was blocked by the worktree's Windows deny-read ACL. Per the
binding ruling, exact path-scoped PowerShell writes were used inside only the
assigned worktree.

## Experiment commands

All commands ran from:

```text
C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.worktrees\daily-prediction-quality
```

Commands:

```powershell
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' -m experiments.feature_ablation
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' -m tree.lgbm
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' -m tree.xgb
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' -m tree.run
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' -m neural.run
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' evaluate.py artifacts/evaluation/daily-tree-20260723-baseline/predictions.parquet artifacts/evaluation/daily-neural-20260723-baseline/predictions.parquet --out docs/evaluation/daily-20260723
```

The evaluator exited 0 and used a previously absent output directory. It did
not overwrite any prior report.

## Raw result summaries and exploratory comparisons

### Feature ablation

The full run completed in about eight minutes and wrote 42 fold rows (seven
candidates by six folds) to `artifacts/feature_ablation.csv`.

Overall values below are origin-count-weighted across folds. Worst-fold is the
maximum fold `pinball_%`.

| candidate | overall pinball % | worst-fold pinball % | overall coverage % |
|---|---:|---:|---:|
| without_funding | 1.459705 | 2.071185 | 79.663866 |
| all | 1.472134 | 2.228898 | 78.833416 |
| without_volatility | 1.474148 | 2.220945 | 79.179436 |
| without_volume | 1.474173 | 2.242686 | 78.981710 |
| legacy | 1.475972 | 2.238526 | 78.991597 |
| without_market | 1.479464 | 2.260984 | 78.843302 |
| without_returns | 1.492380 | 2.314026 | 79.041028 |

Exploratory selection-set observation: `all` measured lower loss than `legacy`
on both overall normalized pinball (1.472134 < 1.475972) and worst-fold
normalized pinball (2.228898 < 2.238526), while combined OOS coverage was inside
78-82%. This comparison was consulted before the later tree run, so it is not
untouched post-selection validation and does not establish a confirmed feature
improvement or locked configuration.

The parent ruling confirmed that the descriptive coverage gate uses combined OOS
coverage, while fold/asset/horizon coverage remains published. `without_funding`
is another exploratory same-fold result, not a post-hoc replacement. Any locked
comparison requires future chronological data strictly after 2026-07-23; rerunning
these folds does not create an untouched test.

### Saved tree rebuilds using the `all` candidate

Both tree rebuilds using the `all` candidate completed on 9,182 fit rows plus 2,300
calibration rows, 27 features, five assets, and data through 2026-07-23.

- LightGBM saved `models/vol7d.joblib` and its ignored split plot.
- XGBoost saved `models/vol7d_xgb.joblib` and its ignored split plot.

These rebuildable model/plot files remain ignored by repository policy and were
not staged.

### Full tree OOS run

The full tree run completed across 1,445 origins and persisted 10,115 rows for
each of XGBoost, LightGBM, `vol_21d`, and `tree_blend`.

| model | pinball | pinball % | coverage % |
|---|---:|---:|---:|
| XGBoost | 166.742851 | 1.403109 | 78.833416 |
| tree_blend | 166.842288 | 1.403945 | 78.695007 |
| LightGBM | 167.689757 | 1.411077 | 78.843302 |
| vol_21d | 167.993178 | 1.413630 | 79.317845 |

Ensemble gate:

| model | overall pinball % | worst-fold pinball % |
|---|---:|---:|
| XGBoost | 1.403109 | 2.228898 |
| vol_21d | 1.413630 | 2.202388 |
| tree_blend | 1.403945 | 2.231679 |

Decision: reject an ensemble-improvement claim. `tree_blend` is worse than
XGBoost overall and worse than both comparators in the worst fold. Its OOS rows
remain persisted for auditability; it is omitted from public headline tables.

### Full neural OOS run

The repository's CPU-only neural path completed in about 35 minutes across the
same 1,445 origins and persisted 10,115 rows for each model.

| model | pinball | pinball % | coverage % |
|---|---:|---:|---:|
| LSTM | 162.978783 | 1.371435 | 79.861592 |
| DLinear | 163.279483 | 1.373965 | 79.752842 |

### Final report bundle

`docs/evaluation/daily-20260723` contains exactly:

- Six PNGs: forecast bands, horizon coverage, model pinball, volatility fit,
  asset performance, and regime performance.
- Five CSVs: overall, by horizon, by asset, by fold, and by regime.
- One `manifest.json`: repository-relative paths and SHA-256 hashes for all
  eleven generated files and both input prediction Parquets, plus hashes and
  embedded configuration for their metadata files.

All six PNGs were visually inspected after ACL-safe thumbnail rendering. Titles,
axes, legends, the 1,445-origin sample size, and the 2021-01-01 to 2026-07-14
OOS origin range render legibly.

The combined tree/neural input contains 60,690 validated current-schema OOS
rows, six models, seven horizons, six folds, and 1,445 unique asset/origin pairs.

## Documentation/public metrics

Created `docs/evaluation-methodology.md` with the exact cutoff, asset/sample
provenance, fit/calibration/test boundaries for all six folds, model
configurations, commands, exploratory candidate comparisons, and caveats.

`README.md` and `RESULTS.md` were updated only from
`metrics_overall.csv`. Public rounded values are:

| model | pinball | coverage % |
|---|---:|---:|
| LSTM | 162.98 | 79.9 |
| DLinear | 163.28 | 79.8 |
| XGBoost | 166.74 | 78.8 |
| LightGBM | 167.69 | 78.8 |
| vol_21d baseline (RESULTS only) | 167.99 | 79.3 |

No `tree_blend` improvement was published. The 15-minute section was left
unchanged.

## Verification

Baseline before Task 7 edit:

```text
python -m unittest discover -s tests -v
Ran 26 tests ... OK

python -m compileall -q crypto tree linear neural experiments fetch.py predict.py evaluate.py
exit 0
```

Fresh final verification on the finished tree:

```text
python -m unittest discover -s tests -v
Ran 27 tests in 4.473s
OK

python -m compileall -q crypto tree linear neural experiments fetch.py predict.py evaluate.py
exit 0
```

Final artifact/reference validator:

```text
PASS: 6 PNGs, 5 CSVs, 60690 validated OOS rows, 15 local links,
public metrics match generated CSV
```

`git diff --check` exited 0. The line-ending messages are Git's existing
Windows LF-to-CRLF warnings, not whitespace errors.

## Task 7 files

Code/tests:

- `experiments/feature_ablation.py`
- `tests/test_feature_ablation.py`

Public documentation/evidence:

- `README.md`
- `RESULTS.md`
- `docs/evaluation-methodology.md`
- `docs/evaluation/daily-20260723/` (six PNGs, five metric CSVs, manifest)
- this report

Raw reproducibility evidence:

- `artifacts/feature_ablation.csv`
- `artifacts/evaluation/daily-tree-20260723-baseline/`
- `artifacts/evaluation/daily-neural-20260723-baseline/`

## Concerns

- Feature and ensemble comparisons reused the reported OOS folds; they are
  exploratory selection-set evidence, not an untouched post-selection holdout.
  A locked configuration needs future chronological data after 2026-07-23.
- `without_funding` was best in the exploratory group removals and warrants a
  separately preregistered future test.
- Overall coverage is inside the target band, but subgroup coverage varies; the
  five CSV tables expose that instability.
- The evidence is historical through 2026-07-23 and cannot support a claim about
  current/live forecast quality.
- The ensemble failed its acceptance gate and is retained only as negative
  evidence.

## Round 1/5 review fixes

Review fixes were implemented on top of Task 7 commit `6f84f95`.

### Publication posture

- `experiments/feature_ablation.py` now labels every ablation row as
  exploratory selection-set evidence.
- `README.md`, `RESULTS.md`, and `docs/evaluation-methodology.md` remove the
  feature-retention/deployability and causal sequence/data-ceiling claims.
- The public 2.3% statement is now only the measured aggregate LSTM-versus-XGBoost
  difference on this sample, with no causal or post-selection-generalization claim.
- The volatility headline now says aggregate losses were modestly lower while
  by-horizon comparisons were mixed.
- The methodology states that future chronological data strictly after
  2026-07-23 is required to validate a locked configuration, and that rerunning
  the same folds does not repair the selection-set limitation.

### Provenance and non-overwriting runners

- `evaluate.py` now requires each prediction Parquet's sibling `metadata.json`
  before creating the output directory.
- It writes `manifest.json` with SHA-256 for five CSVs, six PNGs, two input
  prediction Parquets, and both metadata files. The manifest records repository-
  relative paths, embedded run configuration, cutoff, OOS start/end, six fold
  origins, 1,445 distinct asset/origins, 60,690 rows, six models, and horizons 1-7.
- `tree.run` and `neural.run` accept `--run-id` and `--output-root`. If no run ID
  is supplied, a 12-character UUID suffix is generated. Explicit collisions fail
  after the cutoff/feature load and before fold fitting or neural window creation.
- The evaluator remains non-overwriting, and the methodology gives UUID-based
  commands for collision-free reruns without deleting prior work.

### TDD evidence

RED commands were run before their implementations:

```text
python -m unittest tests.test_evaluation_plots.EvaluationPlotTests.test_cli_writes_hashed_manifest_and_never_overwrites_output -v
FAIL: manifest.json did not exist

python -m unittest tests.test_evaluation.PredictionValidationTests.test_new_run_directory_is_unique_and_rejects_collisions -v
FAIL: crypto.evaluation had no new_run_dir

python -m unittest tests.test_evaluation.PredictionValidationTests.test_runners_reject_output_collisions_before_expensive_work -v
FAIL: runner backtest signatures lacked run_id/output_root

python -m unittest tests.test_evaluation.PredictionValidationTests.test_runner_clis_accept_run_id_and_output_root -v
FAIL: runner modules had no parse_args
```

Focused GREEN:

```text
C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe -m unittest tests.test_evaluation tests.test_evaluation_plots tests.test_ensemble -v
Ran 24 tests in 3.795s
OK
```

### Bundle regeneration and integrity

The published directory was moved to an exact sibling backup, regenerated through
the non-overwriting evaluator at its canonical path, cryptographically validated,
and only then was that temporary backup removed. The evaluator command was:

```powershell
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' evaluate.py artifacts/evaluation/daily-tree-20260723-baseline/predictions.parquet artifacts/evaluation/daily-neural-20260723-baseline/predictions.parquet --out docs/evaluation/daily-20260723
```

Observed regeneration/validation summary:

```text
inputs=2 outputs=11 rows=60690 origins=1445 folds=6
data_cutoff=2026-07-23T00:00:00+00:00
oos=2021-01-01T00:00:00+00:00..2026-07-14T00:00:00+00:00
models=dlinear,lgbm,lstm,tree_blend,vol_21d,xgb
all recorded SHA-256 hashes matched
```

### Round 1 verification

```text
C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe -m unittest discover -s tests -v
Ran 30 tests in 3.085s
OK

C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe -m compileall -q crypto tree linear neural experiments fetch.py predict.py evaluate.py
exit 0

artifact integrity validator
PASS: 2 inputs, 11 outputs, 60,690 rows, 1,445 origins; every content hash and metadata hash matched

documentation reference validator
PASS: 17 local references

published bundle validator
PASS: exactly 5 metric CSVs, 6 PNGs, and manifest.json

unsupported-claim scan
PASS: targeted false headline, causal-attribution, retention, and deployability patterns are absent
```

`git diff --check` exited 0 apart from Git's Windows line-ending warnings.

## Round 2/5 review fixes

Review fixes were implemented on top of `927e97f`.

### Portable provenance root

- `evaluate.py` accepts `--provenance-root`, defaulting to the current invocation
  directory. The canonical command declares the repository root with
  `--provenance-root .`.
- Prediction, metadata, output-table, and chart paths are resolved, required to
  remain inside that root, and serialized as normalized root-relative POSIX paths.
- Inputs or outputs outside the root fail with a clear `ValueError` before the
  evaluator reads prediction data or creates its output directory.
- The absolute fallback fields were removed. Absolute path strings at any nesting
  depth in embedded metadata are normalized through the same root contract, so the
  manifest has no drive-specific values.
- The manifest declares `{"root": ".", "path_format":
  "provenance-root-relative-posix"}`.

### Atomic run reservation

- `reserve_run_dir` now performs the reservation with
  `mkdir(parents=True, exist_ok=False)`. This atomically creates the leaf run
  directory, so a same-ID second reservation fails immediately.
- Tree and neural runners reserve after the cutoff/feature load and before fold
  fitting or neural window construction.
- `save_predictions(..., reserved=True)` consumes only an existing empty
  reservation. It refuses missing, populated, or already-consumed reservations.
  Its ordinary mode still refuses any pre-existing directory.
- A training or write failure intentionally leaves its empty or partial reservation
  in place. The methodology instructs operators to inspect it and retry with a new
  run ID rather than reusing the failed reservation.

### TDD evidence

Atomic reservation RED:

```text
C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe -m unittest tests.test_evaluation.PredictionValidationTests.test_reserve_run_directory_is_atomic_and_rejects_collisions tests.test_evaluation.PredictionValidationTests.test_save_consumes_reserved_empty_directory_once tests.test_evaluation.PredictionValidationTests.test_save_refuses_populated_reserved_directory tests.test_evaluation.PredictionValidationTests.test_runners_leave_atomic_reservation_before_expensive_work -v
FAILED (failures=3, errors=2)
- reserve_run_dir did not exist
- save_predictions rejected the reserved keyword
- runner destinations did not exist when expensive work began
```

Atomic reservation GREEN:

```text
Ran 4 tests in 5.020s
OK
```

Provenance root RED:

```text
C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe -m unittest tests.test_evaluation_plots.EvaluationPlotTests.test_cli_writes_hashed_manifest_and_never_overwrites_output tests.test_evaluation_plots.EvaluationPlotTests.test_cli_rejects_input_outside_provenance_root tests.test_evaluation_plots.EvaluationPlotTests.test_cli_rejects_output_outside_provenance_root -v
FAILED (errors=3)
error: unrecognized arguments: --provenance-root
```

Provenance root GREEN:

```text
Ran 3 tests in 2.649s
OK
```

A nested absolute `cache_path` fixture then proved that normalizing only the
top-level `output_dir` was insufficient:

```text
C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe -m unittest tests.test_evaluation_plots.EvaluationPlotTests.test_cli_writes_hashed_manifest_and_never_overwrites_output -v
FAIL: drive-specific C:\ value remained in nested metadata

after recursive metadata path normalization:
Ran 1 test in 1.260s
OK
```

The first combined run found one test-boundary regression: the ensemble test's
save double still accepted only the old three-argument signature. The production
runner correctly passed `reserved=True`. The test double was updated to mirror the
real boundary and its run output was moved under a temporary directory.

```text
C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe -m unittest tests.test_ensemble.EnsembleTests.test_tree_backtest_selects_on_calibration_and_persists_only_test_blend -v
Ran 1 test in 1.331s
OK

C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe -m unittest tests.test_evaluation tests.test_evaluation_plots tests.test_ensemble -v
Ran 29 tests in 5.729s
OK
```

### Regeneration and verification

The canonical bundle was regenerated with:

```powershell
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' evaluate.py artifacts/evaluation/daily-tree-20260723-baseline/predictions.parquet artifacts/evaluation/daily-neural-20260723-baseline/predictions.parquet --out docs/evaluation/daily-20260723 --provenance-root .
```

Observed:

```text
provenance=. format=provenance-root-relative-posix
inputs=2 outputs=11 rows=60690 origins=1445
absolute_path_fields=0 drive_specific_values=0 hashes=matched
```

Final Round 2 gates:

```text
C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe -m unittest tests.test_evaluation tests.test_evaluation_plots tests.test_ensemble -v
Ran 29 tests in 5.759s
OK

C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe -m unittest discover -s tests -v
Ran 35 tests in 5.593s
OK

C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe -m compileall -q crypto tree linear neural experiments fetch.py predict.py evaluate.py
exit 0

artifact portability/integrity validator
PASS: provenance declared; 0 absolute fields; 0 drive values; 13 artifact hashes and 2 metadata hashes matched

documentation reference validator
PASS: 17 local references

git diff --check
exit 0 (Windows line-ending warnings only)
```
