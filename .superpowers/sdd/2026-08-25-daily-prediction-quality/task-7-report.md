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

## Raw result summaries and decisions

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

Binding decision: retain `all` because it beats `legacy` on both overall
normalized pinball (1.472134 < 1.475972) and worst-fold normalized pinball
(2.228898 < 2.238526), while combined OOS coverage is inside 78-82%.

The parent ruling confirmed that the coverage gate uses combined OOS coverage,
while fold/asset/horizon coverage remains published. `without_funding` is an
exploratory future candidate, not a post-hoc replacement after searching these
same folds.

### Saved tree rebuilds

Both retained-feature rebuilds completed on 9,182 fit rows plus 2,300
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

All six PNGs were visually inspected after ACL-safe thumbnail rendering. Titles,
axes, legends, the 1,445-origin sample size, and the 2021-01-01 to 2026-07-14
OOS origin range render legibly.

The combined tree/neural input contains 60,690 validated current-schema OOS
rows, six models, seven horizons, six folds, and 1,445 unique asset/origin pairs.

## Documentation/public metrics

Created `docs/evaluation-methodology.md` with the exact cutoff, asset/sample
provenance, fit/calibration/test boundaries for all six folds, model
configurations, commands, acceptance decisions, and caveats.

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
- `docs/evaluation/daily-20260723/` (six PNGs, five metric CSVs)
- this report

Raw reproducibility evidence:

- `artifacts/feature_ablation.csv`
- `artifacts/evaluation/daily-tree-20260723-baseline/`
- `artifacts/evaluation/daily-neural-20260723-baseline/`

## Concerns

- Feature acceptance reused the reported OOS folds; there is no untouched
  post-selection holdout. The small `all` versus `legacy` gain may be optimistic.
- `without_funding` was best in the exploratory group removals and warrants a
  separately preregistered future test.
- Overall coverage is inside the target band, but subgroup coverage varies; the
  five CSV tables expose that instability.
- The evidence is historical through 2026-07-23 and cannot support a claim about
  current/live forecast quality.
- The ensemble failed its acceptance gate and is retained only as negative
  evidence.
