# Daily evaluation methodology

## Scope

This report is a **historical out-of-sample measurement through the 2026-07-23
data cutoff**. It is not a current or live forecast, a trading signal, or a P&L
backtest. The latest evaluated forecast origin is 2026-07-14. Because these folds
were consulted for feature and ensemble choices, they are also the selection set:
the comparisons are exploratory, not untouched post-selection validation. A locked
configuration needs future chronological data strictly after 2026-07-23. Rerunning
these folds does not turn them into an untouched test.

The generated evidence is in
[`docs/evaluation/daily-20260723`](evaluation/daily-20260723/). Its five CSV
tables are the source of every daily metric published in `README.md` and
`RESULTS.md`; the six PNGs are views of those same persisted OOS predictions.
[`manifest.json`](evaluation/daily-20260723/manifest.json) records the sample and
configuration metadata and SHA-256 hashes of all eleven outputs plus both input
prediction Parquets and their metadata files. Every recorded path is normalized
relative to the declared repository provenance root; inputs or outputs outside
that root are rejected rather than encoded with machine-specific fallbacks.
Absolute path strings nested inside embedded input metadata follow the same
normalization/rejection rule.

## Data and sample

- Inputs: `data/ohlcv.parquet` plus historical funding joined by
  `crypto.features.build`.
- Assets: BTC, ETH, BNB, SOL, and XRP.
- Raw OHLCV sample: 14,884 daily rows. BTC and ETH begin 2017-08-17, BNB
  2017-11-06, XRP 2018-05-04, and SOL 2020-08-11. Every asset ends at the
  2026-07-23 cutoff.
- Forecasts: horizons 1 through 7 days, with origins every 7 days after at least
  60 bars of history.
- OOS sample: 1,445 unique `(asset, origin)` pairs (289 per asset), from
  2021-01-01 through 2026-07-14.
- Persisted sample: 10,115 rows per model and 60,690 rows total across LSTM,
  DLinear, XGBoost, LightGBM, `vol_21d`, and `tree_blend`.
- Quantiles: q10, q50, and q90; q10-q90 coverage therefore targets 80%.

## Walk-forward boundaries

Each fold trains only on dates strictly before the fold start. The final
calendar fold is partial because the data cutoff is 2026-07-23. Inside every
training window, the last 20% by date quantile is held out for calibration:
fit rows have `date < cut`; calibration rows have `date >= cut`. The empirical
q10/q50/q90 standardized-return table is fitted only on calibration rows and
then fixed for that fold's test rows.

Tree rows require all 27 engineered features plus all price and realized-
volatility targets. Neural fold selection needs only the causal `vol_21d`
driver plus targets, so its eligible history begins earlier.

| fold | test origins (range) | tree fit rows (range) | tree calibration rows (range) |
|---|---:|---|---|
| 2021-01-01 | 261 (2021-01-01 to 2021-12-31) | 1,091 (2019-11-11 to 2020-11-04) | 276 (2020-11-05 to 2020-12-31) |
| 2022-01-01 | 261 (2022-01-01 to 2022-12-31) | 2,552 (2019-11-11 to 2021-08-25) | 640 (2021-08-26 to 2021-12-31) |
| 2023-01-01 | 260 (2023-01-02 to 2023-12-30) | 4,012 (2019-11-11 to 2022-06-13) | 1,005 (2022-06-14 to 2022-12-31) |
| 2024-01-01 | 263 (2024-01-01 to 2024-12-31) | 5,472 (2019-11-11 to 2023-04-01) | 1,370 (2023-04-02 to 2023-12-31) |
| 2025-01-01 | 260 (2025-01-03 to 2025-12-30) | 6,937 (2019-11-11 to 2024-01-19) | 1,735 (2024-01-20 to 2024-12-31) |
| 2026-01-01 | 140 (2026-01-02 to 2026-07-14) | 8,397 (2019-11-11 to 2024-11-06) | 2,100 (2024-11-07 to 2025-12-31) |

| fold | neural fit rows (range) | neural calibration rows (range) |
|---|---|---|
| 2021-01-01 | 3,703 (2017-09-07 to 2020-06-13) | 926 (2020-06-14 to 2020-12-31) |
| 2022-01-01 | 5,159 (2017-09-07 to 2021-04-16) | 1,295 (2021-04-17 to 2021-12-31) |
| 2023-01-01 | 6,619 (2017-09-07 to 2022-02-02) | 1,660 (2022-02-03 to 2022-12-31) |
| 2024-01-01 | 8,079 (2017-09-07 to 2022-11-21) | 2,025 (2022-11-22 to 2023-12-31) |
| 2025-01-01 | 9,544 (2017-09-07 to 2023-09-10) | 2,390 (2023-09-11 to 2024-12-31) |
| 2026-01-01 | 11,004 (2017-09-07 to 2024-06-28) | 2,755 (2024-06-29 to 2025-12-31) |

## Model configurations

All models predict log realized volatility. Predicted sigma is exponentiated,
clipped to `[0.0001, 0.30]`, and converted to price bands with the same
fold-specific empirical calibration.

- XGBoost: 300 estimators, learning rate 0.05, max depth 5, minimum child
  weight 50, row and column subsampling 0.8, histogram tree method.
- LightGBM: 300 estimators, learning rate 0.05, 31 leaves, minimum 50 child
  samples, row and column subsampling 0.8.
- Tree features (27): lagged and rolling returns, rolling volatility, ranges,
  price/MA, BTC market returns, funding features, and the evaluated
  `vol_regime`, `drawdown_63d`, and `volume_z21` candidate features. The exact
  names are persisted in the tree bundle's `metadata.json`.
- `vol_21d`: causal 21-day rolling-volatility baseline, calibrated through the
  same price-band path.
- `tree_blend`: per-fold/per-horizon convex blend of XGBoost and LightGBM.
  Weight is selected on calibration predictions only from 0.00 to 1.00 in
  0.05 steps, with ties closest to 0.50 and then the lower weight.
- Neural inputs: 30-day windows of log return, absolute log return,
  high-low range divided by close, and log-volume change. Channels are
  standardized from fit rows only.
- Neural optimization: Adam, learning rate 0.001, batch size 256, MSE loss,
  maximum 200 epochs, patience 15, seed 42. DLinear uses a 25-day moving-average
  decomposition and two linear heads. LSTM uses one 32-unit layer and 0.1
  dropout before the seven-horizon head.
- The neural implementation used its repository-default CPU path; it does not
  contain a CUDA device branch.

## Exploratory candidate comparisons

The comparisons below use folds that were inspected when choosing which
configuration to carry into artifact generation. They remain out of sample
relative to each fold's model fit and calibration, but they are a selection set
relative to configuration choice. They describe measured behavior on this sample;
they do not validate post-selection generalization or lock a configuration.

### Three-feature selection-set comparison

The recorded comparison was `all` versus `legacy`, where `legacy` excludes exactly
`vol_regime`, `drawdown_63d`, and `volume_z21`. Overall normalized pinball is the
origin-count-weighted mean of the six fold rows; worst-fold is the maximum fold
normalized pinball. Combined OOS coverage, not every thinner subgroup, is the
78-82% descriptive comparison band.

| candidate | overall pinball % | worst-fold pinball % | combined coverage % |
|---|---:|---:|---:|
| all | **1.472134** | **2.228898** | 78.833416 |
| legacy | 1.475972 | 2.238526 | 78.991597 |

On these selection folds, `all` measured lower overall and worst-fold loss while
combined coverage remained inside the band. That observation motivated carrying
`all` into the later same-sample tree run; it is not a confirmed feature benefit.
Per-fold coverage remains visible in `metrics_by_fold.csv`. A locked feature
configuration requires new chronological observations strictly after 2026-07-23;
repeating this comparison on the same folds cannot provide that validation.

The other one-group removals are exploratory diagnostics. `without_funding` had
the lowest measured ablation loss (1.459705% overall, 2.071185% worst fold,
79.663866% coverage). It was not substituted after searching these same folds,
and none of these comparisons supports a confirmed superiority claim.

### Tree ensemble selection-set comparison

The exploratory gate required the ensemble to measure below both XGBoost and
`vol_21d` on overall and worst-fold normalized pinball. It did not.

| model | overall pinball % | worst-fold pinball % | combined coverage % |
|---|---:|---:|---:|
| XGBoost | **1.403109** | 2.228898 | 78.833416 |
| `vol_21d` | 1.413630 | **2.202388** | 79.317845 |
| `tree_blend` | 1.403945 | 2.231679 | 78.695007 |

`tree_blend` remains in the OOS artifact for auditability. No improvement or
baseline-superiority claim is made from this selection-set comparison.

## Generated metrics

The exact unrounded values are in:

- [`metrics_overall.csv`](evaluation/daily-20260723/metrics_overall.csv)
- [`metrics_by_horizon.csv`](evaluation/daily-20260723/metrics_by_horizon.csv)
- [`metrics_by_asset.csv`](evaluation/daily-20260723/metrics_by_asset.csv)
- [`metrics_by_fold.csv`](evaluation/daily-20260723/metrics_by_fold.csv)
- [`metrics_by_regime.csv`](evaluation/daily-20260723/metrics_by_regime.csv)
- [`manifest.json`](evaluation/daily-20260723/manifest.json), which binds the
  output tables/charts and input prediction bundles to SHA-256 hashes and records
  cutoff, OOS dates, folds, origins, rows, models, horizons, embedded run metadata,
  and the provenance-root-relative POSIX path contract.

Regime labels are descriptive fold-local terciles of the persisted origin-time
`vol_21d` driver. They never enter fitting, calibration, feature selection, or
ensemble selection.

## Exact commands

Working directory:

```text
C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.worktrees\daily-prediction-quality
```

Python executable:

```text
C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe
```

Commands executed for this historical report:

```powershell
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' -m compileall -q crypto tree linear neural experiments fetch.py predict.py evaluate.py
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' -m experiments.feature_ablation
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' -m tree.lgbm
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' -m tree.xgb
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' -m tree.run
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' -m neural.run
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' evaluate.py artifacts/evaluation/daily-tree-20260723-baseline/predictions.parquet artifacts/evaluation/daily-neural-20260723-baseline/predictions.parquet --out docs/evaluation/daily-20260723 --provenance-root .
```

The evaluator rejects an existing destination and any input/output outside its
declared `--provenance-root`. Tree and neural runners atomically reserve their run
directory with `mkdir(exist_ok=False)` before fold fitting or neural window
construction. If `--run-id` is omitted, each runner generates a 12-character UUID
suffix. The following commands therefore rerun without deleting or overwriting
earlier work:

```powershell
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' -m tree.run --output-root artifacts/evaluation
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' -m neural.run --output-root artifacts/evaluation
$reportId = [guid]::NewGuid().ToString('N')
& 'C:\Users\gaura\VSCode\PredictionModel\Prediction-Model\.venv\Scripts\python.exe' evaluate.py <new-tree-predictions.parquet> <new-neural-predictions.parquet> --out "docs/evaluation/daily-20260723-$reportId" --provenance-root .
```

The angle-bracket paths are the two newly printed runner output paths. To use
explicit labels instead, pass a unique `--run-id` to each runner; a reused label
fails before expensive model work. If training fails after reservation, that
empty or partially populated directory intentionally remains reserved. Do not
reuse it: inspect it as failure evidence and retry with a new run ID.

## Caveats

- The daily numbers describe this fixed historical sample only. They do not
  establish current performance after 2026-07-23.
- Feature and ensemble comparisons consulted these OOS folds. They are exploratory
  selection-set evidence, not a final untouched holdout. No configuration is locked
  by this report; validation needs future chronological data after 2026-07-23.
- The five-asset universe is small and crypto regimes change. Asset, fold,
  horizon, and regime tables should be read alongside the overall average.
- Coverage is close to 80% overall but varies materially by fold and subgroup.
- Seven horizons from one origin share information; 10,115 rows per model are
  not 10,115 independent forecasting events.
- Pinball in price units is dominated by high-priced assets. `pinball_%`
  normalizes the loss by mean last price for cross-asset interpretation.
- The evaluation measures forecast calibration and accuracy, not tradability;
  it includes no fees, slippage, execution model, or portfolio returns.
