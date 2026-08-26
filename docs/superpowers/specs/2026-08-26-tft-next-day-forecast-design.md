# TFT Next-Day Forecasting Design

## Objective

Build a reproducible Temporal Fusion Transformer (TFT) research pipeline that
uses 15-minute cryptocurrency data to forecast the next 24 hours. The model
will produce an intraday price path and q10, q50, and q90 price forecasts. The
96th forecast step is the headline next-day prediction.

The user will run the full GPU training. This change prepares and verifies the
data, model, evaluation, artifact, and reporting code and provides the exact
training command.

## Scope

The work includes:

- deterministic cleaning and causal feature construction for 15-minute data;
- a seven-day input window and a 96-bar forecast horizon;
- a single global TFT shared across BTC, ETH, BNB, SOL, and XRP;
- fixed-model chronological calibration and testing;
- corrected one-sided conformal calibration;
- forecast, return, direction, and uncertainty diagnostics;
- versioned predictions, model checkpoints, metadata, metrics, and graphs;
- automated tests that do not require full GPU training.

## Non-goals

This phase will not:

- generate BUY, SELL, or HOLD decisions;
- simulate trades, positions, fees, slippage, P&L, or portfolio returns;
- place paper or live orders;
- schedule periodic retraining;
- claim profitable or tradable performance;
- run the full TFT training on the user's behalf.

## Data contract

The source is `data/ohlcv_15m.parquet`. Required columns are `asset`, `date`,
`open`, `high`, `low`, `close`, and `volume`. The supported universe is BTC,
ETH, BNB, SOL, and XRP. Timestamps are UTC candle-open timestamps; a bar's close
is available only after the 15-minute candle finishes.

Before features are built, the pipeline must reject:

- duplicate `(asset, date)` rows;
- unsorted or timezone-naive timestamps;
- non-positive OHLC prices;
- negative volume;
- `high < low` or OHLC values outside the candle range;
- incomplete final candles;
- missing required assets or insufficient history.

The current local dataset contains 977,630 rows from 2021-01-01 through
2026-07-31. It has seven exchange-wide gap events per asset, totaling 70
missing bars per asset; the longest observed gap is 18 bars. These measurements
describe the current file and are not hard-coded acceptance criteria.

## Missing-bar handling

Each asset is reindexed to a complete 15-minute UTC grid. Missing bars are
represented causally:

- `open`, `high`, `low`, and `close` use the most recent observed close;
- `volume` is set to zero;
- `missing_bar` is set to one, and zero for observed bars.

No value may be backfilled from a future bar. Evaluation origins are excluded
when any of their following 96 target bars is synthetic. Input windows may
contain synthetic bars because the indicator makes the outage observable to
the model. Gap counts, durations, and excluded origins are stored in run
metadata.

## Causal features

Historical inputs are computed after grid completion and use only the current
or earlier completed bars:

- `log_return`: `log(close).diff()`;
- `abs_return`: absolute log return;
- `range_pct`: `(high - low) / close`;
- `log_volume`: `log1p(volume)`;
- `log_volume_change`: first difference of `log1p(volume)`;
- `rv_96`: square root of the trailing sum of squared log returns over 96 bars;
- `rv_672`: square root of the trailing sum of squared log returns over 672 bars;
- `momentum_96`: `log(close / close.shift(96))`;
- `momentum_672`: `log(close / close.shift(672))`;
- `volume_z96`: trailing 96-bar z-score of `log1p(volume)`;
- `missing_bar`: observed/synthetic indicator.

Rolling calculations include the current completed bar but never future bars.
Zero rolling standard deviation produces a zero z-score rather than infinity or
NaN. Rows without the complete 672-bar history needed by the model are removed.

Known-future calendar inputs are deterministic from the forecast timestamp:

- sine and cosine of time of day;
- sine and cosine of day of week.

## Forecast target and time alignment

The model target is log closing price. At a 00:00 UTC forecast origin, the last
available observation is the close of the candle labeled 23:45 UTC. The model
forecasts the 96 candles labeled from 00:00 through 23:45 UTC. The final candle
closes at the following 00:00 UTC, exactly 24 hours after the origin.

The q10, q50, and q90 log-price predictions are exponentiated into positive
price forecasts. Next-day return quantiles are then calculated relative to the
last completed close:

`return_q = forecast_price_q / origin_close - 1`

Because this transformation is monotonic, it preserves quantile ordering. The
pipeline reports the full 96-step path but uses step 96 for headline next-day
metrics.

## TFT configuration

The initial fixed configuration is:

- frequency: 15 minutes;
- input size: 672 bars (seven days);
- horizon: 96 bars (24 hours);
- quantiles: 0.1, 0.5, and 0.9;
- loss: multi-quantile pinball loss;
- scaler: robust per-series scaling;
- hidden size: 128;
- maximum training steps: 4,000;
- random seed: 42;
- historical exogenous inputs: the causal feature set above;
- future exogenous inputs: calendar features above.

The model is global: weights are shared across all assets while each asset
remains a distinct time series. Hyperparameters are fixed before final testing
and cannot be tuned on conformal-calibration or test outcomes. Batch size and
accelerator selection may be exposed as operational CLI options without
changing forecast semantics.

## Chronological evaluation protocol

The experiment simulates a model trained once and then deployed without
retraining:

1. Reserve the final 365 daily origins as the evaluation year.
2. Fit one TFT using only earlier observations. A 28-day slice immediately
   before the evaluation year is reserved for training diagnostics and any
   supported early-stopping behavior.
3. Keep the fitted model frozen for all 365 evaluation origins. The
   NeuralForecast cross-validation call therefore uses `refit=False`
   intentionally and documents the run as fixed-model chronological
   evaluation, not expanding-window validation.
4. Use the first 219 origins (60%) only for interval calibration.
5. Use the final 146 origins (40%) as the untouched test set.
6. Use a step size of 96 bars so forecast origins occur once per day and the
   headline 24-hour targets do not overlap.

All five assets must share the same origin grid. The pipeline fails if an asset
is missing an origin, if truth differs across model comparisons, or if fewer
than 100 valid test origins remain after excluding missing-target windows.

## Conformal interval calibration

Calibration is performed in log-price space, separately for each forecast
horizon. Lower and upper tails are handled independently:

- lower nonconformity: `q10 - actual`;
- upper nonconformity: `actual - q90`.

For each horizon, the calibration-only 90th percentile of each tail score is
computed using a finite-sample conformal quantile. Each adjustment is clipped
at zero so calibration may widen but never tighten a native interval:

- calibrated q10 = raw q10 minus the lower adjustment;
- calibrated q90 = raw q90 plus the upper adjustment.

The q50 forecast is unchanged. Test outcomes never affect these adjustments.
The system stores raw and calibrated predictions so the effect of calibration
can be audited.

## Baselines and metrics

The TFT is compared on identical test keys and truth against two fully
probabilistic baselines. Both estimate 15-minute volatility as the standard
deviation of log returns over the trailing seven days and scale it by the square
root of forecast horizon. The fixed normal-reference value for q10/q90 is
`1.2815515655`; this is a simple benchmark assumption, not a distributional
claim about crypto returns.

- persistence-volatility: q50 remains at the origin log price for every
  horizon, while q10/q90 equal q50 plus or minus the scaled volatility band;
- momentum-volatility: q50 adds `(h / 96)` times the prior 24-hour log return
  to the origin log price, while q10/q90 use the same volatility band around
  that projected median.

All baseline log-price forecasts are exponentiated before price-space metrics
are calculated. The same origin close, horizon, truth, asset, and split keys are
required for TFT and both baselines.

Headline step-96 metrics are:

- row-normalized pinball percentage across q10, q50, and q90;
- MAPE of the q50 price forecast;
- empirical q10-q90 interval coverage, targeting 80%;
- normalized interval width `(q90 - q10) / origin_close`;
- next-day directional accuracy with a 95% confidence interval;
- next-day return MAE;
- R-squared on next-day returns, never on price levels;
- predicted-versus-realized return correlation.

Directional confidence intervals resample complete daily origins rather than
individual asset rows so correlated assets are not treated as independent.
Metrics are also broken down by asset, forecast horizon, evaluation segment,
and a causal volatility regime derived from trailing realized volatility.

No performance claim is allowed solely because MAPE is low. A positive TFT
claim requires improvement over the predefined baselines on untouched test
data, with sample size, date range, and subgroup behavior disclosed.

## Reports and graphs

Each completed run produces:

- the last-origin 96-step forecast path for every asset with q10-q90 bands;
- next-day actual versus predicted prices over the test period;
- predicted versus realized next-day returns;
- coverage and normalized pinball loss by asset;
- performance by causal volatility regime;
- raw-versus-calibrated interval diagnostics;
- a compact overall model-and-baseline metrics table.

Graphs use only out-of-sample calibration or test predictions, label which
split is shown, and include the date range and number of origins. Headline
graphs use only the untouched test split.

## Versioned artifacts and provenance

Runs are written atomically beneath a caller-selected output root using a safe,
unique run identifier. Existing output directories are never overwritten. A
completed run contains:

- the serialized TFT checkpoint needed for later inference;
- raw cross-validation predictions;
- calibrated and uncalibrated test predictions;
- overall and subgroup metric tables;
- generated graphs;
- metadata and a content-hash manifest.

Metadata includes the run identifier, pipeline name, model class, complete
configuration, random seed, source-data path, data cutoff and range, asset and
row counts, gap statistics, training/validation/calibration/test boundaries,
feature names, package versions, Git commit, elapsed training time, and device
information. Paths stored in committed reports must be portable and must not
expose machine-specific absolute paths.

Raw checkpoints and large prediction files remain local unless deliberately
published. Small report tables, graphs, and their manifest may be committed.

## Command-line contract

The training runner will accept explicit options for at least:

- `--model tft`;
- `--run-id`;
- `--output-root`;
- `--accelerator`;
- `--batch-size`.

It validates and reserves the output directory before loading the full dataset
or starting training. The final handoff will include one PowerShell command
that uses the repository virtual environment and current worktree. Running that
command is the user's responsibility.

## Failure handling

The pipeline fails before publishing a report when it encounters:

- invalid or incomplete input data;
- non-causal feature behavior;
- insufficient training, calibration, or test history;
- target windows crossing missing bars;
- missing or duplicated forecast keys;
- NaN or infinite predictions;
- crossed quantiles;
- mismatched truth or origins between TFT and baselines;
- calibration using test rows;
- unsafe run identifiers or output collisions;
- incomplete metadata or manifest hashes.

A failed training run may preserve diagnostic logs in its reserved local run
directory but cannot be marked complete or used for reported metrics.

## Automated verification

Unit and fixture-level integration tests will verify:

- input validation and deterministic gap completion;
- exact feature formulas and causal behavior under future-data corruption;
- seven-day lookback and 96-bar target alignment;
- 00:00 UTC origin selection and last-completed-bar semantics;
- exclusion of targets that cross synthetic bars;
- fixed-model train/validation/calibration/test separation;
- separate lower- and upper-tail conformal adjustments;
- metric and confidence-interval calculations;
- baseline alignment and comparability checks;
- quantile ordering and absence of invalid predictions;
- deterministic model construction;
- safe, atomic artifact creation and manifest verification;
- report creation from small saved prediction fixtures.

Tests must run without full GPU training. A lightweight model-construction
smoke test may instantiate TFT but must not execute the complete experiment.

## Acceptance conditions

Implementation is complete when:

- the existing repository test suite and all new TFT tests pass;
- the feature causality corruption test passes for every asset;
- a small fixture run produces valid versioned artifacts and all required
  graphs;
- the runner exposes the documented user-operated training command;
- no full TFT training has been performed by Codex;
- no trading signals, trading backtest, or execution code has been added.

Forecast performance is a later empirical result, not an implementation
acceptance condition. After the user runs training, the resulting artifacts
must pass validation before their numbers are documented or used on a resume.
