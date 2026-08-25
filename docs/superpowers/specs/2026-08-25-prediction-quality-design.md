# Prediction Quality and Portfolio Evaluation Design

## Objective

Improve the credibility and presentation of the forecasting project without
optimizing against the test set. The daily 1–7 day volatility pipeline remains
the primary project; high-frequency and directional forecasts remain secondary
research tracks until their validation is strong enough.

The primary acceptance criterion is lower walk-forward pinball loss than the
21-day volatility baseline while maintaining 78–82% q10–q90 coverage across
assets and horizons. Improvements must appear across multiple folds, not only in
one aggregate score.

## Evaluation boundaries

Each walk-forward fold has three chronological partitions:

1. Fit data trains model parameters.
2. Calibration data selects interval calibration and any ensemble weights.
3. Test data is evaluated once and is never used for feature, model, or chart selection.

Daily, hourly, and 15-minute results remain separate. Every reported metric must
identify pipeline, model, dataset cutoff, forecast horizon, fold count, origin
count, and configuration.

## Daily pipeline

The existing daily feature and model interfaces remain intact. Candidate work is
limited to measurable additions:

- data-quality indicators for stale or imputed inputs;
- market-regime features derived only from historical volatility and trend;
- feature-group ablations for returns, volatility, range/volume, BTC, and funding;
- an ensemble of existing daily models, with weights selected on calibration data;
- calibration tuning performed independently for each horizon.

One candidate is changed at a time. It is retained only when it improves mean
pinball loss and does not materially degrade coverage, per-asset behavior, or
worst-fold performance.

## High-frequency pipeline

Before model tuning, add an explicit missing-bar indicator and define deterministic
handling for prices and volume on exchange gaps. Establish last-price and simple
return/volatility baselines, then increase the number and regime coverage of test
origins. Only after those gates pass should the current 4,000-step models be rerun.

High-frequency metrics stay out of the headline resume result until they cover at
least 100 test origins spanning multiple market regimes and beat a naïve baseline.

## Directional research gate

Direction is evaluated separately at each horizon using balanced accuracy and a
confidence interval, plus a simple strategy simulation with fees and slippage.
Research stops if performance does not beat chance consistently across folds,
assets, and regimes. A negative finding remains a valid project result.

## Reproducible evaluation artifact

A single evaluation entry point will read saved out-of-sample predictions and
produce:

- a versioned metrics table;
- actual price with q10–q90 forecast bands;
- coverage by horizon;
- normalized pinball loss by model;
- predicted versus realized volatility;
- performance by asset and market regime.

Charts use only out-of-sample predictions, label sample size and date range, and
show baselines alongside learned models. Generated artifacts are written to one
documented output directory and do not silently overwrite historical runs.

## Verification and failure handling

Automated checks cover feature causality, chronological split ordering, duplicate
forecast keys, quantile ordering, metric calculations, and chart generation from a
small fixture. Evaluation fails loudly on missing artifacts, stale configuration,
NaN predictions, crossed quantiles, or mismatched horizons.

## Stop conditions

The daily track stops when one reproducible candidate beats the baseline and current
best model under the primary acceptance criterion, or when three well-motivated
candidates fail to improve worst-fold performance. The project will not add more
architectures or features after that point without a specific hypothesis supported
by error analysis.
