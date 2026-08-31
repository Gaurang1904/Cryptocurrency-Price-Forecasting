# Task 6 Report — TFT next-day report

## Scope

Created `neural/tft_plots.py` and `tests/test_tft_plots.py`.  The report
function accepts calibration rows plus untouched test rows, rejects a mixed or
non-test headline split and absent required baselines, writes the four TFT
metric tables, and returns exactly the six requested PNG diagnostics.

The renderer configures Matplotlib's noninteractive `Agg` backend, uses only
TFT evaluation data (it does not import the daily plotting module), writes
deterministic static files, and closes each figure in a `finally` block plus a
top-level cleanup guard.  Paths and comparison charts use explicit line
styles, markers, labels, reference/identity lines, and readable table CSVs.

## RED evidence

After adding the fixture-report tests and before adding the module:

```text
python -m unittest tests.test_tft_plots -v
ModuleNotFoundError: No module named 'neural.tft_plots'
FAILED (errors=1)
```

This was the expected missing-module failure.

## GREEN evidence

```text
python -m unittest tests.test_tft_plots -v
Ran 2 tests in 13.842s
OK
```

The tests assert the exact four CSV and six PNG filenames, closed Matplotlib
figures, test-only headline enforcement, and both TFT baselines.

## Focused plot verification

```text
python -m unittest tests.test_tft_plots tests.test_evaluation_plots -v
Ran 8 tests in 17.067s
OK
```

## Full-suite verification

`python -m unittest discover -s tests -v` exited 0 after executing the
repository test suite. `python -m unittest` and default discovery at the
repository root report zero tests because this project keeps tests beneath the
`tests` directory; the `-s tests` invocation is the effective full-suite
command.

The TFT fixture yields NumPy runtime warnings from the pre-existing
correlation calculation when a synthetic return series is constant. They do
not fail tests; this task leaves metric semantics unchanged.

## Self-review

- `REPORT_RENDERERS` is in the asserted six-PNG order and each helper saves
  once using `bbox_inches="tight"` and closes in `finally`.
- Forecast paths use the latest test origin, all 96 15-minute steps, all
  assets, calibrated q10–q90 bands, and observed prices.
- Next-day plots, returns scatter, asset/regime performance, and raw-vs-
  calibrated coverage all use test rows and include the OOS caption with the
  distinct daily-origin count.
- No training or trading paths were modified.

## Pre-review test-quality cleanup

The h=96 fixture rows originally produced constant actual/predicted returns for some models, causing NumPy constant-correlation `RuntimeWarning` output.  `tests/test_tft_plots.py` now applies a deterministic 0.01% step variation by origin and asset to h=96 actual/quantile values, and applies the same variation to baseline quantiles after baseline construction.  Model names, row keys, 96 horizons, calibration/test split labels, and report graph/table semantics are unchanged.  Production code and metric definitions were not modified.

Baseline focused run (before edit):

```text
python -W default -m unittest tests.test_tft_plots tests.test_evaluation_plots -v
```

`test_report_creates_exact_tables_and_graphs_and_closes_figures` passed but emitted repeated NumPy `RuntimeWarning: invalid value encountered in divide` messages from `numpy/lib/_function_base_impl.py:3036-3037` during constant h=96 correlations.

Focused verification after edit:

```text
python -W error -m unittest tests.test_tft_plots tests.test_evaluation_plots -v
Ran 8 tests in 33.401s
OK
```

Full-suite verification after edit:

```text
python -W error -m unittest discover -s tests -v
Ran 95 tests in 30.753s
OK
```

Both suites were warning-free with warnings promoted to errors.  Self-review found only the intended fixture-data change in `tests/test_tft_plots.py`; no production files changed.
