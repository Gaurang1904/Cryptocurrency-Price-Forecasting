# TFT Next-Day Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, reproducible TFT pipeline that consumes 15-minute crypto data and produces calibrated q10/q50/q90 forecasts for the next 96 bars, with the 24-hour endpoint as the headline prediction.

**Architecture:** Keep the existing two-hour NeuralForecast experiments intact and add a focused TFT path. Separate data preparation, forecast evaluation, artifact persistence, plotting, and orchestration so each can be tested without GPU training. The runner deliberately fits once before a 365-origin evaluation year, calibrates on its first 219 origins, and reports only the final 146 untouched origins.

**Tech Stack:** Python 3, pandas, NumPy, PyArrow, PyTorch, NeuralForecast 3.2.0, matplotlib, unittest.

**Spec:** `docs/superpowers/specs/2026-08-26-tft-next-day-forecast-design.md`

## Global Constraints

- Supported assets are exactly BTC, ETH, BNB, SOL, and XRP.
- Input frequency is 15 minutes; input size is 672 bars; forecast horizon is 96 bars.
- TFT quantiles are 0.1, 0.5, and 0.9; hidden size is 128; maximum steps are 4,000; random seed is 42.
- The model uses causal historical inputs and deterministic future calendar inputs only.
- Evaluation uses 365 daily origins with `step_size=96` and intentional `refit=False`.
- The first 219 origins are conformal calibration; the final 146 are untouched test origins; at least 100 valid test origins must remain.
- Missing OHLC bars use the previous observed close, missing volume is zero, and `missing_bar=1`; no future backfill is allowed.
- q10 and q90 are conformalized separately in log-price space and may widen but never tighten.
- Full TFT training is user-operated. Tests and fixture reports must not execute the 4,000-step experiment.
- Do not add trading signals, trading simulation, fees, slippage, P&L, order placement, or retraining schedules.
- Existing daily evaluation and existing LSTM/GRU/N-HiTS behavior must remain compatible.

---

### Task 1: Validate and complete the 15-minute OHLCV grid

**Files:**
- Create: `neural/tft_data.py`
- Create: `tests/test_tft_data.py`

**Interfaces:**
- Consumes: raw `pd.DataFrame` with `asset,date,open,high,low,close,volume`.
- Produces: `validate_ohlcv_15m(raw, assets, as_of) -> pd.DataFrame`, `complete_ohlcv_grid(raw, assets) -> tuple[pd.DataFrame, dict[str, GapStats]]`, and immutable `GapStats`.

- [ ] **Step 1: Write failing validation and gap-completion tests**

```python
# tests/test_tft_data.py
import unittest
import pandas as pd

from neural.tft_data import complete_ohlcv_grid, validate_ohlcv_15m


def tiny_raw():
    dates = pd.to_datetime([
        "2026-01-01 00:00Z", "2026-01-01 00:15Z", "2026-01-01 00:45Z",
    ])
    rows = []
    for asset, offset in (("BTC", 0.0), ("ETH", 100.0)):
        for i, date in enumerate(dates):
            close = 100.0 + offset + i
            rows.append({
                "asset": asset, "date": date, "open": close - 0.2,
                "high": close + 0.5, "low": close - 0.5,
                "close": close, "volume": 10.0 + i,
            })
    return pd.DataFrame(rows)


class TftDataContractTests(unittest.TestCase):
    def test_grid_completion_is_causal_and_marks_missing_bars(self):
        raw = validate_ohlcv_15m(
            tiny_raw(), assets=("BTC", "ETH"),
            as_of=pd.Timestamp("2026-01-01 01:00Z"),
        )
        grid, stats = complete_ohlcv_grid(raw, assets=("BTC", "ETH"))
        missing = grid[grid.date == pd.Timestamp("2026-01-01 00:30Z")]
        self.assertEqual(set(missing.missing_bar), {1})
        self.assertEqual(set(missing.volume), {0.0})
        self.assertEqual(missing.set_index("asset").loc["BTC", "close"], 101.0)
        self.assertEqual(stats["BTC"].synthetic_bars, 1)
        self.assertEqual(stats["BTC"].max_gap_bars, 1)

    def test_validation_rejects_duplicate_and_incomplete_bars(self):
        raw = tiny_raw()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_ohlcv_15m(
                pd.concat([raw, raw.iloc[[0]]]), assets=("BTC", "ETH"),
                as_of=pd.Timestamp("2026-01-01 01:00Z"),
            )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_ohlcv_15m(
                raw, assets=("BTC", "ETH"),
                as_of=pd.Timestamp("2026-01-01 00:50Z"),
            )

    def test_validation_rejects_invalid_ohlcv_and_missing_assets(self):
        raw = tiny_raw()
        raw.loc[0, "high"] = raw.loc[0, "low"] - 1
        with self.assertRaisesRegex(ValueError, "OHLC"):
            validate_ohlcv_15m(
                raw, assets=("BTC", "ETH"),
                as_of=pd.Timestamp("2026-01-01 01:00Z"),
            )
        with self.assertRaisesRegex(ValueError, "assets"):
            validate_ohlcv_15m(
                tiny_raw().query("asset == 'BTC'"), assets=("BTC", "ETH"),
                as_of=pd.Timestamp("2026-01-01 01:00Z"),
            )
```

- [ ] **Step 2: Run the tests and confirm the module is missing**

Run: `python -m unittest tests.test_tft_data -v`

Expected: FAIL because `neural.tft_data` does not exist.

- [ ] **Step 3: Implement strict validation and deterministic completion**

```python
# neural/tft_data.py
from dataclasses import dataclass
import pandas as pd

FREQ = "15min"
REQUIRED_ASSETS = ("BTC", "ETH", "BNB", "SOL", "XRP")
REQUIRED_COLUMNS = ("asset", "date", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class GapStats:
    observed_bars: int
    synthetic_bars: int
    gap_events: int
    max_gap_bars: int


def validate_ohlcv_15m(raw, assets=REQUIRED_ASSETS, as_of=None):
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")
    frame = raw.loc[raw.asset.isin(assets), REQUIRED_COLUMNS].copy()
    if set(frame.asset.unique()) != set(assets):
        raise ValueError("missing required assets")
    if frame.duplicated(["asset", "date"]).any():
        raise ValueError("duplicate (asset, date)")
    if frame.date.dt.tz is None:
        raise ValueError("timestamps must be timezone-aware UTC")
    if not all(g.date.is_monotonic_increasing for _, g in frame.groupby("asset")):
        raise ValueError("timestamps must be sorted within asset")
    prices = frame[["open", "high", "low", "close"]]
    invalid = ((prices <= 0).any(axis=1) | (frame.volume < 0)
               | (frame.high < prices[["open", "close", "low"]].max(axis=1))
               | (frame.low > prices[["open", "close", "high"]].min(axis=1)))
    if invalid.any():
        raise ValueError("invalid OHLC or volume")
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if ((frame.date + pd.Timedelta(FREQ)) > as_of).any():
        raise ValueError("incomplete final candle")
    return frame.reset_index(drop=True)


def complete_ohlcv_grid(raw, assets=REQUIRED_ASSETS):
    frames, summaries = [], {}
    for asset in assets:
        observed = raw[raw.asset == asset].set_index("date").sort_index()
        index = pd.date_range(observed.index.min(), observed.index.max(), freq=FREQ, tz="UTC")
        grid = observed.reindex(index)
        missing = grid.close.isna()
        event_start = missing & ~missing.shift(fill_value=False)
        runs = missing.groupby((missing != missing.shift()).cumsum()).transform("sum")
        previous_close = grid.close.ffill()
        for column in ("open", "high", "low", "close"):
            grid.loc[missing, column] = previous_close.loc[missing]
        grid.loc[missing, "volume"] = 0.0
        grid["missing_bar"] = missing.astype("int8")
        grid["asset"] = asset
        grid.index.name = "date"
        frames.append(grid.reset_index())
        summaries[asset] = GapStats(
            observed_bars=len(observed), synthetic_bars=int(missing.sum()),
            gap_events=int(event_start.sum()),
            max_gap_bars=int(runs[missing].max()) if missing.any() else 0,
        )
    return pd.concat(frames, ignore_index=True), summaries
```

- [ ] **Step 4: Run the focused tests**

Run: `python -m unittest tests.test_tft_data -v`

Expected: all Task 1 tests PASS.

- [ ] **Step 5: Commit the data contract**

```powershell
git add neural/tft_data.py tests/test_tft_data.py
git commit -m "feat(tft): validate intraday source data"
```

---

### Task 2: Build causal TFT features and aligned daily origins

**Files:**
- Modify: `neural/tft_data.py`
- Modify: `tests/test_tft_data.py`

**Interfaces:**
- Consumes: validated/completed grid from Task 1.
- Produces: `PreparedTFTData`, `prepare_tft_data(raw, assets, as_of)`, `eligible_daily_origins(context, horizon)`, and `assert_tft_feature_causality(raw, cutoff, assets, as_of)`.

- [ ] **Step 1: Add failing feature, alignment, and causality tests**

```python
# append to tests/test_tft_data.py
import numpy as np
from neural.tft_data import (
    HIST_EXOG, PreparedTFTData, assert_tft_feature_causality,
    eligible_daily_origins, prepare_tft_data,
)


def long_raw_fixture(assets=("BTC", "ETH"), periods=14 * 96):
    dates = pd.date_range("2025-12-20", periods=periods, freq="15min", tz="UTC")
    rows = []
    for index, asset in enumerate(assets, start=1):
        close = index * 100 + np.arange(periods) * 0.01
        rows.append(pd.DataFrame({
            "asset": asset, "date": dates, "open": close,
            "high": close * 1.001, "low": close * 0.999,
            "close": close, "volume": 10 + np.arange(periods) % 7,
        }))
    raw = pd.concat(rows, ignore_index=True)
    return raw, pd.Timestamp("2025-12-28 23:45Z"), dates[-1] + pd.Timedelta("15min")


def prepared_fixture_with_one_missing_target_bar():
    raw, _, as_of = long_raw_fixture()
    raw = raw[raw.date != pd.Timestamp("2026-01-02 12:00Z")].reset_index(drop=True)
    return prepare_tft_data(raw, assets=("BTC", "ETH"), as_of=as_of)


class TftFeatureTests(unittest.TestCase):
    def test_prepared_frame_has_exact_causal_features_and_daily_endpoint(self):
        dates = pd.date_range("2025-12-20", periods=800, freq="15min", tz="UTC")
        rows = []
        for asset, scale in (("BTC", 1.0), ("ETH", 2.0)):
            close = scale * (100 + np.arange(len(dates)) * 0.01)
            rows.append(pd.DataFrame({
                "asset": asset, "date": dates, "open": close,
                "high": close * 1.001, "low": close * 0.999,
                "close": close, "volume": 10 + np.arange(len(dates)) % 7,
            }))
        prepared = prepare_tft_data(
            pd.concat(rows), assets=("BTC", "ETH"),
            as_of=pd.Timestamp("2026-01-01 12:00Z"),
        )
        self.assertIsInstance(prepared, PreparedTFTData)
        self.assertEqual(set(HIST_EXOG) - set(prepared.model_frame), set())
        self.assertEqual(prepared.model_frame.ds.max().strftime("%H:%M"), "23:45")
        self.assertTrue(np.isfinite(prepared.model_frame[list(HIST_EXOG)]).all().all())

    def test_target_origins_crossing_synthetic_bars_are_excluded_for_all_assets(self):
        prepared = prepared_fixture_with_one_missing_target_bar()
        origins = eligible_daily_origins(prepared.context_frame, horizon=96)
        affected = pd.Timestamp("2026-01-02 00:00Z")
        self.assertNotIn(affected, origins)

    def test_future_corruption_does_not_change_past_features(self):
        raw, cutoff, as_of = long_raw_fixture()
        assert_tft_feature_causality(
            raw, cutoff=cutoff, assets=("BTC", "ETH"), as_of=as_of,
        )
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run: `python -m unittest tests.test_tft_data.TftFeatureTests -v`

Expected: FAIL because the preparation interfaces are undefined.

- [ ] **Step 3: Implement the approved causal feature set**

```python
# append to neural/tft_data.py
import numpy as np

INPUT_SIZE = 672
HORIZON = 96
HIST_EXOG = (
    "log_return", "abs_return", "range_pct", "log_volume",
    "log_volume_change", "rv_96", "rv_672", "momentum_96",
    "momentum_672", "volume_z96", "missing_bar",
)
FUTR_EXOG = ("tod_sin", "tod_cos", "dow_sin", "dow_cos")


@dataclass(frozen=True)
class PreparedTFTData:
    model_frame: pd.DataFrame
    context_frame: pd.DataFrame
    gap_stats: dict
    data_end: pd.Timestamp


def _add_features(grid):
    parts = []
    for _, group in grid.groupby("asset", sort=False):
        g = group.sort_values("date").copy()
        log_close = np.log(g.close)
        g["log_return"] = log_close.diff()
        g["abs_return"] = g.log_return.abs()
        g["range_pct"] = (g.high - g.low) / g.close
        g["log_volume"] = np.log1p(g.volume)
        g["log_volume_change"] = g.log_volume.diff()
        g["rv_96"] = g.log_return.pow(2).rolling(96).sum().pow(0.5)
        g["rv_672"] = g.log_return.pow(2).rolling(672).sum().pow(0.5)
        g["sigma_672"] = g.log_return.rolling(672).std(ddof=0)
        g["momentum_96"] = log_close - log_close.shift(96)
        g["momentum_672"] = log_close - log_close.shift(672)
        mean = g.log_volume.rolling(96).mean()
        std = g.log_volume.rolling(96).std(ddof=0)
        g["volume_z96"] = ((g.log_volume - mean) / std.replace(0, np.nan)).fillna(0)
        minute = g.date.dt.hour * 60 + g.date.dt.minute
        g["tod_sin"] = np.sin(2 * np.pi * minute / 1440)
        g["tod_cos"] = np.cos(2 * np.pi * minute / 1440)
        g["dow_sin"] = np.sin(2 * np.pi * g.date.dt.dayofweek / 7)
        g["dow_cos"] = np.cos(2 * np.pi * g.date.dt.dayofweek / 7)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)
```

`prepare_tft_data` must call Task 1 validation/completion, trim every asset to the latest shared 23:45 UTC timestamp, compute features, drop only rows lacking the 672-bar history, and return:

```python
model_frame = featured.rename(columns={"asset": "unique_id"}).assign(
    ds=lambda d: d.date.dt.tz_convert("UTC").dt.tz_localize(None),
    y=lambda d: np.log(d.close),
)[["unique_id", "ds", "y", *HIST_EXOG, *FUTR_EXOG]]

context_frame = featured.assign(
    ds=lambda d: d.date.dt.tz_convert("UTC").dt.tz_localize(None),
)[[
    "asset", "ds", "open", "close", "missing_bar",
    "rv_672", "sigma_672", "momentum_96",
]]
```

NeuralForecast receives UTC-naive `ds` values because its frequency validator
expects a timezone-naive regular grid. `to_tft_forecasts` in Task 3 localizes
them back to UTC before publishing evidence. `eligible_daily_origins` must
return common timezone-aware 00:00 UTC origins whose preceding cutoff is 23:45
UTC and whose next 96 context rows contain no synthetic bar for any asset.
`assert_tft_feature_causality` must rebuild after corrupting post-cutoff OHLCV,
then compare every historical feature at or before the cutoff for every asset.

- [ ] **Step 4: Run feature tests and the complete data test module**

Run: `python -m unittest tests.test_tft_data -v`

Expected: PASS, including the corruption check.

- [ ] **Step 5: Commit feature preparation**

```powershell
git add neural/tft_data.py tests/test_tft_data.py
git commit -m "feat(tft): build causal next-day features"
```

---

### Task 3: Convert and validate TFT forecasts and aligned baselines

**Files:**
- Create: `neural/tft_evaluation.py`
- Create: `tests/test_tft_evaluation.py`

**Interfaces:**
- Consumes: NeuralForecast CV rows and `PreparedTFTData.context_frame` from Task 2.
- Produces: `to_tft_forecasts(cv, context, model) -> pd.DataFrame`, `validate_tft_forecasts(frame, expected_horizons)`, `split_calibration_test(frame, eligible_origins, n_calibration, n_test)`, and `make_tft_baselines(reference) -> pd.DataFrame`.
- Forecast schema: `model,asset,origin,target_time,split,h,y,last,regime_driver,origin_sigma,origin_momentum,q10,q50,q90`.

- [ ] **Step 1: Write failing conversion, split, and baseline tests**

```python
# tests/test_tft_evaluation.py
import unittest
import numpy as np
import pandas as pd

from neural.tft_evaluation import (
    make_tft_baselines, split_calibration_test,
    to_tft_forecasts, validate_tft_forecasts,
)


def cv_fixture(origins=2, horizons=3, assets=("BTC", "ETH")):
    cutoffs = pd.date_range("2025-01-01 23:45", periods=origins, freq="D")
    rows, context = [], []
    for asset_index, asset in enumerate(assets, start=1):
        asset_offset = asset_index * 100.0
        for origin_index, cutoff in enumerate(cutoffs):
            last = asset_offset + origin_index
            context.append({
                "asset": asset, "ds": cutoff, "close": last,
                "rv_672": 0.02, "sigma_672": 0.001,
                "momentum_96": 0.01, "missing_bar": 0,
            })
            for h in range(1, horizons + 1):
                target_log = np.log(last * (1 + 0.001 * h))
                rows.append({
                    "index": len(rows), "unique_id": asset,
                    "ds": cutoff + pd.Timedelta("15min") * h, "cutoff": cutoff,
                    "TFT-lo-80.0": target_log - 0.02,
                    "TFT-median": target_log,
                    "TFT-hi-80.0": target_log + 0.02,
                    "y": target_log,
                })
    return pd.DataFrame(rows), pd.DataFrame(context)


def forecast_fixture(origins=4, horizons=3, assets=("BTC", "ETH")):
    origin_index = pd.date_range("2025-01-01", periods=origins, freq="D", tz="UTC")
    rows = []
    for asset_index, asset in enumerate(assets, start=1):
        for date_index, origin in enumerate(origin_index):
            last = asset_index * 100.0 + date_index
            for h in range(1, horizons + 1):
                actual = last * (1 + 0.0015 * h)
                predicted = last * (1 + 0.001 * h)
                rows.append({
                    "model": "tft_raw", "asset": asset, "origin": origin,
                    "target_time": origin + pd.Timedelta("15min") * h,
                    "split": "unassigned", "h": h, "y": actual, "last": last,
                    "regime_driver": 0.02 + date_index * 0.0001,
                    "origin_sigma": 0.001, "origin_momentum": 0.01,
                    "q10": predicted * 0.98, "q50": predicted,
                    "q90": predicted * 1.02,
                })
    frame = pd.DataFrame(rows)
    return frame, origin_index


class TftForecastFrameTests(unittest.TestCase):
    def test_conversion_uses_candle_close_times_and_price_space(self):
        cv, context = cv_fixture(origins=2, horizons=3, assets=("BTC", "ETH"))
        frame = to_tft_forecasts(cv, context, model="tft_raw")
        first = frame.sort_values(["asset", "origin", "h"]).iloc[0]
        self.assertEqual(first.origin.minute, 0)
        self.assertEqual(first.target_time, first.origin + pd.Timedelta("15min"))
        self.assertEqual(first.h, 1)
        self.assertAlmostEqual(first.q50, np.exp(cv.iloc[0]["TFT-median"]))
        validate_tft_forecasts(frame, expected_horizons=(1, 2, 3))

    def test_split_uses_219_calibration_and_146_untouched_origins(self):
        frame, origins = forecast_fixture(origins=365, horizons=2)
        calibration, test = split_calibration_test(
            frame, origins, n_calibration=219, n_test=146,
        )
        self.assertEqual(calibration.origin.nunique(), 219)
        self.assertEqual(test.origin.nunique(), 146)
        self.assertLess(calibration.origin.max(), test.origin.min())
        self.assertEqual(set(calibration.split), {"calibration"})
        self.assertEqual(set(test.split), {"test"})

    def test_baselines_preserve_keys_truth_and_quantile_order(self):
        reference, _ = forecast_fixture(origins=4, horizons=3)
        baselines = make_tft_baselines(reference)
        self.assertEqual(set(baselines.model), {"persistence_vol", "momentum_vol"})
        self.assertFalse(baselines.duplicated(["model", "asset", "origin", "h"]).any())
        self.assertTrue((baselines.q10 <= baselines.q50).all())
        self.assertTrue((baselines.q50 <= baselines.q90).all())
        for _, group in baselines.groupby("model"):
            pd.testing.assert_frame_equal(
                group[["asset", "origin", "h", "y", "last"]].reset_index(drop=True),
                reference[["asset", "origin", "h", "y", "last"]].reset_index(drop=True),
            )
```

- [ ] **Step 2: Run the module and verify missing-interface failures**

Run: `python -m unittest tests.test_tft_evaluation.TftForecastFrameTests -v`

Expected: FAIL because `neural.tft_evaluation` does not exist.

- [ ] **Step 3: Implement forecast mapping and strict validation**

Use exact keys and fail on NaN, infinity, duplicates, incomplete horizons, crossed quantiles, non-common model grids, or fewer than 100 test origins.

```python
# neural/tft_evaluation.py
FORECAST_KEY = ["asset", "origin", "h"]
KEY = ["model", *FORECAST_KEY]
REQUIRED = [
    "model", "asset", "origin", "target_time", "split", "h", "y", "last",
    "regime_driver", "origin_sigma", "origin_momentum", "q10", "q50", "q90",
]


def _quantile_columns(cv):
    return {
        "q10": next(c for c in cv if "-lo-" in c),
        "q50": next(c for c in cv if c.endswith("-median")),
        "q90": next(c for c in cv if "-hi-" in c),
    }


def split_calibration_test(frame, eligible_origins, n_calibration=219, n_test=146):
    ordered = pd.DatetimeIndex(sorted(set(eligible_origins)))
    required = n_calibration + n_test
    if len(ordered) < required:
        raise ValueError(f"need {required} eligible daily origins, found {len(ordered)}")
    selected = ordered[-required:]
    cal_origins, test_origins = selected[:n_calibration], selected[n_calibration:]
    calibration = frame[frame.origin.isin(cal_origins)].copy()
    test = frame[frame.origin.isin(test_origins)].copy()
    calibration["split"], test["split"] = "calibration", "test"
    if test.origin.nunique() < 100:
        raise ValueError("fewer than 100 valid test origins")
    return calibration, test
```

`to_tft_forecasts` must localize NeuralForecast's UTC-naive `cutoff` and `ds`
back to UTC, interpret both as candle-open timestamps, set `origin=cutoff+15min`, `target_time=ds+15min`, calculate `h=(ds-cutoff)/15min`, exponentiate log truth and quantiles, join the cutoff's close, `rv_672`, `sigma_672`, and `momentum_96` from context, and initialize every row with `split="unassigned"`.

- [ ] **Step 4: Implement reproducible probabilistic baselines**

For every reference row:

```python
z = 1.2815515655
root_h = np.sqrt(reference.h.to_numpy())
origin_log = np.log(reference["last"].to_numpy())
band = z * reference.origin_sigma.to_numpy() * root_h

persistence_mid = origin_log
momentum_mid = origin_log + (reference.h.to_numpy() / 96) * reference.origin_momentum.to_numpy()
```

Build `persistence_vol` and `momentum_vol` rows with `q10=exp(mid-band)`, `q50=exp(mid)`, and `q90=exp(mid+band)`. Preserve all context and truth columns exactly and validate the concatenated result.

- [ ] **Step 5: Run focused and existing forecast-validation tests**

Run: `python -m unittest tests.test_tft_evaluation tests.test_evaluation -v`

Expected: PASS with no daily-evaluation regression.

- [ ] **Step 6: Commit forecast conversion and baselines**

```powershell
git add neural/tft_evaluation.py tests/test_tft_evaluation.py
git commit -m "feat(tft): align next-day forecast evidence"
```

---

### Task 4: Add leakage-safe calibration and diagnostic metrics

**Files:**
- Modify: `neural/tft_evaluation.py`
- Modify: `tests/test_tft_evaluation.py`

**Interfaces:**
- Consumes: calibration and test frames from Task 3.
- Produces: `calibrate_tft_intervals(calibration, test, alpha)`, `fit_regime_cutpoints(calibration)`, `apply_regimes(frame, cutpoints)`, `headline_metrics(frame, bootstrap_samples, seed)`, and `tft_metric_tables(calibration, test)`.

- [ ] **Step 1: Write failing asymmetric calibration and metric tests**

```python
# append to tests/test_tft_evaluation.py
from neural.tft_evaluation import calibrate_tft_intervals, headline_metrics


def asymmetric_interval_fixture():
    frame, _ = forecast_fixture(origins=8, horizons=2, assets=("BTC",))
    ordered = sorted(frame.origin.unique())
    calibration = frame[frame.origin.isin(ordered[:5])].copy().assign(split="calibration")
    test = frame[frame.origin.isin(ordered[5:])].copy().assign(split="test")
    calibration.loc[:, "y"] = 100.0
    calibration.loc[:, "q10"] = 101.0
    calibration.loc[:, "q50"] = 102.0
    calibration.loc[:, "q90"] = 104.0
    return calibration, test


def metric_fixture_with_price_level_trend():
    origins = pd.date_range("2025-01-01", periods=30, freq="D", tz="UTC")
    rows = []
    for model in ("good_returns", "bad_returns"):
        for index, origin in enumerate(origins):
            last = 1000.0 + index * 10
            actual_return = 0.01 if index % 2 == 0 else -0.01
            predicted_return = actual_return if model == "good_returns" else -actual_return
            actual, median = last * (1 + actual_return), last * (1 + predicted_return)
            rows.append({
                "model": model, "asset": "BTC", "origin": origin,
                "target_time": origin + pd.Timedelta("1D"), "split": "test",
                "h": 96, "y": actual, "last": last, "regime_driver": 0.02,
                "origin_sigma": 0.001, "origin_momentum": 0.0,
                "q10": median * 0.98, "q50": median, "q90": median * 1.02,
            })
    return pd.DataFrame(rows)


class TftCalibrationMetricTests(unittest.TestCase):
    def test_calibration_uses_only_calibration_rows_and_never_tightens(self):
        calibration, test = asymmetric_interval_fixture()
        first = calibrate_tft_intervals(calibration, test, alpha=0.20)
        corrupted = test.assign(y=test.y * 100)
        second = calibrate_tft_intervals(calibration, corrupted, alpha=0.20)
        np.testing.assert_allclose(first.q10, second.q10)
        np.testing.assert_allclose(first.q90, second.q90)
        self.assertTrue((first.q10 <= test.q10).all())
        self.assertTrue((first.q90 >= test.q90).all())
        self.assertNotEqual(
            np.log(test.q10.iloc[0]) - np.log(first.q10.iloc[0]),
            np.log(first.q90.iloc[0]) - np.log(test.q90.iloc[0]),
        )

    def test_headline_metrics_use_only_horizon_96_and_return_r_squared(self):
        frame = metric_fixture_with_price_level_trend()
        metrics = headline_metrics(frame, bootstrap_samples=200, seed=42)
        self.assertEqual(set(metrics.index), set(frame.model))
        self.assertIn("pinball_%", metrics)
        self.assertIn("direction_accuracy", metrics)
        self.assertIn("direction_ci_low", metrics)
        self.assertIn("return_r2", metrics)
        self.assertLess(metrics.loc["bad_returns", "return_r2"], 0)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m unittest tests.test_tft_evaluation.TftCalibrationMetricTests -v`

Expected: FAIL because the calibration and metric functions are undefined.

- [ ] **Step 3: Implement finite-sample one-sided calibration**

```python
def _higher_quantile(values, level):
    return float(np.quantile(np.asarray(values), min(level, 1.0), method="higher"))


def calibrate_tft_intervals(calibration, test, alpha=0.20):
    adjusted = test.copy()
    for horizon, cal_h in calibration.groupby("h", sort=True):
        n = len(cal_h)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha / 2)) / n)
        lower_scores = np.log(cal_h.q10) - np.log(cal_h.y)
        upper_scores = np.log(cal_h.y) - np.log(cal_h.q90)
        lower = max(0.0, _higher_quantile(lower_scores, level))
        upper = max(0.0, _higher_quantile(upper_scores, level))
        mask = adjusted.h == horizon
        adjusted.loc[mask, "q10"] = np.exp(np.log(adjusted.loc[mask, "q10"]) - lower)
        adjusted.loc[mask, "q90"] = np.exp(np.log(adjusted.loc[mask, "q90"]) + upper)
    adjusted["model"] = "tft_calibrated"
    validate_tft_forecasts(adjusted, expected_horizons=sorted(test.h.unique()))
    return adjusted
```

Require identical calibration/test model identity before changing the model tag. Raise when a horizon has no calibration rows.

- [ ] **Step 4: Implement headline and subgroup metrics**

Use `crypto.backtest.score` for pinball percentage, MAPE, coverage, and normalized width. For h=96 rows calculate predicted and actual simple returns from `q50/last-1` and `y/last-1`, then calculate return MAE, return R-squared, Pearson correlation, and sign agreement. Bootstrap the 95% direction interval by sampling whole origin dates with replacement and retaining every asset row belonging to each sampled date.

```python
def _direction_interval(group, bootstrap_samples, seed):
    origins = group.origin.drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(bootstrap_samples):
        sampled = rng.choice(origins, size=len(origins), replace=True)
        rows = pd.concat(
            [group.loc[group.origin.eq(origin)] for origin in sampled],
            ignore_index=True,
        )
        estimates.append(np.mean(np.sign(rows.q50 / rows["last"] - 1)
                                 == np.sign(rows.y / rows["last"] - 1)) * 100)
    return np.quantile(estimates, [0.025, 0.975])


def headline_metrics(frame, bootstrap_samples=2000, seed=42):
    final = frame.loc[frame.h.eq(96)].copy()
    if final.empty:
        raise ValueError("headline metrics require h=96 rows")
    diagnostics = []
    for model, group in final.groupby("model", sort=True):
        actual = group.y.to_numpy() / group["last"].to_numpy() - 1
        predicted = group.q50.to_numpy() / group["last"].to_numpy() - 1
        residual = np.sum((actual - predicted) ** 2)
        total = np.sum((actual - actual.mean()) ** 2)
        low, high = _direction_interval(group, bootstrap_samples, seed)
        diagnostics.append({
            "model": model,
            "return_mae": np.mean(np.abs(actual - predicted)),
            "return_r2": 1 - residual / total if total > 0 else np.nan,
            "return_correlation": np.corrcoef(actual, predicted)[0, 1],
            "direction_accuracy": np.mean(np.sign(predicted) == np.sign(actual)) * 100,
            "direction_ci_low": low, "direction_ci_high": high,
        })
    return score(final).join(pd.DataFrame(diagnostics).set_index("model"))
```

Fit volatility-regime terciles from unique calibration `(asset, origin, regime_driver)` rows and apply those fixed cutpoints to test rows. `tft_metric_tables` returns:

```python
{
    "overall": headline_metrics(test),
    "by_horizon": grouped_interval_metrics(test, ["model", "h"]),
    "by_asset": grouped_headline_metrics(test, ["model", "asset"]),
    "by_regime": grouped_headline_metrics(apply_regimes(test, cutpoints), ["model", "regime"]),
}
```

- [ ] **Step 5: Run evaluation tests and the full existing metric suite**

Run: `python -m unittest tests.test_tft_evaluation tests.test_backtest tests.test_evaluation -v`

Expected: PASS; return R-squared is never calculated on price levels.

- [ ] **Step 6: Commit calibration and metrics**

```powershell
git add neural/tft_evaluation.py tests/test_tft_evaluation.py
git commit -m "feat(tft): calibrate next-day uncertainty"
```

---

### Task 5: Add versioned TFT checkpoints and evidence manifests

**Files:**
- Modify: `crypto/evaluation.py`
- Create: `neural/tft_artifacts.py`
- Create: `tests/test_tft_artifacts.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: run metadata, NeuralForecast object, raw CV, raw test, calibrated test, metrics/graphs created later.
- Produces: backward-compatible `reserve_run_dir(..., pipeline="daily")`, `reserve_tft_run(data_end, run_id, root)`, `write_raw_cv(output_dir, raw_cv)`, `save_tft_core(...)`, `finalize_tft_run(output_dir, metadata)`, and `verify_tft_manifest(output_dir)`.

- [ ] **Step 1: Write failing backward-compatibility and artifact tests**

```python
# append to tests/test_evaluation.py
def test_run_directory_pipeline_prefix_defaults_to_daily(self):
    from crypto.evaluation import default_run_dir
    path = default_run_dir("tree", "2026-07-23", "abc", root="out")
    self.assertEqual(path.as_posix(), "out/daily-tree-20260723-abc")
    hf = default_run_dir("tft", "2026-07-31", "abc", root="out", pipeline="hf15m")
    self.assertEqual(hf.as_posix(), "out/hf15m-tft-20260731-abc")
```

```python
# tests/test_tft_artifacts.py
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from neural.tft_artifacts import (
    finalize_tft_run, reserve_tft_run, save_tft_core, verify_tft_manifest, write_raw_cv,
)
from tests.test_tft_evaluation import forecast_fixture


def raw_cv_frame():
    return pd.DataFrame({
        "unique_id": ["BTC"], "cutoff": [pd.Timestamp("2025-01-01 23:45")],
        "ds": [pd.Timestamp("2025-01-02 00:00")], "y": [4.6],
        "TFT-lo-80.0": [4.5], "TFT-median": [4.6], "TFT-hi-80.0": [4.7],
    })


def raw_test_frame():
    frame, _ = forecast_fixture(origins=2, horizons=2, assets=("BTC",))
    return frame.assign(split="test")


def calibrated_test_frame():
    return raw_test_frame().assign(model="tft_calibrated")


def complete_metadata():
    return {
        "run_id": "fixture", "pipeline": "hf15m", "family": "tft",
        "data_path": "data/ohlcv_15m.parquet", "data_start": "2021-01-01",
        "data_end": "2026-07-31", "assets": ["BTC"], "rows": 1000,
        "gap_stats": {}, "features": ["log_return"],
        "future_features": ["tod_sin"], "config": {"horizon": 96},
        "train_end": "2024-07-31", "validation_start": "2024-07-04",
        "validation_end": "2024-07-31", "calibration_start": "2024-08-01",
        "calibration_end": "2025-03-07", "test_start": "2025-03-08",
        "test_end": "2025-07-31", "package_versions": {"neuralforecast": "3.2.0"},
        "git_commit": "fixture", "elapsed_seconds": 1.0, "device": "cpu",
    }


class FakeNeuralForecast:
    def save(self, path, save_dataset=True, overwrite=False):
        target = Path(path)
        target.mkdir()
        (target / "weights.ckpt").write_bytes(b"weights")


class TftArtifactTests(unittest.TestCase):
    def test_complete_run_hashes_checkpoint_predictions_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = reserve_tft_run("2026-07-31", "fixture", Path(tmp))
            self.assertEqual(json.loads((out / "status.json").read_text())["state"], "incomplete")
            write_raw_cv(out, raw_cv_frame())
            save_tft_core(
                out, FakeNeuralForecast(), raw_test_frame(), calibrated_test_frame(),
            )
            finalize_tft_run(out, complete_metadata(), extra_paths=[])
            verify_tft_manifest(out)
            self.assertEqual(json.loads((out / "status.json").read_text())["state"], "complete")
            self.assertTrue((out / "model" / "weights.ckpt").exists())
            self.assertTrue((out / "raw_cv.parquet").exists())
            self.assertTrue((out / "manifest.json").exists())

    def test_collision_and_unsafe_run_ids_fail_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            reserve_tft_run("2026-07-31", "fixture", Path(tmp))
            with self.assertRaises(FileExistsError):
                reserve_tft_run("2026-07-31", "fixture", Path(tmp))
            with self.assertRaisesRegex(ValueError, "run_id"):
                reserve_tft_run("2026-07-31", "../escape", Path(tmp))
```

- [ ] **Step 2: Run tests and verify new APIs fail**

Run: `python -m unittest tests.test_tft_artifacts tests.test_evaluation -v`

Expected: FAIL because pipeline-aware reservation and TFT artifacts are absent.

- [ ] **Step 3: Generalize run-directory naming without changing daily defaults**

```python
# crypto/evaluation.py
def default_run_dir(tag, data_end, run_id, root=Path("artifacts/evaluation"), pipeline="daily"):
    stamp = pd.Timestamp(data_end).strftime("%Y%m%d")
    return Path(root) / f"{pipeline}-{tag}-{stamp}-{run_id}"


def reserve_run_dir(tag, data_end, run_id=None, root=Path("artifacts/evaluation"), pipeline="daily"):
    run_id = _validated_run_id(run_id)
    root = Path(root).resolve()
    output_dir = default_run_dir(tag, data_end, run_id, root, pipeline=pipeline).resolve()
    output_dir.relative_to(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir
```

Preserve the existing explicit `ValueError` for paths outside `root`; do not expose `_validated_run_id` from the module.

- [ ] **Step 4: Implement incomplete/complete TFT artifact lifecycle**

`reserve_tft_run` calls `reserve_run_dir("tft", ..., pipeline="hf15m")` and atomically writes `status.json` with `{"state":"incomplete"}`. `write_raw_cv` atomically writes `raw_cv.parquet` without overwriting, immediately after cross-validation returns. `save_tft_core` calls `nf.save(str(output_dir / "model"), save_dataset=True, overwrite=False)` and writes `predictions_raw_test.parquet` and `predictions_calibrated_test.parquet` without overwriting.

`finalize_tft_run` must validate required metadata fields, write portable `metadata.json`, hash every checkpoint, prediction, table, graph, and metadata file using SHA-256, write `manifest.json`, verify every hash, then atomically replace status with `{"state":"complete"}`. Required metadata fields are:

```python
REQUIRED_METADATA = {
    "run_id", "pipeline", "family", "data_path", "data_end", "data_start",
    "assets", "rows", "gap_stats", "features", "future_features", "config",
    "train_end", "validation_start", "validation_end", "calibration_start",
    "calibration_end", "test_start", "test_end", "package_versions",
    "git_commit", "elapsed_seconds", "device",
}
```

Convert absolute paths underneath the configured provenance root to relative POSIX paths; reject paths outside it. `verify_tft_manifest` recalculates every recorded hash and rejects missing, extra, or changed evidence files.

- [ ] **Step 5: Run artifact and existing evaluation tests**

Run: `python -m unittest tests.test_tft_artifacts tests.test_evaluation -v`

Expected: PASS; daily directories retain their original names.

- [ ] **Step 6: Commit artifact provenance**

```powershell
git add crypto/evaluation.py neural/tft_artifacts.py tests/test_tft_artifacts.py tests/test_evaluation.py
git commit -m "feat(tft): preserve reproducible model runs"
```

---

### Task 6: Render TFT-only tables and prediction diagnostics

**Files:**
- Create: `neural/tft_plots.py`
- Create: `tests/test_tft_plots.py`

**Interfaces:**
- Consumes: calibration rows plus untouched test rows containing TFT and both baselines.
- Produces: `render_tft_report(calibration, test, output_dir) -> list[Path]`.

- [ ] **Step 1: Write failing fixture-report tests**

```python
# tests/test_tft_plots.py
import tempfile
import unittest
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from neural.tft_plots import render_tft_report
from neural.tft_evaluation import make_tft_baselines
from tests.test_tft_evaluation import forecast_fixture


def report_fixture():
    frame, origins = forecast_fixture(origins=6, horizons=96, assets=("BTC", "ETH"))
    calibration = frame[frame.origin.isin(origins[:2])].copy().assign(split="calibration")
    raw_test = frame[frame.origin.isin(origins[2:])].copy().assign(split="test")
    calibrated = raw_test.assign(model="tft_calibrated")
    baselines = make_tft_baselines(raw_test)
    return calibration, pd.concat([raw_test, calibrated, baselines], ignore_index=True)


class TftPlotTests(unittest.TestCase):
    def test_report_creates_exact_tables_and_graphs_and_closes_figures(self):
        calibration, test = report_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            paths = render_tft_report(calibration, test, Path(tmp))
            names = {path.name for path in paths}
            self.assertEqual(names, {
                "metrics_overall.csv", "metrics_by_horizon.csv",
                "metrics_by_asset.csv", "metrics_by_regime.csv",
                "forecast_path.png", "next_day_forecasts.png",
                "returns_scatter.png", "performance_by_asset.png",
                "performance_by_regime.png", "calibration_comparison.png",
            })
            self.assertEqual([], plt.get_fignums())

    def test_headline_graphs_reject_non_test_rows_and_missing_baselines(self):
        calibration, test = report_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "test"):
                render_tft_report(calibration, test.assign(split="calibration"), Path(tmp))
            with self.assertRaisesRegex(ValueError, "baseline"):
                render_tft_report(
                    calibration, test[test.model.str.startswith("tft")], Path(tmp),
                )
```

- [ ] **Step 2: Run tests and verify the plot module is absent**

Run: `python -m unittest tests.test_tft_plots -v`

Expected: FAIL because `neural.tft_plots` does not exist.

- [ ] **Step 3: Implement deterministic report generation**

Use matplotlib's `Agg` backend and always close figures after saving. `render_tft_report` validates inputs, writes the four tables from `tft_metric_tables`, and produces:

```python
matplotlib.use("Agg")


def render_tft_report(calibration, test, output_dir):
    if set(test.split) != {"test"}:
        raise ValueError("headline graphs require only test rows")
    if not {"persistence_vol", "momentum_vol"}.issubset(set(test.model)):
        raise ValueError("both baseline models are required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = tft_metric_tables(calibration, test)
    paths = []
    for name, table in tables.items():
        path = output_dir / f"metrics_{name}.csv"
        table.to_csv(path)
        paths.append(path)
    for renderer in REPORT_RENDERERS:
        paths.append(renderer(calibration, test, output_dir))
    plt.close("all")
    return paths
```

Define `REPORT_RENDERERS` in the exact filename order asserted by the test, with one focused helper per graph. Each helper creates its own figure, saves once with `bbox_inches="tight"`, closes that figure in `finally`, and returns the output path. It must implement the following plots:

- `forecast_path.png`: the latest untouched origin's 96 calibrated TFT steps for every asset with q10-q90 bands;
- `next_day_forecasts.png`: h=96 actual and q50 over all test origins, faceted by asset;
- `returns_scatter.png`: h=96 predicted versus realized return with a zero line and identity line;
- `performance_by_asset.png`: pinball percentage and direction accuracy by asset;
- `performance_by_regime.png`: pinball percentage and direction accuracy by fixed regime;
- `calibration_comparison.png`: raw and calibrated coverage by horizon with the 80% reference.

Every title includes split, OOS start/end, and distinct daily-origin count. Do not import or reuse the daily plot module because it hard-codes seven-day labels and the `vol_21d` baseline.

- [ ] **Step 4: Run TFT plot tests and existing plot tests**

Run: `python -m unittest tests.test_tft_plots tests.test_evaluation_plots -v`

Expected: PASS and no open matplotlib figures.

- [ ] **Step 5: Commit TFT reporting**

```powershell
git add neural/tft_plots.py tests/test_tft_plots.py
git commit -m "feat(tft): render next-day forecast evidence"
```

---

### Task 7: Wire the user-operated TFT runner and CLI

**Files:**
- Create: `neural/tft_runner.py`
- Modify: `neural/nf_run.py`
- Create: `tests/test_tft_runner.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: Task 2 data, Task 3/4 evaluation, Task 5 artifacts, Task 6 reports.
- Produces: `TFTExperimentConfig`, `build_tft(config)`, `run_tft(config, nf_class=NeuralForecast) -> Path`, and the preserved command `python -m neural.nf_run --model tft ...`.

- [ ] **Step 1: Write failing deterministic-construction and no-training runner tests**

```python
# tests/test_tft_runner.py
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

from neural.tft_runner import TFTExperimentConfig, build_tft, run_tft
from tests.test_tft_evaluation import cv_fixture


def fixture_parquet_loader(path, columns=None):
    periods = 732 * 96
    dates = pd.date_range("2024-01-01", periods=periods, freq="15min", tz="UTC")
    rows = []
    for index, asset in enumerate(("BTC", "ETH"), start=1):
        close = index * 100 + np.arange(periods) * 0.0001
        rows.append(pd.DataFrame({
            "asset": asset, "date": dates, "open": close,
            "high": close * 1.001, "low": close * 0.999,
            "close": close, "volume": 10 + np.arange(periods) % 7,
        }))
    raw = pd.concat(rows, ignore_index=True)
    return raw if columns is None else raw.loc[:, columns]


class FakeNF:
    instances = []
    def __init__(self, models, freq):
        self.models, self.freq = models, freq
        self.cross_validation_calls = []
        FakeNF.instances.append(self)
    def cross_validation(self, frame, **kwargs):
        self.cross_validation_calls.append(kwargs)
        return cv_fixture(origins=365, horizons=96, assets=("BTC", "ETH"))
    def save(self, path, save_dataset=True, overwrite=False):
        target = Path(path); target.mkdir(); (target / "fixture.ckpt").write_bytes(b"x")


class TftRunnerTests(unittest.TestCase):
    def test_model_construction_is_seeded_before_tft_initialization(self):
        config = TFTExperimentConfig(run_id="fixture", assets=("BTC", "ETH"))
        first = build_tft(config)
        torch.manual_seed(999)
        second = build_tft(config)
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]))

    @patch("neural.tft_runner.pd.read_parquet", side_effect=fixture_parquet_loader)
    def test_runner_uses_fixed_model_daily_cv_and_never_refits(self, _):
        with tempfile.TemporaryDirectory() as tmp:
            config = TFTExperimentConfig(
                run_id="fixture", output_root=Path(tmp),
                assets=("BTC", "ETH"), max_steps=1,
            )
            run_tft(config, nf_class=FakeNF)
            call = FakeNF.instances[-1].cross_validation_calls[-1]
            self.assertEqual(call["n_windows"], 365)
            self.assertEqual(call["step_size"], 96)
            self.assertEqual(call["val_size"], 28 * 96)
            self.assertIs(call["refit"], False)

    def test_cli_forwards_operational_tft_options(self):
        from neural.nf_run import main
        with patch("neural.nf_run.run_tft") as run:
            main([
                "--model", "tft", "--run-id", "manual-1",
                "--output-root", "runs", "--accelerator", "gpu",
                "--batch-size", "16",
            ])
        config = run.call_args.args[0]
        self.assertEqual(config.run_id, "manual-1")
        self.assertEqual(config.accelerator, "gpu")
        self.assertEqual(config.batch_size, 16)
```

The runner fixture must use two assets and a small fake forecast frame; it must never invoke real Lightning training.

- [ ] **Step 2: Run tests and confirm runner APIs are missing**

Run: `python -m unittest tests.test_tft_runner -v`

Expected: FAIL because `neural.tft_runner` and `nf_run.main` do not exist.

- [ ] **Step 3: Implement deterministic TFT construction**

```python
# neural/tft_runner.py
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import TFT

from neural.tft_data import FUTR_EXOG, HIST_EXOG, HORIZON, INPUT_SIZE, REQUIRED_ASSETS


@dataclass(frozen=True)
class TFTExperimentConfig:
    run_id: str
    data_path: Path = Path("data/ohlcv_15m.parquet")
    output_root: Path = Path("artifacts/evaluation")
    assets: tuple[str, ...] = REQUIRED_ASSETS
    input_size: int = INPUT_SIZE
    horizon: int = HORIZON
    n_windows: int = 365
    calibration_origins: int = 219
    test_origins: int = 146
    validation_bars: int = 28 * 96
    hidden_size: int = 128
    max_steps: int = 4000
    batch_size: int = 32
    accelerator: str = "auto"
    seed: int = 42


def build_tft(config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    loss = MQLoss(quantiles=[0.1, 0.5, 0.9])
    return TFT(
        h=config.horizon, input_size=config.input_size,
        hist_exog_list=list(HIST_EXOG), futr_exog_list=list(FUTR_EXOG),
        hidden_size=config.hidden_size, loss=loss,
        valid_loss=MQLoss(quantiles=[0.1, 0.5, 0.9]), scaler_type="robust",
        max_steps=config.max_steps, early_stop_patience_steps=5,
        val_check_steps=100, batch_size=config.batch_size,
        random_seed=config.seed, accelerator=config.accelerator,
        enable_progress_bar=True, logger=False,
    )
```

Pin `neuralforecast==3.2.0` in `requirements.txt` because the runner depends on
the inspected TFT, cross-validation, and checkpoint APIs from that version.

- [ ] **Step 4: Implement the orchestration without hiding training**

`run_tft` must perform this explicit sequence:

```python
def run_tft(config, nf_class=NeuralForecast):
    data_end = read_data_end(config.data_path)
    output_dir = reserve_tft_run(data_end, config.run_id, config.output_root)
    started = time.monotonic()
    try:
        prepared = prepare_tft_data(config.data_path, assets=config.assets)
        nf = nf_class(models=[build_tft(config)], freq="15min")
        raw_cv = nf.cross_validation(
            prepared.model_frame,
            n_windows=config.n_windows,
            step_size=config.horizon,
            val_size=config.validation_bars,
            refit=False,
        )
        write_raw_cv(output_dir, raw_cv)
        forecasts = to_tft_forecasts(raw_cv, prepared.context_frame, model="tft_raw")
        calibration, raw_test = split_calibration_test(
            forecasts, prepared.eligible_origins,
            config.calibration_origins, config.test_origins,
        )
        calibrated = calibrate_tft_intervals(calibration, raw_test, alpha=0.20)
        baselines = make_tft_baselines(raw_test)
        comparison = pd.concat([raw_test, calibrated, baselines], ignore_index=True)
        save_tft_core(output_dir, nf, raw_cv, raw_test, calibrated)
        report_paths = render_tft_report(calibration, comparison, output_dir)
        metadata = build_metadata(config, prepared, calibration, raw_test,
                                  elapsed_seconds=time.monotonic() - started)
        finalize_tft_run(output_dir, metadata, extra_paths=report_paths)
        verify_tft_manifest(output_dir)
        return output_dir
    except Exception as error:
        write_failure(output_dir, error)
        raise
```

1. Read only the date column to determine `data_end`, then reserve the run directory before full data preparation.
2. Load and prepare the complete source with `prepare_tft_data`.
3. Construct `NeuralForecast(models=[build_tft(config)], freq="15min")`.
4. Call `cross_validation(model_frame, n_windows=365, step_size=96, val_size=2688, refit=False)` exactly once.
5. Save raw CV immediately so a later scoring failure never requires retraining.
6. Convert forecasts, remove origins invalidated by synthetic target bars, and split 219/146.
7. Create raw test, calibrated test, persistence-volatility, and momentum-volatility frames on identical keys.
8. Save the fitted checkpoint and prediction files.
9. Render tables/graphs, assemble complete metadata, finalize hashes, and verify the manifest.
10. On any exception, preserve `status.json` as incomplete, write a concise `failure.json`, and re-raise.

Do not catch or downgrade data, calibration, validation, or artifact-integrity errors.

- [ ] **Step 5: Refactor `neural.nf_run` into a callable CLI without changing other models**

Move argument parsing into `main(argv=None)`. For `tft`, require/accept `--run-id`, `--output-root`, `--accelerator`, and `--batch-size`, build `TFTExperimentConfig`, and call `run_tft`. Preserve the existing LSTM, GRU, N-HiTS, and ensemble code paths and their two-hour configuration. Reject TFT-only operational options when another model is selected rather than silently ignoring them.

The user-facing command must be:

```powershell
python -m neural.nf_run --model tft --run-id <safe-run-id> --output-root artifacts/evaluation --accelerator gpu --batch-size 16
```

- [ ] **Step 6: Run runner, artifact, and legacy high-frequency tests**

Run: `python -m unittest tests.test_tft_runner tests.test_tft_artifacts tests.test_tft_data tests.test_tft_evaluation tests.test_tft_plots -v`

Expected: PASS without executing real training.

Run: `python -m neural.nf_run --help`

Expected: exits 0 and documents the new TFT options.

- [ ] **Step 7: Commit the runner**

```powershell
git add neural/tft_runner.py neural/nf_run.py tests/test_tft_runner.py requirements.txt
git commit -m "feat(tft): orchestrate user-run next-day training"
```

---

### Task 8: Document, verify, and stop before training

**Files:**
- Modify: `README.md`
- Modify: `RESULTS.md`
- Modify: `neural/__init__.py`
- Test: all `tests/test_*.py`

**Interfaces:**
- Consumes: completed Tasks 1-7.
- Produces: accurate user documentation, one user-operated PowerShell command, and final verification evidence without a trained TFT run.

- [ ] **Step 1: Update documentation without inventing results**

In `README.md`, add a “Next-day TFT experiment” section that states:

- 15-minute inputs, 672-bar lookback, 96-bar/24-hour horizon;
- fixed-model 365-origin evaluation with 219 calibration and 146 test origins;
- full training is intentionally user-operated;
- exact command with `<safe-run-id>` and a batch-size note for GPU memory;
- output-directory contents and how to identify `status.json` as complete;
- no buy/sell system or P&L claim exists in this phase.

In `RESULTS.md`, retain the old two-hour TFT numbers only under an explicit “legacy 2-hour experiment” label and state that they are not comparable to the new next-day configuration. Add a “next-day TFT: pending user run” section with no metric placeholders or estimated values. Update `neural/__init__.py` so it no longer says TFT is unbuilt.

- [ ] **Step 2: Run the complete test suite**

Run: `python -m unittest discover -s tests -v`

Expected: all existing 49 tests plus new TFT tests PASS.

- [ ] **Step 3: Run static and whitespace verification**

Run: `python -m compileall crypto neural tests`

Expected: exit 0.

Run: `git diff --check`

Expected: no output and exit 0.

- [ ] **Step 4: Run the real-data causality check without training**

```powershell
python -c "import pandas as pd; from neural.tft_data import assert_tft_feature_causality; raw=pd.read_parquet('data/ohlcv_15m.parquet'); cutoff=raw.date.max()-pd.Timedelta('7D'); assert_tft_feature_causality(raw, cutoff=cutoff, as_of=pd.Timestamp.now(tz='UTC')); print('TFT causality: PASS')"
```

Expected: `TFT causality: PASS`. This may build features twice but must not construct or train TFT.

- [ ] **Step 5: Verify no training artifacts or trading code were added**

Run: `git status --short`

Expected before the documentation commit: only the three documented files are modified.

Run: `git diff --name-only a6dd939..HEAD | Select-String -Pattern "_cv_tft|weights.ckpt|predictions.*parquet"`

Expected: no generated TFT prediction or checkpoint file is tracked.

Run: `rg -n -i "buy|sell|position|pnl|sharpe|slippage|order" neural/tft_*.py`

Expected: no trading implementation; matches are allowed only in explicit non-goal/error text and must be reviewed manually.

- [ ] **Step 6: Commit documentation and final verification state**

```powershell
git add README.md RESULTS.md neural/__init__.py
git commit -m "docs: explain user-run next-day TFT experiment"
```

- [ ] **Step 7: Report the handoff without running training**

Report the final test count, causality result, branch commit, exact user command, expected output directory, and confirmation that Codex did not train TFT. Do not quote new TFT performance until the user's completed run passes manifest validation.
