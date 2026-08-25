# Daily Prediction Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reproducible out-of-sample daily metrics and portfolio-quality charts, then evaluate narrowly scoped feature and ensemble candidates without test-set tuning.

**Architecture:** Preserve the existing feature, calibration, and walk-forward interfaces. Add a small evaluation module that validates and stores forecast rows, computes breakdown tables, and renders charts; model-family runners call it after producing out-of-sample predictions. Candidate features and blends are evaluated through the same folds and retained only when they improve pinball loss without moving q10–q90 coverage outside 78–82%.

**Tech Stack:** Python 3.12, pandas, NumPy, matplotlib, unittest, Parquet/CSV

**Spec:** `docs/superpowers/specs/2026-08-25-prediction-quality-design.md`

## Global Constraints

- Daily, hourly, and 15-minute results remain separate.
- Fit, calibration, and test partitions are chronological; test data cannot select features, calibration, or blend weights.
- Every output identifies dataset cutoff, horizons, fold count, origin count, configuration, and model.
- Headline success means lower walk-forward pinball loss than `vol_21d` while maintaining 78–82% q10–q90 coverage.
- No new architecture or dependency is added without a measured failure that requires it.
- Generated charts use only out-of-sample predictions and show the baseline.

---

### Task 1: Validate and store out-of-sample prediction rows

**Files:**
- Create: `crypto/evaluation.py`
- Create: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: a `pandas.DataFrame` with `model`, `asset`, `origin`, `fold`, `h`, `y`, `last`, `q10`, `q50`, and `q90`.
- Produces: `validate_predictions(frame) -> None` and `save_predictions(frame, output_dir, metadata) -> pathlib.Path`.

- [ ] **Step 1: Write failing validation tests**

```python
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from crypto.evaluation import save_predictions, validate_predictions


def valid_frame():
    return pd.DataFrame({
        "model": ["xgb"], "asset": ["BTC"],
        "origin": [pd.Timestamp("2025-01-01", tz="UTC")],
        "fold": [pd.Timestamp("2025-01-01", tz="UTC")],
        "h": [1], "y": [101.0], "last": [100.0],
        "sigma": [0.02], "rv": [0.018],
        "q10": [95.0], "q50": [100.0], "q90": [105.0],
    })


class PredictionValidationTests(unittest.TestCase):
    def test_valid_predictions_are_accepted(self):
        validate_predictions(valid_frame())

    def test_crossed_quantiles_are_rejected(self):
        frame = valid_frame().assign(q10=106.0)
        with self.assertRaisesRegex(ValueError, "crossed quantiles"):
            validate_predictions(frame)

    def test_duplicate_forecast_keys_are_rejected(self):
        frame = pd.concat([valid_frame(), valid_frame()], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate forecast keys"):
            validate_predictions(frame)

    def test_save_writes_predictions_and_metadata_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = save_predictions(valid_frame(), Path(tmp) / "run-1", {"pipeline": "daily"})
            self.assertTrue((out / "predictions.parquet").exists())
            self.assertTrue((out / "metadata.json").exists())
            with self.assertRaises(FileExistsError):
                save_predictions(valid_frame(), Path(tmp) / "run-1", {"pipeline": "daily"})
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_evaluation -v`

Expected: FAIL because `crypto.evaluation` does not exist.

- [ ] **Step 3: Implement the minimal validation and storage module**

```python
import json
from pathlib import Path

import pandas as pd

REQUIRED = ["model", "asset", "origin", "fold", "h", "y", "last", "sigma", "rv", "q10", "q50", "q90"]
KEY = ["model", "asset", "origin", "h"]


def validate_predictions(frame):
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"missing prediction columns: {missing}")
    if frame[REQUIRED].isna().any().any():
        raise ValueError("NaN predictions")
    if frame.duplicated(KEY).any():
        raise ValueError("duplicate forecast keys")
    if ((frame.q10 > frame.q50) | (frame.q50 > frame.q90)).any():
        raise ValueError("crossed quantiles")
    if not frame.h.between(1, 7).all():
        raise ValueError("mismatched horizons")


def save_predictions(frame, output_dir, metadata):
    validate_predictions(frame)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return output_dir
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_evaluation -v`

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add crypto/evaluation.py tests/test_evaluation.py
git commit -m "feat: validate daily forecast artifacts"
```

---

### Task 2: Preserve origin and fold provenance in daily backtests

**Files:**
- Modify: `tree/run.py:29-61`
- Modify: `neural/core.py:101-135`
- Modify: `crypto/evaluation.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `start` yielded by `crypto.backtest.run_folds` and `test.date` forecast origins.
- Produces: `default_run_dir(tag, data_end, run_id, root=Path("artifacts/evaluation")) -> Path` and persisted tree/neural prediction bundles.

- [ ] **Step 1: Add a failing deterministic run-directory test**

```python
from crypto.evaluation import default_run_dir

def test_run_directory_contains_pipeline_and_data_cutoff(self):
    got = default_run_dir("tree", pd.Timestamp("2026-07-23", tz="UTC"), "baseline", Path("out"))
    self.assertEqual(got, Path("out/daily-tree-20260723-baseline"))
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_evaluation.PredictionValidationTests.test_run_directory_contains_pipeline_and_data_cutoff -v`

Expected: FAIL because `default_run_dir` is missing.

- [ ] **Step 3: Implement the directory helper**

```python
def default_run_dir(tag, data_end, run_id, root=Path("artifacts/evaluation")):
    stamp = pd.Timestamp(data_end).strftime("%Y%m%d")
    return Path(root) / f"daily-{tag}-{stamp}-{run_id}"
```

- [ ] **Step 4: Add provenance columns to each family’s records**

In both daily backtest loops, retain `start`:

```python
for train, test, start in run_folds(feat, cols):
```

Add these fields to every result frame:

```python
"origin": test.date.to_numpy(),
"fold": np.repeat(start, len(test)),
"sigma": s_te,
"rv": test[f"rv{h}"].to_numpy(),
```

After scoring, save the result with configuration metadata:

```python
from crypto.evaluation import default_run_dir, save_predictions

save_predictions(res, default_run_dir("tree", feat.date.max(), "baseline"), {
    "pipeline": "daily", "family": "tree", "data_end": feat.date.max(),
    "horizons": H, "folds": res.fold.nunique(),
    "origins": res[["asset", "origin"]].drop_duplicates().shape[0],
    "features": cols,
})
```

Use `family="neural"`, the neural model names, `LOOKBACK`, and `CHANNELS` in the neural metadata.

- [ ] **Step 5: Run focused tests and compile affected modules**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_evaluation -v`

Expected: all evaluation tests pass.

Run: `.\.venv\Scripts\python.exe -m compileall -q crypto tree neural`

Expected: exit code 0.

- [ ] **Step 6: Commit**

```powershell
git add crypto/evaluation.py tests/test_evaluation.py tree/run.py neural/core.py
git commit -m "feat: retain daily forecast provenance"
```

---

### Task 3: Generate honest metric breakdowns

**Files:**
- Modify: `crypto/evaluation.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: validated prediction rows.
- Produces: `metric_tables(frame) -> dict[str, pandas.DataFrame]` with `overall`, `by_horizon`, `by_asset`, and `by_fold` tables.

- [ ] **Step 1: Add failing literal-metric tests**

```python
from crypto.evaluation import metric_tables

def test_metric_tables_report_coverage_by_horizon_and_asset(self):
    frame = pd.concat([
        valid_frame(),
        valid_frame().assign(asset="ETH", y=110.0, origin=pd.Timestamp("2025-01-02", tz="UTC")),
    ], ignore_index=True)
    tables = metric_tables(frame)
    self.assertEqual(tables["overall"].loc["xgb", "coverage"], 50.0)
    self.assertEqual(tables["by_asset"].loc[("xgb", "BTC"), "coverage"], 100.0)
    self.assertEqual(tables["by_asset"].loc[("xgb", "ETH"), "coverage"], 0.0)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_evaluation.PredictionValidationTests.test_metric_tables_report_coverage_by_horizon_and_asset -v`

Expected: FAIL because `metric_tables` is missing.

- [ ] **Step 3: Implement grouped metrics by reusing shared scoring logic**

```python
from crypto.backtest import score


def _group_score(frame, keys):
    rows = []
    for values, group in frame.groupby(keys, sort=False):
        values = values if isinstance(values, tuple) else (values,)
        row = score(group).reset_index().iloc[0].to_dict()
        rows.append(dict(zip(keys, values)) | row)
    return pd.DataFrame(rows).set_index(keys)


def metric_tables(frame):
    validate_predictions(frame)
    return {
        "overall": score(frame),
        "by_horizon": _group_score(frame, ["model", "h"]),
        "by_asset": _group_score(frame, ["model", "asset"]),
        "by_fold": _group_score(frame, ["model", "fold"]),
    }
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_evaluation -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add crypto/evaluation.py tests/test_evaluation.py
git commit -m "feat: break down daily forecast metrics"
```

---

### Task 4: Render the portfolio evaluation bundle

**Files:**
- Create: `evaluate.py`
- Create: `crypto/evaluation_plots.py`
- Create: `tests/test_evaluation_plots.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: one or more saved `predictions.parquet` files.
- Produces: CSV metric tables plus `forecast_bands.png`, `coverage_by_horizon.png`, `pinball_by_model.png`, `volatility_fit.png`, and `performance_by_asset.png`.

- [ ] **Step 1: Write a failing chart-generation smoke test**

```python
import tempfile
import unittest
from pathlib import Path

from crypto.evaluation_plots import render_bundle
from tests.test_evaluation import valid_frame


class EvaluationPlotTests(unittest.TestCase):
    def test_render_bundle_creates_all_expected_charts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            render_bundle(valid_frame(), out)
            expected = {
                "forecast_bands.png", "coverage_by_horizon.png",
                "pinball_by_model.png", "volatility_fit.png",
                "performance_by_asset.png",
            }
            self.assertEqual(expected, {p.name for p in out.glob("*.png")})
```

- [ ] **Step 2: Run the chart test and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_evaluation_plots -v`

Expected: FAIL because `crypto.evaluation_plots` does not exist.

- [ ] **Step 3: Implement five focused matplotlib charts**

Implement `render_bundle(frame, output_dir)` using the Agg backend. Each chart must:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

- plot only rows supplied by the validated out-of-sample frame;
- include model/baseline labels, origin count, and date range in title or subtitle;
- use normalized `pinball_%` for cross-asset comparisons;
- show the 80% target as a horizontal line on coverage charts;
- close every figure after saving.

- [ ] **Step 4: Implement the evaluation CLI**

```python
import argparse
from pathlib import Path

import pandas as pd

from crypto.evaluation import metric_tables, validate_predictions
from crypto.evaluation_plots import render_bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    frame = pd.concat([pd.read_parquet(p) for p in args.predictions], ignore_index=True)
    validate_predictions(frame)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, table in metric_tables(frame).items():
        table.to_csv(args.out / f"metrics_{name}.csv")
    render_bundle(frame, args.out)
```

- [ ] **Step 5: Run tests and a fixture CLI smoke test**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_evaluation tests.test_evaluation_plots -v`

Expected: all tests pass and all figures are closed.

- [ ] **Step 6: Document exact usage**

Add to `README.md`:

```powershell
.\.venv\Scripts\python.exe evaluate.py artifacts/evaluation/daily-tree-YYYYMMDD/predictions.parquet artifacts/evaluation/daily-neural-YYYYMMDD/predictions.parquet --out artifacts/reports/daily-YYYYMMDD
```

- [ ] **Step 7: Commit**

```powershell
git add evaluate.py crypto/evaluation_plots.py tests/test_evaluation_plots.py README.md
git commit -m "feat: render daily evaluation report"
```

---

### Task 5: Add causal regime features and feature-group ablations

**Files:**
- Modify: `crypto/features.py:13-104`
- Create: `experiments/feature_ablation.py`
- Create: `tests/test_features.py`
- Create: `tests/test_feature_ablation.py`

**Interfaces:**
- Consumes: the existing daily OHLCV/funding feature frame.
- Produces: `feature_groups(columns) -> dict[str, list[str]]`, causal `vol_regime`, `drawdown_63d`, and `volume_z21` features, and an ablation result table evaluated through existing folds.

- [ ] **Step 1: Write failing feature tests**

```python
import unittest

import numpy as np
import pandas as pd

from crypto.features import check_causal, feature_groups, make_features


class RegimeFeatureTests(unittest.TestCase):
    def test_regime_features_are_present_and_causal(self):
        n = 500
        df = pd.DataFrame({
            "asset": "BTC", "date": pd.date_range("2024-01-01", periods=n, tz="UTC"),
            "open": np.arange(n) + 100.0, "high": np.arange(n) + 102.0,
            "low": np.arange(n) + 99.0, "close": np.arange(n) + 101.0,
            "volume": np.arange(n) + 1000.0,
        })
        feat = make_features(df)
        for column in ["vol_regime", "drawdown_63d", "volume_z21"]:
            self.assertIn(column, feat.columns)
        check_causal(df, ["vol_regime", "drawdown_63d", "volume_z21"])

    def test_feature_groups_cover_each_feature_once(self):
        groups = feature_groups(["ret_lag1", "vol_21d", "range", "btc_ret", "fund_7d"])
        flattened = [column for columns in groups.values() for column in columns]
        self.assertCountEqual(flattened, ["ret_lag1", "vol_21d", "range", "btc_ret", "fund_7d"])
```

- [ ] **Step 2: Run feature tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_features -v`

Expected: FAIL because the new columns and `feature_groups` are missing.

- [ ] **Step 3: Implement the three backward-looking features**

Inside each asset group in `make_features`:

```python
g["vol_regime"] = g["vol_21d"] / g["vol_63d"].clip(lower=1e-6)
g["drawdown_63d"] = g.close / g.close.rolling(63).max() - 1
lv = np.log(g.volume.replace(0, np.nan))
g["volume_z21"] = (lv - lv.rolling(21).mean()) / lv.rolling(21).std()
```

Extend `PREFIXES` with `("drawdown_", "volume_")` and implement mutually exclusive groups:

```python
def feature_groups(columns):
    rules = {
        "returns": ("ret_", "px_", "drawdown_"),
        "volatility": ("vol_", "range", "har_"),
        "volume": ("volume_",),
        "market": ("btc_",),
        "funding": ("fund",),
    }
    return {name: [c for c in columns if c.startswith(prefixes)]
            for name, prefixes in rules.items()}
```

Add an assertion that the flattened groups equal the supplied feature list without duplicates.

- [ ] **Step 4: Run feature and full causality tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_features -v`

Expected: all feature tests pass.

Run: `.\.venv\Scripts\python.exe -c "import pandas as pd; from crypto.data import OHLCV_OUT; from crypto.features import check_causal; check_causal(pd.read_parquet(OHLCV_OUT)); print('PASS')"`

Expected: `PASS`.

- [ ] **Step 5: Implement the ablation experiment with a fixed baseline**

`experiments/feature_ablation.py` must evaluate:

```python
candidates = {"all": cols}
for group, members in feature_groups(cols).items():
    candidates[f"without_{group}"] = [c for c in cols if c not in members]
```

Use the existing `run_folds`, `split_calibration`, XGBoost fitter, calibration,
and `score` functions. Save one row per candidate/fold with `pinball_%`, coverage,
origin count, and data cutoff. Do not choose a winner inside the script.

- [ ] **Step 6: Add a test that every ablation removes exactly its named group**

```python
def test_ablation_candidates_remove_only_named_group(self):
    cols = ["ret_lag1", "vol_21d", "volume_z21", "btc_ret", "fund_7d"]
    candidates = ablation_candidates(cols)
    self.assertEqual(candidates["without_volume"], ["ret_lag1", "vol_21d", "btc_ret", "fund_7d"])
```

- [ ] **Step 7: Run tests and commit**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_features tests.test_feature_ablation -v`

Expected: all tests pass.

```powershell
git add crypto/features.py experiments/feature_ablation.py tests/test_features.py tests/test_feature_ablation.py
git commit -m "feat: evaluate causal regime features"
```

---

### Task 6: Add a calibration-selected daily ensemble

**Files:**
- Create: `crypto/ensemble.py`
- Create: `tests/test_ensemble.py`
- Modify: `tree/run.py`
- Modify: `RESULTS.md`

**Interfaces:**
- Consumes: aligned calibration predictions from XGBoost and LightGBM, keyed by `asset`, `origin`, and `h`.
- Produces: `select_weight(left, right, grid=None) -> float` and `blend_predictions(left, right, weight, model="tree_blend") -> pandas.DataFrame`.

- [ ] **Step 1: Write failing blend tests**

```python
import unittest

from crypto.ensemble import blend_predictions, select_weight
from tests.test_evaluation import valid_frame


class EnsembleTests(unittest.TestCase):
    def test_blend_is_key_aligned_and_preserves_quantile_order(self):
        left = valid_frame().assign(model="xgb", q10=90.0, q50=100.0, q90=110.0)
        right = valid_frame().assign(model="lgbm", q10=94.0, q50=102.0, q90=114.0)
        got = blend_predictions(left, right, 0.25)
        self.assertEqual(got.q50.iloc[0], 101.5)
        self.assertTrue((got.q10 <= got.q50).all() and (got.q50 <= got.q90).all())

    def test_weight_selection_uses_only_supplied_calibration_rows(self):
        left = valid_frame().assign(model="xgb", q50=101.0)
        right = valid_frame().assign(model="lgbm", q50=110.0)
        self.assertEqual(select_weight(left, right, grid=[0.0, 1.0]), 1.0)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_ensemble -v`

Expected: FAIL because `crypto.ensemble` does not exist.

- [ ] **Step 3: Implement key-aligned quantile blending**

Use keys `asset`, `origin`, `fold`, and `h`. Reject missing or duplicate keys.
Blend q10/q50/q90 as `weight * left + (1 - weight) * right`, preserve `y` and
`last` only after asserting they match, and validate the result with
`validate_predictions`.

- [ ] **Step 4: Implement calibration-only weight selection**

For each weight in `grid or np.linspace(0, 1, 21)`, call `blend_predictions` on
the supplied calibration frames and select the weight with minimum `pinball_%`
from `score`. Resolve ties toward `0.5`; do not inspect test rows.

- [ ] **Step 5: Integrate the blend into each tree fold**

In `tree/run.py`, retain calibrated price-band frames for XGBoost and LightGBM,
select the weight using only the fold’s `cal` predictions, and apply that fixed
weight to the fold’s `test` predictions. Append `tree_blend` to the same result
frame and persist the selected weight in fold metadata.

- [ ] **Step 6: Run unit tests and the tree walk-forward experiment**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_ensemble tests.test_evaluation -v`

Expected: all tests pass.

Run: `.\.venv\Scripts\python.exe -m tree.run`

Expected: a new daily tree evaluation bundle containing `xgb`, `lgbm`,
`vol_21d`, and `tree_blend`, with no validation errors.

- [ ] **Step 7: Apply the acceptance gate**

Compare overall and worst-fold `pinball_%` with XGBoost and `vol_21d`. Retain
`tree_blend` in `RESULTS.md` only if it lowers overall pinball loss and remains
within 78–82% coverage without materially worsening the worst fold. Otherwise,
document it as a rejected experiment without changing the headline result.

- [ ] **Step 8: Commit**

```powershell
git add crypto/ensemble.py tests/test_ensemble.py tree/run.py RESULTS.md
git commit -m "feat: evaluate calibrated tree ensemble"
```

---

### Task 7: Final daily evaluation and portfolio handoff

**Files:**
- Modify: `README.md`
- Modify: `RESULTS.md`
- Create: `docs/evaluation-methodology.md`
- Generate: `docs/evaluation/*.png`
- Generate: `docs/evaluation/*.csv`

**Interfaces:**
- Consumes: final saved tree and neural out-of-sample prediction bundles.
- Produces: one documented, reproducible daily report and updated resume-safe metrics.

- [ ] **Step 1: Run the complete verification suite**

Run: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass with zero errors.

Run: `.\.venv\Scripts\python.exe -m compileall -q crypto tree linear neural experiments fetch.py predict.py evaluate.py`

Expected: exit code 0.

- [ ] **Step 2: Rebuild only models affected by retained feature changes**

If Task 5’s new full feature set wins the acceptance gate, run:

```powershell
.\.venv\Scripts\python.exe -m tree.lgbm
.\.venv\Scripts\python.exe -m tree.xgb
```

If it does not win, revert only the candidate feature additions and keep the
existing artifacts. Neural models require retraining only if their raw sequence
channels or training code changed; this plan does not change them.

- [ ] **Step 3: Generate the final report without overwriting prior runs**

```powershell
.\.venv\Scripts\python.exe evaluate.py artifacts/evaluation/daily-tree-YYYYMMDD/predictions.parquet artifacts/evaluation/daily-neural-YYYYMMDD/predictions.parquet --out docs/evaluation/daily-YYYYMMDD
```

Expected: five PNG charts and four CSV metric tables, all labeled with sample size
and date range.

- [ ] **Step 4: Document exact provenance and caveats**

`docs/evaluation-methodology.md` must state data cutoff, assets, fold dates,
origin count, model configurations, fit/calibration/test boundaries, baselines,
accepted/rejected candidates, and the exact command used to reproduce the report.

- [ ] **Step 5: Update public results using only the final bundle**

Replace daily values in `README.md` and `RESULTS.md` only when they match the
generated CSV tables. Keep high-frequency results in their separate section.

- [ ] **Step 6: Commit generated evidence and documentation**

```powershell
git add README.md RESULTS.md docs/evaluation docs/evaluation-methodology.md
git commit -m "docs: publish reproducible daily evaluation"
```
