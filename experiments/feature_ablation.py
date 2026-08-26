"""Explore XGBoost feature-group ablations on the standard expanding folds.

These folds are the feature-selection sample, not untouched post-selection
validation. Run: python -m experiments.feature_ablation
"""

from pathlib import Path

import numpy as np
import pandas as pd

from crypto.backtest import H, run_folds, score
from crypto.data import OHLCV_OUT
from crypto.features import build, feature_groups
from crypto.model import bands, calibrate, clip_sigma, split_calibration
from tree.adapter import inputs
from tree.xgb import fit_vol

OUT = Path("artifacts/feature_ablation.csv")


def ablation_candidates(cols):
    candidates = {
        "all": cols,
        "legacy": [
            c for c in cols
            if c not in {"vol_regime", "drawdown_63d", "volume_z21"}
        ],
    }
    for group, members in feature_groups(cols).items():
        candidates[f"without_{group}"] = [c for c in cols if c not in members]
    return candidates


def run_ablation(feat, cols):
    """Return exploratory selection-set scores for each candidate and fold."""
    rows = []
    for name, candidate_cols in ablation_candidates(cols).items():
        for train, test, start in run_folds(feat, candidate_cols):
            train = train.dropna(subset=[f"rv{h}" for h in range(1, H + 1)])
            fit, cal = split_calibration(train)
            recs = []
            for h in range(1, H + 1):
                model = fit_vol(fit, candidate_cols, h)
                cal_sigma = clip_sigma(np.exp(model.predict(inputs(cal, candidate_cols))))
                test_sigma = clip_sigma(np.exp(model.predict(inputs(test, candidate_cols))))
                last = test.close.to_numpy()
                recs.append(pd.DataFrame({
                    "model": name, "y": last * np.exp(test[f"y{h}"].to_numpy()), "last": last,
                    **{f"q{int(q * 100)}": values
                       for q, values in bands(calibrate(cal[f"y{h}"], cal_sigma, h),
                                              last, test_sigma, h).items()},
                }))
            metrics = score(pd.concat(recs, ignore_index=True)).loc[name]
            rows.append({"candidate": name, "fold": start, "pinball_%": metrics["pinball_%"],
                         "coverage": metrics["coverage"],
                         "origin_count": test[["asset", "date"]].drop_duplicates().shape[0],
                         "data_cutoff": feat.date.max()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    feat, cols = build(
        pd.read_parquet(OHLCV_OUT), include_candidates=True
    )
    table = run_ablation(feat, cols)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"wrote {len(table)} exploratory selection-set fold rows to {OUT}")
