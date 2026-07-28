"""Shared training orchestration: fit every horizon, calibrate, save, plot.

Each family's entry file (tree/lgbm.py, tree/xgb.py, ...) is thin - it defines
its fitter and calls train_and_save. This keeps the fit/calibrate/save/plot
logic in ONE place instead of copied per model.
"""

from pathlib import Path

import joblib
import pandas as pd

from crypto.data import OHLCV_OUT
from crypto.features import H, build, check_causal
from crypto.model import QUANTILES, fit_all

MODELS = Path("models")
DEPLOYED = "lgbm"  # the model predict.py loads as models/vol7d.joblib


def artifact_path(name):
    """The deployed model keeps the canonical name; others get a suffix."""
    return MODELS / ("vol7d.joblib" if name == DEPLOYED else f"vol7d_{name}.joblib")


def train_and_save(fit_one, name):
    df = pd.read_parquet(OHLCV_OUT)
    check_causal(df)  # cheap insurance: refuse to train on leaking features

    feat, cols = build(df)
    train = feat.dropna(subset=cols + [f"rv{h}" for h in range(1, H + 1)]
                               + [f"y{h}" for h in range(1, H + 1)])
    models, z, n_fit, n_cal = fit_all(train, cols, fit_one)

    art = {"models": models, "z": z, "cols": cols, "H": H, "quantiles": QUANTILES,
           "model": name, "fit_rows": n_fit, "cal_rows": n_cal,
           "trained_at": pd.Timestamp.now(tz="UTC"), "data_end": feat.date.max()}

    out = artifact_path(name)
    out.parent.mkdir(exist_ok=True)
    joblib.dump(art, out)

    print(f"[{name}] trained on {n_fit} rows, calibrated on {n_cal}")
    print(f"{len(cols)} features, {df.asset.nunique()} assets, data through {art['data_end'].date()}")
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)\n")
    print("calibration z-table (sigmas of trailing vol)")
    print(pd.DataFrame(z).T.round(3).to_string())

    from crypto.plots import train_split
    print(f"\nwrote {train_split(feat, cols, models, out=Path('plots') / f'train_split_{name}.svg')}")
    return art
