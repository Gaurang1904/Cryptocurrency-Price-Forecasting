"""Backtest the tree family against its baseline. python -m tree.run

Every tree model (LightGBM, XGBoost) plus vol_21d, same folds, same scoring.
Add a model by dropping one entry in FITTERS.
"""

import numpy as np
import pandas as pd

from crypto.backtest import H, coverage_by_h, log, run_folds, score
from crypto.data import OHLCV_OUT
from crypto.evaluation import default_run_dir, save_predictions
from crypto.features import build
from crypto.model import bands, calibrate, clip_sigma, split_calibration
from tree.adapter import inputs
from tree import lgbm, xgb

FITTERS = {"lgbm": lgbm.fit_vol, "xgb": xgb.fit_vol}


def model_sigma(fit_one, fit, cal, test, cols, h):
    m = fit_one(fit, cols, h)
    return (clip_sigma(np.exp(m.predict(inputs(cal, cols)))),
            clip_sigma(np.exp(m.predict(inputs(test, cols)))))


def backtest(fitters=FITTERS):
    feat, cols = build(pd.read_parquet(OHLCV_OUT))

    recs, vol_err = [], []
    for train, test, start in run_folds(feat, cols):
        train = train.dropna(subset=[f"rv{h}" for h in range(1, H + 1)])
        # Calibration must see OUT-OF-SAMPLE sigma: trees fit their training rows
        # too well, so an in-sample z-table comes out far too narrow.
        fit, cal = split_calibration(train)
        last = test.close.to_numpy()

        for h in range(1, H + 1):
            y = last * np.exp(test[f"y{h}"].to_numpy())
            truth = np.log(test[f"rv{h}"].to_numpy())
            sig = {
                name: model_sigma(f, fit, cal, test, cols, h)
                for name, f in fitters.items()
            }
            sig["vol_21d"] = (cal.vol_21d.to_numpy(), test.vol_21d.to_numpy())

            for name, (s_cal, s_te) in sig.items():
                z = calibrate(cal[f"y{h}"].to_numpy(), s_cal, h)
                recs.append(pd.DataFrame({
                    "model": name, "asset": test.asset.values, "origin": test.date.to_numpy(),
                    "fold": np.repeat(start, len(test)), "h": h, "y": y, "last": last,
                    "sigma": s_te, "rv": test[f"rv{h}"].to_numpy(),
                    "regime_driver": test.vol_21d.to_numpy(),
                    **{f"q{int(q * 100)}": v for q, v in bands(z, last, s_te, h).items()},
                }))
                vol_err.append({"model": name, "h": h,
                                "mae": np.nanmean(np.abs(truth - np.log(s_te)))})

    res = pd.concat(recs, ignore_index=True)
    save_predictions(res, default_run_dir("tree", feat.date.max(), "baseline"), {
        "pipeline": "daily", "family": "tree", "data_end": feat.date.max(),
        "horizons": H, "folds": res.fold.nunique(),
        "origins": res[["asset", "origin"]].drop_duplicates().shape[0],
        "features": cols,
    })
    res.attrs["feature_count"] = len(cols)
    return res, pd.DataFrame(vol_err)


if __name__ == "__main__":
    res, vol_err = backtest()
    summary = score(res)
    origins = res[["asset", "origin"]].drop_duplicates().shape[0]
    print(f"{res.attrs['feature_count']} features, {origins} origins\n")
    print("VOLATILITY MAE on log realised vol (lower better)")
    print(vol_err.pivot_table(index="h", columns="model", values="mae").round(4).to_string())
    print("\nRESULTING PRICE INTERVALS")
    print(summary.round(2).to_string())
    print("\ncoverage % by horizon day (target 80)")
    print(coverage_by_h(res).round(1).to_string())
    log("tree", summary, origins=origins, features=res.attrs["feature_count"])
