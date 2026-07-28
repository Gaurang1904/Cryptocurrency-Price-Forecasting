"""Per-coin model comparison. One model per (coin, horizon), scored identically.

Every candidate predicts the same quantity - daily volatility over the next h
days - and every prediction is turned into a price interval by the same
calibration. So pinball loss compares them directly.

Run: python -m experiments.model_zoo
"""

import numpy as np
import pandas as pd

from crypto.backtest import H, log, run_folds, score
from crypto.data import OHLCV_OUT
from crypto.features import build
from crypto.model import bands, calibrate, clip_sigma, split_calibration
from linear.adapter import HAR_COLS
from linear.model import fit_garch, fit_har, fit_ridge, garch_sigma
from tree.lgbm import fit_vol

REPORT = "report_by_coin.csv"


def sigmas(fit, cal, test, cols, h, garch=None):
    """{model: (sigma_cal, sigma_test)} - calibration sigma must be out-of-sample."""
    out = {"vol_21d": (cal.vol_21d.to_numpy(), test.vol_21d.to_numpy())}

    har = fit_har(fit, h)
    out["har_rv"] = (clip_sigma(np.exp(har.predict(cal[HAR_COLS]))),
                     clip_sigma(np.exp(har.predict(test[HAR_COLS]))))

    ridge = fit_ridge(fit, cols, h)
    out["ridge"] = (clip_sigma(np.exp(ridge.predict(cal[cols].fillna(0)))),
                    clip_sigma(np.exp(ridge.predict(test[cols].fillna(0)))))

    lgbm = fit_vol(fit, cols, h)
    out["lightgbm"] = (clip_sigma(np.exp(lgbm.predict(cal[cols]))),
                       clip_sigma(np.exp(lgbm.predict(test[cols]))))

    if garch is not None:
        out["garch"] = (garch.reindex(cal.date).to_numpy(),
                        garch.reindex(test.date).to_numpy())
    return out


if __name__ == "__main__":
    feat, cols = build(pd.read_parquet(OHLCV_OUT))
    recs, vol_err = [], []

    for asset, g in feat.groupby("asset", sort=False):
        print(f"--- {asset}")
        ret = pd.Series(np.log(g.close).diff().to_numpy(), index=g.date.to_numpy()).dropna()

        for train, test, start in run_folds(g, cols):
            train = train.dropna(subset=[f"rv{h}" for h in range(1, H + 1)] + HAR_COLS)
            fit, cal = split_calibration(train)
            if len(fit) < 200 or test.empty:
                continue

            # GARCH params from the fit window only, then refilter the whole
            # series so every origin gets its own conditional variance.
            params = fit_garch(ret[ret.index < fit.date.max()]).params

            for h in range(1, H + 1):
                last = test.close.to_numpy()
                y = last * np.exp(test[f"y{h}"].to_numpy())
                truth = np.log(test[f"rv{h}"].to_numpy())
                gs = garch_sigma(ret, params, h)

                for name, (s_cal, s_te) in sigmas(fit, cal, test, cols, h, gs).items():
                    z = calibrate(cal[f"y{h}"].to_numpy(), s_cal, h)
                    recs.append(pd.DataFrame({
                        "model": name, "asset": asset, "h": h, "y": y, "last": last,
                        **{f"q{int(q * 100)}": v
                           for q, v in bands(z, last, s_te, h).items()},
                    }))
                    vol_err.append({"asset": asset, "model": name, "h": h,
                                    "mae": np.nanmean(np.abs(truth - np.log(s_te)))})

    res = pd.concat(recs, ignore_index=True)

    print("\nVOLATILITY MAE on log realised vol, by coin (lower better)")
    vol = (pd.DataFrame(vol_err).pivot_table(index="model", columns="asset", values="mae"))
    print(vol.round(4).to_string())

    print("\nPINBALL % of price by coin (lower better)")
    per_coin = pd.concat([score(g).assign(asset=a) for a, g in res.groupby("asset")])
    pin = per_coin.reset_index().pivot(index="model", columns="asset", values="pinball_%")
    print(pin.round(2).to_string())

    print("\nCOVERAGE % by coin (target 80)")
    cov = per_coin.reset_index().pivot(index="model", columns="asset", values="coverage")
    print(cov.round(1).to_string())

    print("\nWINNER by coin (lowest pinball)")
    print(pin.idxmin().to_frame("best").to_string())
    print(f"\nmargin over 2nd place, %: "
          f"{((pin.apply(sorted).iloc[1] / pin.min() - 1) * 100).round(1).to_dict()}")

    per_coin.reset_index().to_csv(REPORT, index=False)
    print(f"\nwrote {REPORT}")
    log("model_zoo", score(res), origins=len(res) // (5 * H), assets=res.asset.nunique())
