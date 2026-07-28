"""Backtest the linear family per coin. python -m linear.run

HAR-RV, Ridge, GARCH(1,1) vs vol_21d. All three are per-series, so this loops
over coins and fits each one on that coin's own history.
"""

import numpy as np
import pandas as pd

from crypto.backtest import H, log, run_folds, score
from crypto.data import OHLCV_OUT
from crypto.features import build
from crypto.model import bands, calibrate, clip_sigma, split_calibration
from linear.adapter import HAR_COLS, ridge_pipeline
from linear.model import fit_garch, fit_har, fit_ridge, garch_sigma


def sigmas(fit, cal, test, cols, h, garch):
    out = {"vol_21d": (cal.vol_21d.to_numpy(), test.vol_21d.to_numpy())}
    har = fit_har(fit, h)
    out["har_rv"] = (clip_sigma(np.exp(har.predict(cal[HAR_COLS]))),
                     clip_sigma(np.exp(har.predict(test[HAR_COLS]))))
    ridge = fit_ridge(fit, cols, h)
    out["ridge"] = (clip_sigma(np.exp(ridge.predict(cal[cols].fillna(0)))),
                    clip_sigma(np.exp(ridge.predict(test[cols].fillna(0)))))
    out["garch"] = (garch.reindex(cal.date).to_numpy(), garch.reindex(test.date).to_numpy())
    return out


if __name__ == "__main__":
    feat, cols = build(pd.read_parquet(OHLCV_OUT))
    recs = []

    for asset, g in feat.groupby("asset", sort=False):
        print(f"--- {asset}")
        ret = pd.Series(np.log(g.close).diff().to_numpy(), index=g.date.to_numpy()).dropna()
        for train, test, _ in run_folds(g, cols):
            train = train.dropna(subset=[f"rv{h}" for h in range(1, H + 1)] + HAR_COLS)
            fit, cal = split_calibration(train)
            if len(fit) < 200 or test.empty:
                continue
            params = fit_garch(ret[ret.index < fit.date.max()]).params

            for h in range(1, H + 1):
                last = test.close.to_numpy()
                y = last * np.exp(test[f"y{h}"].to_numpy())
                gs = garch_sigma(ret, params, h)
                for name, (s_cal, s_te) in sigmas(fit, cal, test, cols, h, gs).items():
                    z = calibrate(cal[f"y{h}"].to_numpy(), s_cal, h)
                    recs.append(pd.DataFrame({
                        "model": name, "asset": asset, "h": h, "y": y, "last": last,
                        **{f"q{int(q * 100)}": v for q, v in bands(z, last, s_te, h).items()},
                    }))

    res = pd.concat(recs, ignore_index=True)
    per_coin = pd.concat([score(g).assign(asset=a) for a, g in res.groupby("asset")])
    print("\nPINBALL % of price by coin (lower better)")
    print(per_coin.reset_index().pivot(index="model", columns="asset", values="pinball_%").round(2).to_string())
    log("linear", score(res), origins=len(res) // (4 * H), assets=res.asset.nunique())
