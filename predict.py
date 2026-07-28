"""Load the saved model and print the 7-day forecast. `--asset BTC` to filter."""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from crypto.data import OHLCV_OUT
from crypto.features import build
from crypto.model import bands, clip_sigma
from crypto.train import artifact_path

MAX_STALE_DAYS = 3


def forecast(art, feat, h=None):
    h = h or art["H"]
    now = feat.groupby("asset", sort=False).tail(1)
    bad = now[now[art["cols"]].isna().any(axis=1)]
    if not bad.empty:  # skip loudly - never forecast an asset from NaN features
        print(f"skipping {len(bad)} asset(s) with missing features: {list(bad.asset)}\n")
        now = now.drop(bad.index)
    assert not now.empty, "no asset has complete features"

    sigma = clip_sigma(np.exp(art["models"][h].predict(now[art["cols"]])))
    last = now.close.to_numpy()
    out = pd.DataFrame({"asset": now.asset.values, "last": last, "vol_pred_%": sigma * 100})
    for q, v in bands(art["z"][h], last, sigma, h).items():
        out[f"q{int(q * 100)}"] = v
    out["band_%"] = (out.q90 - out.q10) / out["last"] * 100
    return out.set_index("asset").sort_values("band_%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", help="single asset, e.g. BTC")
    ap.add_argument("--model", default="lgbm", help="lgbm, xgb, lstm, or dlinear")
    ap.add_argument("--horizon", type=int, default=7, help="days ahead, 1-7")
    args = ap.parse_args()

    feat, cols = build(pd.read_parquet(OHLCV_OUT))
    asof = feat.date.max()
    stale = (pd.Timestamp.now(tz="UTC").normalize() - asof).days
    if stale > MAX_STALE_DAYS:
        print(f"WARNING: data is {stale} days old, run fetch.py\n")

    if args.model in ("lstm", "dlinear"):  # neural: .pt + windowed torch forward
        from neural.core import load_and_forecast
        out, art = load_and_forecast(args.model, feat, args.horizon)
    else:                                  # tree: .joblib + tabular predict
        path = artifact_path(args.model)
        if not path.exists():
            sys.exit(f"no model at {path} - run `python -m tree.{args.model}` first")
        art = joblib.load(path)
        assert cols == art["cols"], "feature set changed since training - retrain"
        out = forecast(art, feat, args.horizon)

    if args.asset:
        out = out.loc[[args.asset]]
    print(f"{args.horizon}-day forecast from {asof.date()} "
          f"({args.model} trained {art['trained_at'].date()})\n")
    print(out.round(4).to_string())
