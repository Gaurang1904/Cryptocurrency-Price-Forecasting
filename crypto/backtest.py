"""Walk-forward evaluation, naive baselines, scoring, and the results log.

Every model in this repo is scored by these functions on these origins.
A model that cannot beat `flat` is not earning its complexity.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from crypto.features import H

RESULTS = Path("results.csv")
STRIDE = 7      # days between forecast origins (non-overlapping windows)
MIN_HIST = 60   # bars required before the first origin
DRIFT_WIN = 30
TEST_START = pd.Timestamp("2021-01-01", tz="UTC")
FOLD = pd.DateOffset(years=1)


# --- baselines ------------------------------------------------------------

def naive(hist):
    """hist: closes up to and including the origin. Returns {name: array(H)}."""
    last = hist[-1]
    return {
        "flat": np.full(H, last),
        "drift": last + np.arange(1, H + 1) * np.diff(hist[-DRIFT_WIN:]).mean(),
        "seasonal": hist[-H:].copy(),  # y[t+h] = y[t+h-7]
    }


def walk_forward(df, predict=naive):
    """Roll an expanding window over every asset. One row per (origin, model, h)."""
    recs = []
    for asset, g in df.groupby("asset", sort=False):
        close, dates = g["close"].to_numpy(), g["date"].to_numpy()
        for t in range(MIN_HIST, len(close) - H, STRIDE):
            hist, actual = close[: t + 1], close[t + 1 : t + 1 + H]
            naive_mae = np.abs(np.diff(hist)).mean()
            for name, pred in predict(hist).items():
                for h in range(H):
                    recs.append((asset, dates[t], name, h + 1,
                                 actual[h], pred[h], hist[-1], naive_mae))
    return pd.DataFrame(recs, columns=["asset", "origin", "model", "h",
                                       "y", "yhat", "last", "naive_mae"])


# --- folds ----------------------------------------------------------------

def origin_mask(feat):
    """Every STRIDE-th bar per asset after MIN_HIST - matches walk_forward."""
    idx = feat.groupby("asset", sort=False).cumcount()
    n = feat.groupby("asset", sort=False)["close"].transform("size")
    return (idx >= MIN_HIST) & (idx < n - H) & ((idx - MIN_HIST) % STRIDE == 0)


def run_folds(feat, cols):
    """Yield (train, test, start) per expanding-window fold. Train is strictly earlier."""
    origins = feat[origin_mask(feat)]
    targets = [f"y{h}" for h in range(1, H + 1)]
    for start in pd.date_range(TEST_START, feat.date.max(), freq=FOLD, tz="UTC"):
        train = feat[feat.date < start].dropna(subset=cols + targets)
        test = origins[(origins.date >= start) & (origins.date < start + FOLD)].dropna(subset=cols)
        if not test.empty:
            yield train, test, start


# --- scoring --------------------------------------------------------------

def report(res):
    """Point-forecast metrics: MAE, MAPE, MASE, directional accuracy."""
    res = res.assign(
        ae=(res.y - res.yhat).abs(),
        ape=(res.y - res.yhat).abs() / res.y,
        scaled_ae=(res.y - res.yhat).abs() / res.naive_mae,
    )
    summary = res.groupby("model").agg(
        MAE=("ae", "mean"), MAPE=("ape", "mean"), MASE=("scaled_ae", "mean"))
    summary["MAPE"] *= 100

    # Direction of the cumulative H-day move, the only call you could trade on.
    end = res[res.h == H]
    hit = np.sign(end.yhat - end.last) == np.sign(end.y - end.last)
    summary["DirAcc"] = hit.groupby(end.model).mean() * 100

    per_h = res.pivot_table(index="model", columns="h", values="ape", aggfunc="mean") * 100
    return summary.sort_values("MASE"), per_h


def pinball(y, yhat, q):
    d = y - yhat
    return np.maximum(q * d, (q - 1) * d).mean()


def score(res, quantiles=(0.1, 0.5, 0.9)):
    """Interval metrics: pinball loss, coverage, band width, median MAPE."""
    lo_q, mid_q, hi_q = quantiles[0], 0.5, quantiles[-1]
    rows = []
    for model, g in res.groupby("model"):
        lo, mid, hi = (g[f"q{int(q * 100)}"] for q in (lo_q, mid_q, hi_q))
        rows.append({
            "model": model,
            "pinball": np.mean([pinball(g.y, g[f"q{int(q * 100)}"], q) for q in quantiles]),
            "coverage": ((g.y >= lo) & (g.y <= hi)).mean() * 100,
            "width": ((hi - lo) / g["last"]).mean() * 100,
            "med_MAPE": ((g.y - mid).abs() / g.y).mean() * 100,
            # Pinball is in price units, so it cannot be compared across coins.
            # Scaling by price makes BTC and XRP readable in one table.
            "pinball_%": np.mean([pinball(g.y, g[f"q{int(q * 100)}"], q)
                                  for q in quantiles]) / g["last"].mean() * 100,
        })
    return pd.DataFrame(rows).set_index("model").sort_values("pinball")


def coverage_by_h(res):
    return (res.assign(hit=(res.y >= res.q10) & (res.y <= res.q90))
               .pivot_table(index="model", columns="h", values="hit") * 100)


def point_metrics(res):
    """Median-forecast metrics. R2 is computed on RETURNS, never price.

    R2 on price is a vanity 0.99 (price barely moves bar-to-bar, so 'predict last'
    already explains ~all variance). On returns it is honest - and will sit near
    0, which is the true signal. MAE/RMSE are in price units so they mix scales
    across coins; read MAPE / R2_ret / DirAcc for the cross-coin comparison.
    """
    rows = []
    for model, g in res.groupby("model"):
        a, p, last = g.y.to_numpy(), g.q50.to_numpy(), g["last"].to_numpy()
        ar, pr = a / last - 1, p / last - 1                 # actual vs predicted return
        ss_res, ss_tot = np.sum((ar - pr) ** 2), np.sum((ar - ar.mean()) ** 2)
        end = g[g.h == g.h.max()]                           # cumulative-move direction
        rows.append({
            "model": model,
            "MAE": np.mean(np.abs(a - p)),
            "RMSE": np.sqrt(np.mean((a - p) ** 2)),
            "MAPE": np.mean(np.abs(a - p) / a) * 100,
            "R2_ret": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "DirAcc": (np.sign(end.q50 - end["last"]) == np.sign(end.y - end["last"])).mean() * 100,
        })
    return pd.DataFrame(rows).set_index("model")


# --- results log ----------------------------------------------------------

def log(tag, summary, **meta):
    """Append a scored run to results.csv.

    Long format (run, tag, model, metric, value) so scripts reporting different
    metrics still share one schema.
    ponytail: a CSV is experiment tracking until two people need to query it.
    """
    # Fixed schema: meta goes in ONE column as "k=v;..." so runs with different
    # meta keys (origins vs horizon) never misalign the appended CSV.
    meta_str = ";".join(f"{k}={v}" for k, v in meta.items())
    long = (summary.reset_index()
            .melt(id_vars="model", var_name="metric", value_name="value")
            .assign(run=pd.Timestamp.now(tz="UTC").floor("s"), tag=tag, meta=meta_str))
    cols = ["run", "tag", "meta", "model", "metric", "value"]
    long[cols].to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
    print(f"\nlogged {len(long)} rows to {RESULTS}")


def _self_check():
    """Constant price: flat is exact. Rising line: drift beats flat."""
    n = 200
    df = pd.DataFrame({"asset": "TEST",
                       "date": pd.date_range("2020-01-01", periods=n, tz="UTC"),
                       "close": np.full(n, 100.0)})
    res = walk_forward(df)
    assert (res[res.model == "flat"].eval("y - yhat").abs().max()) == 0
    df["close"] = np.arange(n, dtype=float) + 100
    s, _ = report(walk_forward(df))
    assert s.loc["drift", "MAE"] < s.loc["flat", "MAE"]
