"""Feature and target construction.

Every feature is backward-looking, so building the table once over the whole
frame introduces no look-ahead. Targets use shift(-h) and are labels ONLY -
putting one in the feature list is the leak that invalidates everything.
"""

import numpy as np
import pandas as pd

from crypto.data import FUNDING_OUT

H = 7  # forecast horizon in days
PREFIXES = ("ret_", "vol_", "range", "px_", "btc_", "fund", "drawdown_", "volume_")
CANDIDATE_FEATURES = {"vol_regime", "drawdown_63d", "volume_z21"}


def feature_cols(feat, include_candidates=False):
    columns = [c for c in feat.columns if c.startswith(PREFIXES)]
    if include_candidates:
        return columns
    return [c for c in columns if c not in CANDIDATE_FEATURES]


def feature_groups(columns):
    rules = {
        "returns": ("ret_", "px_", "drawdown_"),
        "volatility": ("vol_", "range", "har_"),
        "volume": ("volume_",),
        "market": ("btc_",),
        "funding": ("fund",),
    }
    groups = {name: [c for c in columns if c.startswith(prefixes)]
              for name, prefixes in rules.items()}
    flattened = [c for members in groups.values() for c in members]
    assert len(columns) == len(set(columns)), "feature columns contain duplicates"
    assert sorted(flattened) == sorted(columns), "feature groups must cover each feature once"
    return groups


def make_features(df):
    btc = (df[df.asset == "BTC"].set_index("date")["close"]
           .pipe(np.log).diff().rename("btc_ret"))
    out = []
    for _, g in df.groupby("asset", sort=False):
        g = g.sort_values("date").copy()
        lr = np.log(g["close"]).diff()

        for k in [1, 2, 3, 5, 8, 13]:
            g[f"ret_lag{k}"] = lr.shift(k - 1)
        for w in [5, 10, 21, 63]:
            g[f"ret_{w}d"] = lr.rolling(w).sum()
            g[f"vol_{w}d"] = lr.rolling(w).std()
        g["vol_regime"] = g["vol_21d"] / g["vol_63d"].clip(lower=1e-6)
        g["drawdown_63d"] = g.close / g.close.rolling(63).max() - 1
        lv = np.log(g.volume.replace(0, np.nan))
        g["volume_z21"] = (lv - lv.rolling(21).mean()) / lv.rolling(21).std()
        g["range"] = (g.high - g.low) / g.close
        g["range_5d"] = g["range"].rolling(5).mean()
        g["vol_ratio"] = g.volume / g.volume.rolling(21).mean()
        g["px_vs_ma63"] = np.log(g.close / g.close.rolling(63).mean())
        g["naive_mae"] = g.close.diff().abs().expanding().mean()  # MASE scale

        # HAR-RV inputs: realised vol over the past 1 / 5 / 22 days, logged.
        sq = lr**2
        for w, name in [(1, "har_1d"), (5, "har_5d"), (22, "har_22d")]:
            g[name] = np.log(np.sqrt(sq.rolling(w).mean()).clip(lower=1e-6))

        for h in range(1, H + 1):
            # Endpoint metadata is label-only. Shifting each asset's actual date
            # sequence handles missing calendar days without a row/global shortcut.
            g[f"label_end{h}"] = g.date.shift(-h)
            g[f"y{h}"] = np.log(g.close.shift(-h) / g.close)

        g = g.join(btc, on="date")
        g["btc_ret_5d"] = g.btc_ret.rolling(5).sum()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def add_funding(feat):
    f = pd.read_parquet(FUNDING_OUT)
    stale = sorted(set(feat.asset.unique()) - set(f.asset.unique()))
    if stale:  # a silent left-join miss costs whole assets at the next dropna
        print(f"WARNING: no funding for {len(stale)} assets {stale[:6]}... "
              f"- rerun fetch.py or they get dropped from training")
    feat = feat.merge(f[["asset", "date", "funding_sum"]], on=["asset", "date"], how="left")
    g = feat.groupby("asset", sort=False)["funding_sum"]
    feat["fund_7d"] = g.transform(lambda s: s.rolling(7).mean())
    feat["fund_21d"] = g.transform(lambda s: s.rolling(21).mean())
    # Crowding: today's funding relative to its own recent range.
    feat["fund_z"] = g.transform(lambda s: (s - s.rolling(63).mean()) / s.rolling(63).std())
    return feat


def add_vol_targets(feat):
    """rv_h = RMS of the log returns over t+1..t+h. Label only, never a feature."""
    out = []
    for _, g in feat.groupby("asset", sort=False):
        g = g.copy()
        sq = np.log(g.close).diff() ** 2
        for h in range(1, H + 1):
            # Floor: an unchanged close gives rv=0, and log(0) poisons the target.
            g[f"rv{h}"] = np.sqrt(sq.rolling(h).sum().shift(-h) / h).clip(lower=1e-6)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build(df, funding=True, include_candidates=False):
    feat = make_features(df)
    if funding:
        feat = add_funding(feat)
    feat = add_vol_targets(feat)
    return feat, feature_cols(feat, include_candidates=include_candidates)


def check_causal(df, cols=None):
    """Corrupt every asset from a date cutoff and compare every earlier row."""
    dates = pd.Index(df["date"].drop_duplicates()).sort_values()
    if len(dates) < 3:
        raise ValueError("causality check needs at least three distinct dates")
    tail = min(200, max(1, len(dates) // 3))
    cutoff = dates[-tail]

    a = make_features(df)
    cols = feature_cols(a) if cols is None else list(cols)
    missing = sorted(set(cols) - set(a.columns))
    if missing:
        raise ValueError(f"requested causality columns are missing: {missing}")

    bad = df.copy()
    bad.loc[bad.date >= cutoff, "close"] *= 3
    b = make_features(bad)
    missing = sorted(set(cols) - set(b.columns))
    if missing:
        raise ValueError(f"requested causality columns are missing after corruption: {missing}")

    keys = ["asset", "date"]
    left = a.loc[a.date < cutoff, keys + cols].sort_values(keys).reset_index(drop=True)
    right = b.loc[b.date < cutoff, keys + cols].sort_values(keys).reset_index(drop=True)
    if not left.equals(right):
        raise AssertionError("look-ahead in features")
