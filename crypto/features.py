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


def feature_cols(feat):
    return [c for c in feat.columns if c.startswith(PREFIXES)]


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


def build(df, funding=True):
    feat = make_features(df)
    if funding:
        feat = add_funding(feat)
    feat = add_vol_targets(feat)
    return feat, feature_cols(feat)


def check_causal(df, cols=None):
    """Corrupting the tail must not move any earlier feature value.

    This is the test that would have caught a target column sitting in the
    feature list. Cheap enough to run before every training job.
    """
    a = make_features(df)
    cols = cols or [c for c in feature_cols(a)]
    bad = df.copy()
    bad.loc[bad.index[-200:], "close"] *= 3
    b = make_features(bad)
    cut = pd.Timestamp(a.date.max()) - pd.DateOffset(days=400)
    a, b = a[a.date < cut], b[b.date < cut]
    shared = [c for c in cols if c in a.columns]
    assert np.allclose(a[shared].fillna(0), b[shared].fillna(0)), "look-ahead in features"
