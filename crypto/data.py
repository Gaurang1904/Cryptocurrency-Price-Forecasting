"""Raw ingestion from Binance. No features, no renaming, no targets.

Timestamps are the candle OPEN time in UTC: the bar labelled 2024-01-01 covers
2024-01-01 00:00 -> 2024-01-02 00:00 and is only complete after that closes.
"""

import time
from pathlib import Path

import ccxt
import pandas as pd

DATA = Path("data")
OHLCV_OUT = DATA / "ohlcv.parquet"
FUNDING_OUT = DATA / "funding.parquet"
OI_OUT = DATA / "open_interest.parquet"

# Which coins to model. Set to None to fall back on the top TOP_N by volume.
ASSETS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]
TOP_N = 50
MIN_BARS = 365  # a series shorter than a year is noise in a panel
TIMEFRAME = "1d"
START = "2017-01-01T00:00:00Z"
FUNDING_START = "2019-09-01T00:00:00Z"  # binance perps launched late 2019
LIMIT = 1000  # binance max per request

# Stablecoins and fiat pairs: no crypto volatility to model, drop them.
STABLES = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EURI", "AEUR", "USD1", "XUSD",
           "EUR", "GBP", "JPY", "TRY", "BRL", "ARS", "AUD", "RUB", "ZAR", "PLN", "RON"}


def universe(ex, n=TOP_N):
    """Top n active spot USDT pairs by 24h quote volume.

    ponytail: ranking by TODAY's volume is survivorship bias - coins that died
    never appear. Reconstitute per-date if a result ever hinges on it; for
    volatility modelling it does not.
    """
    if ASSETS:
        return list(ASSETS)
    ex.load_markets()
    ranked = []
    for symbol, t in ex.fetch_tickers().items():
        m = ex.markets.get(symbol)
        if not (m and m.get("spot") and m.get("active") and m.get("quote") == "USDT"):
            continue
        if m.get("base") in STABLES or not t.get("quoteVolume"):
            continue
        ranked.append((symbol, t["quoteVolume"]))
    ranked.sort(key=lambda r: -r[1])
    return [s for s, _ in ranked[:n]]


def _page(call, since, now, label):
    """Page a time-series endpoint forward. Raises rather than truncating."""
    rows, retries = [], 0
    while since < now:
        try:
            batch = call(since)
            retries = 0
        except Exception as e:
            retries += 1
            if retries > 5:
                at = pd.to_datetime(since, unit="ms", utc=True)
                raise RuntimeError(f"{label}: failed 5x at {at}") from e
            time.sleep(2**retries)
            continue
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1][0] if isinstance(batch[-1], list) else batch[-1]["timestamp"]
        if last <= since:  # no forward progress, stop rather than spin
            break
        since = last + 1
    return rows


def check(df):
    """Invariants that must hold before anything downstream trusts this file."""
    assert not df.duplicated(["asset", "date"]).any(), "duplicate (asset, date)"
    assert df["date"].dt.tz is not None, "timestamps must be tz-aware UTC"
    assert df["date"].max() < pd.Timestamp.now(tz="UTC").normalize(), "incomplete bar present"
    assert (df[["open", "high", "low", "close"]] > 0).all().all(), "non-positive price"
    assert (df["high"] >= df["low"]).all(), "high < low"
    for asset, g in df.groupby("asset"):
        assert g["date"].is_monotonic_increasing, f"{asset}: dates not sorted"


def fetch_ohlcv(ex=None):
    ex = ex or ccxt.binance({"enableRateLimit": True})
    start_ms, now = ex.parse8601(START), ex.milliseconds()
    symbols = universe(ex)
    print(f"universe: {len(symbols)} symbols by 24h volume")

    frames = []
    for symbol in symbols:
        rows = _page(lambda s: ex.fetch_ohlcv(symbol, TIMEFRAME, s, limit=LIMIT),
                     start_ms, now, symbol)
        if len(rows) < MIN_BARS:
            print(f"  {symbol}: {len(rows)} bars, too short, skipped")
            continue
        f = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        f["asset"] = symbol.split("/")[0]
        frames.append(f)
        print(f"  {symbol}: {len(f)} bars")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    # Today's bar is still forming - keeping it would leak a partial close.
    df = df[df["date"] < pd.Timestamp.now(tz="UTC").normalize()]
    df = (df.drop_duplicates(["asset", "date"])
            .sort_values(["asset", "date"]).reset_index(drop=True))
    df = df[["asset", "date", "open", "high", "low", "close", "volume"]]

    check(df)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OHLCV_OUT, index=False)
    print(f"\n{len(df)} rows, {df.asset.nunique()} assets -> {OHLCV_OUT}")
    print(f"{df.date.min().date()} .. {df.date.max().date()}")
    return df


def fetch_derivatives(ex=None):
    """Funding is refetched in full. Open interest is APPENDED - binance only
    serves ~30 days, so that file is the one you cannot rebuild."""
    ex = ex or ccxt.binance({"enableRateLimit": True})
    start_ms, now = ex.parse8601(FUNDING_START), ex.milliseconds()
    cutoff = pd.Timestamp.now(tz="UTC").normalize()

    assets = sorted(pd.read_parquet(OHLCV_OUT, columns=["asset"]).asset.unique())
    print(f"{len(assets)} assets from {OHLCV_OUT}")

    funding, oi = [], []
    for asset in assets:
        # Binance lists cheap tokens as 1000PEPE, 1000SHIB, ... on perps.
        rows, symbol = [], f"{asset}/USDT:USDT"
        for cand in (symbol, f"1000{asset}/USDT:USDT"):
            try:
                rows = _page(lambda s: ex.fetch_funding_rate_history(cand, s, limit=LIMIT),
                             start_ms, now, cand)
                symbol = cand
                break
            except Exception as e:
                err = type(e).__name__
        if not rows:
            print(f"  {asset}: no funding ({err})")
        if rows:
            funding.append(pd.DataFrame({
                "asset": asset,
                "ts": pd.to_datetime([r["timestamp"] for r in rows], unit="ms", utc=True),
                "funding_rate": [float(r["fundingRate"]) for r in rows],
            }))
        print(f"  {asset}: {len(rows)} funding settlements")

        try:
            rows = ex.fetch_open_interest_history(symbol, "1d", limit=500)
            oi.append(pd.DataFrame({
                "asset": asset,
                "date": pd.to_datetime([r["timestamp"] for r in rows], unit="ms", utc=True),
                "open_interest": [float(r["openInterestAmount"]) for r in rows],
            }))
        except Exception as e:
            print(f"  {asset}: open interest unavailable ({type(e).__name__})")

    # Funding settles 3x daily. Daily sum = what a position actually paid.
    f = pd.concat(funding, ignore_index=True)
    f["date"] = f["ts"].dt.floor("D")
    f = (f[f["date"] < cutoff].groupby(["asset", "date"], as_index=False)
         .agg(funding_sum=("funding_rate", "sum"), funding_n=("funding_rate", "size")))
    f.to_parquet(FUNDING_OUT, index=False)
    print(f"\nfunding: {len(f)} rows, {f.date.min().date()} .. {f.date.max().date()}")

    if oi:
        o = pd.concat(oi, ignore_index=True)
        o["date"] = o["date"].dt.floor("D")
        o = o[o["date"] < cutoff]
        if OI_OUT.exists():  # accumulate: old rows can never be refetched
            o = pd.concat([pd.read_parquet(OI_OUT), o], ignore_index=True)
        o = (o.drop_duplicates(["asset", "date"], keep="last")
              .sort_values(["asset", "date"]).reset_index(drop=True))
        o.to_parquet(OI_OUT, index=False)
        print(f"open interest: {len(o)} rows, {o.date.min().date()} .. {o.date.max().date()}")
    return f
