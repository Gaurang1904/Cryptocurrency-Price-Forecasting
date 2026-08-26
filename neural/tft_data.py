from dataclasses import dataclass

import pandas as pd


FREQ = "15min"
REQUIRED_ASSETS = ("BTC", "ETH", "BNB", "SOL", "XRP")
REQUIRED_COLUMNS = ("asset", "date", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class GapStats:
    observed_bars: int
    synthetic_bars: int
    gap_events: int
    max_gap_bars: int


def validate_ohlcv_15m(raw, assets=REQUIRED_ASSETS, as_of=None):
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")
    frame = raw.loc[raw.asset.isin(assets), REQUIRED_COLUMNS].copy()
    if set(frame.asset.unique()) != set(assets):
        raise ValueError("missing required assets")
    if frame.duplicated(["asset", "date"]).any():
        raise ValueError("duplicate (asset, date)")
    if frame.date.dt.tz is None:
        raise ValueError("timestamps must be timezone-aware UTC")
    if not all(g.date.is_monotonic_increasing for _, g in frame.groupby("asset")):
        raise ValueError("timestamps must be sorted within asset")
    prices = frame[["open", "high", "low", "close"]]
    invalid = ((prices <= 0).any(axis=1) | (frame.volume < 0)
               | (frame.high < prices[["open", "close", "low"]].max(axis=1))
               | (frame.low > prices[["open", "close", "high"]].min(axis=1)))
    if invalid.any():
        raise ValueError("invalid OHLC or volume")
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if ((frame.date + pd.Timedelta(FREQ)) > as_of).any():
        raise ValueError("incomplete final candle")
    return frame.reset_index(drop=True)


def complete_ohlcv_grid(raw, assets=REQUIRED_ASSETS):
    frames, summaries = [], {}
    for asset in assets:
        observed = raw[raw.asset == asset].set_index("date").sort_index()
        index = pd.date_range(observed.index.min(), observed.index.max(), freq=FREQ, tz="UTC")
        grid = observed.reindex(index)
        missing = grid.close.isna()
        event_start = missing & ~missing.shift(fill_value=False)
        runs = missing.groupby((missing != missing.shift()).cumsum()).transform("sum")
        previous_close = grid.close.ffill()
        for column in ("open", "high", "low", "close"):
            grid.loc[missing, column] = previous_close.loc[missing]
        grid.loc[missing, "volume"] = 0.0
        grid["missing_bar"] = missing.astype("int8")
        grid["asset"] = asset
        grid.index.name = "date"
        frames.append(grid.reset_index())
        summaries[asset] = GapStats(
            observed_bars=len(observed), synthetic_bars=int(missing.sum()),
            gap_events=int(event_start.sum()),
            max_gap_bars=int(runs[missing].max()) if missing.any() else 0,
        )
    return pd.concat(frames, ignore_index=True), summaries
