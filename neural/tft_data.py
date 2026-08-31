from dataclasses import dataclass

import numpy as np
import pandas as pd


FREQ = "15min"
REQUIRED_ASSETS = ("BTC", "ETH", "BNB", "SOL", "XRP")
REQUIRED_COLUMNS = ("asset", "date", "open", "high", "low", "close", "volume")
MIN_HISTORY_BARS = 2

INPUT_SIZE = 672
HORIZON = 96

HIST_EXOG = (
    "log_return", "abs_return", "range_pct", "log_volume",
    "log_volume_change", "rv_96", "rv_672", "momentum_96",
    "momentum_672", "volume_z96", "missing_bar",
)
FUTR_EXOG = ("tod_sin", "tod_cos", "dow_sin", "dow_cos")


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
    requested = tuple(assets)
    if set(requested) - set(REQUIRED_ASSETS):
        raise ValueError("unsupported requested assets")
    source_assets = set(raw.asset.dropna().unique())
    if source_assets - set(REQUIRED_ASSETS):
        raise ValueError("unsupported source assets")
    frame = raw.loc[raw.asset.isin(requested), REQUIRED_COLUMNS].copy()
    if set(frame.asset.unique()) != set(assets):
        raise ValueError("missing required assets")
    if frame.duplicated(["asset", "date"]).any():
        raise ValueError("duplicate (asset, date)")
    if frame.date.dt.tz is None or str(frame.date.dt.tz) not in ("UTC", "UTC+00:00"):
        raise ValueError("timestamps must be timezone-aware UTC")
    if ((frame.date.dt.minute % 15 != 0) | (frame.date.dt.second != 0) | (frame.date.dt.microsecond != 0)).any():
        raise ValueError("timestamps must be 15-minute aligned")
    if not all(g.date.is_monotonic_increasing for _, g in frame.groupby("asset")):
        raise ValueError("timestamps must be sorted within asset")
    if frame.groupby("asset").size().lt(MIN_HISTORY_BARS).any():
        raise ValueError("insufficient history")
    values = frame[["open", "high", "low", "close", "volume"]]
    if not all(pd.api.types.is_numeric_dtype(values[column]) for column in values):
        raise ValueError("OHLCV values must be numeric and finite")
    numeric = values.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("OHLCV values must be numeric and finite")
    prices = numeric[["open", "high", "low", "close"]]
    invalid = ((prices <= 0).any(axis=1) | (numeric.volume < 0)
               | (numeric.high < prices[["open", "close", "low"]].max(axis=1))
               | (numeric.low > prices[["open", "close", "high"]].min(axis=1)))
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

@dataclass(frozen=True)
class PreparedTFTData:
    model_frame: pd.DataFrame
    context_frame: pd.DataFrame
    gap_stats: dict
    data_end: pd.Timestamp


def _add_features(grid):
    parts = []
    for _, group in grid.groupby("asset", sort=False):
        g = group.sort_values("date").copy()
        log_close = np.log(g.close)
        g["log_return"] = log_close.diff()
        g["abs_return"] = g.log_return.abs()
        g["range_pct"] = (g.high - g.low) / g.close
        g["log_volume"] = np.log1p(g.volume)
        g["log_volume_change"] = g.log_volume.diff()
        g["rv_96"] = g.log_return.pow(2).rolling(96).sum().pow(0.5)
        g["rv_672"] = g.log_return.pow(2).rolling(672).sum().pow(0.5)
        g["sigma_672"] = g.log_return.rolling(672).std(ddof=0)
        g["momentum_96"] = log_close - log_close.shift(96)
        g["momentum_672"] = log_close - log_close.shift(672)
        mean = g.log_volume.rolling(96).mean()
        std = g.log_volume.rolling(96).std(ddof=0)
        g["volume_z96"] = ((g.log_volume - mean) / std.replace(0, np.nan)).fillna(0)
        minute = g.date.dt.hour * 60 + g.date.dt.minute
        g["tod_sin"] = np.sin(2 * np.pi * minute / 1440)
        g["tod_cos"] = np.cos(2 * np.pi * minute / 1440)
        g["dow_sin"] = np.sin(2 * np.pi * g.date.dt.dayofweek / 7)
        g["dow_cos"] = np.cos(2 * np.pi * g.date.dt.dayofweek / 7)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _latest_shared_daily_end(grid, assets):
    common_end = min(grid.loc[grid.asset == asset, "date"].max() for asset in assets)
    data_end = common_end.normalize() + pd.Timedelta(hours=23, minutes=45)
    if data_end > common_end:
        data_end -= pd.Timedelta(days=1)
    return data_end


def prepare_tft_data(raw, assets=REQUIRED_ASSETS, as_of=None):
    assets = tuple(assets)
    validated = validate_ohlcv_15m(raw, assets=assets, as_of=as_of)
    grid, gap_stats = complete_ohlcv_grid(validated, assets=assets)
    data_end = _latest_shared_daily_end(grid, assets)
    grid = grid[grid.date <= data_end]
    featured = _add_features(grid).dropna(subset=HIST_EXOG).reset_index(drop=True)
    if set(featured.asset.unique()) != set(assets):
        raise ValueError("insufficient usable 672-bar history")
    model_frame = featured.rename(columns={"asset": "unique_id"}).assign(
        ds=lambda d: d.date.dt.tz_convert("UTC").dt.tz_localize(None),
        y=lambda d: np.log(d.close),
    )[["unique_id", "ds", "y", *HIST_EXOG, *FUTR_EXOG]]
    context_frame = featured.assign(
        ds=lambda d: d.date.dt.tz_convert("UTC").dt.tz_localize(None),
    )[[
        "asset", "ds", "open", "close", "missing_bar",
        "rv_672", "sigma_672", "momentum_96",
    ]]
    return PreparedTFTData(model_frame, context_frame, gap_stats, data_end)


def eligible_daily_origins(context, horizon):
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    groups = {}
    candidates = None
    for asset, group in context.groupby("asset", sort=False):
        indexed = group.sort_values("ds").set_index("ds")
        groups[asset] = indexed
        midnights = set(indexed.index[(indexed.index.hour == 0) & (indexed.index.minute == 0)])
        candidates = midnights if candidates is None else candidates & midnights
    origins = []
    for origin in sorted(candidates or set()):
        target = pd.date_range(origin, periods=horizon, freq=FREQ)
        cutoff = origin - pd.Timedelta(FREQ)
        windows = [group.reindex(target) for group in groups.values() if cutoff in group.index]
        if len(windows) == len(groups) and all(
            len(window) == horizon and window.missing_bar.notna().all()
            and not window.missing_bar.any() for window in windows
        ):
            origins.append(pd.Timestamp(origin, tz="UTC"))
    return pd.DatetimeIndex(origins, tz="UTC")


def assert_tft_feature_causality(raw, cutoff, assets=REQUIRED_ASSETS, as_of=None):
    cutoff = pd.Timestamp(cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tz is None else cutoff.tz_convert("UTC")
    baseline = prepare_tft_data(raw, assets=assets, as_of=as_of)
    corrupted = raw.copy()
    future = corrupted.date > cutoff
    if not future.any():
        raise ValueError("no post-cutoff bars to corrupt")
    corrupted.loc[future, ["open", "high", "low", "close"]] *= 1.1
    corrupted.loc[future, "volume"] += 1
    rebuilt = prepare_tft_data(corrupted, assets=assets, as_of=as_of)
    cutoff_ds = cutoff.tz_localize(None)
    expected = baseline.model_frame[baseline.model_frame.ds <= cutoff_ds]
    actual = rebuilt.model_frame[rebuilt.model_frame.ds <= cutoff_ds]
    pd.testing.assert_frame_equal(
        expected.set_index(["unique_id", "ds"])[list(HIST_EXOG)].sort_index(),
        actual.set_index(["unique_id", "ds"])[list(HIST_EXOG)].sort_index(),
    )
