import numpy as np
import pandas as pd


FORECAST_KEY = ["asset", "origin", "h"]
KEY = ["model", *FORECAST_KEY]
REQUIRED = [
    "model", "asset", "origin", "target_time", "split", "h", "y", "last",
    "regime_driver", "origin_sigma", "origin_momentum", "q10", "q50", "q90",
]
_CONTEXT_COLUMNS = [
    "target_time", "split", "y", "last", "regime_driver", "origin_sigma",
    "origin_momentum",
]
_NUMERIC_COLUMNS = [
    "h", "y", "last", "regime_driver", "origin_sigma", "origin_momentum",
    "q10", "q50", "q90",
]
_FREQUENCY = pd.Timedelta("15min")


def _quantile_columns(cv):
    return {
        "q10": next(column for column in cv if "-lo-" in column),
        "q50": next(column for column in cv if column.endswith("-median")),
        "q90": next(column for column in cv if "-hi-" in column),
    }


def _utc(values):
    timestamps = pd.to_datetime(values)
    if getattr(timestamps.dt, "tz", None) is None:
        return timestamps.dt.tz_localize("UTC")
    return timestamps.dt.tz_convert("UTC")


def to_tft_forecasts(cv, context, model):
    quantiles = _quantile_columns(cv)
    forecasts = cv.copy()
    forecasts["cutoff"] = _utc(forecasts["cutoff"])
    forecasts["ds"] = _utc(forecasts["ds"])

    cutoff_context = context[[
        "asset", "ds", "close", "rv_672", "sigma_672", "momentum_96",
    ]].copy()
    cutoff_context["ds"] = _utc(cutoff_context["ds"])
    cutoff_context = cutoff_context.rename(columns={
        "ds": "cutoff", "close": "last", "rv_672": "regime_driver",
        "sigma_672": "origin_sigma", "momentum_96": "origin_momentum",
    })
    forecasts = forecasts.merge(
        cutoff_context, left_on=["unique_id", "cutoff"],
        right_on=["asset", "cutoff"], how="left", validate="many_to_one",
    )
    elapsed = forecasts["ds"] - forecasts["cutoff"]
    if (elapsed <= pd.Timedelta(0)).any() or (elapsed % _FREQUENCY != pd.Timedelta(0)).any():
        raise ValueError("forecast timestamps must be positive 15-minute horizons")
    frame = pd.DataFrame({
        "model": model,
        "asset": forecasts["asset"],
        "origin": forecasts["cutoff"] + _FREQUENCY,
        "target_time": forecasts["ds"] + _FREQUENCY,
        "split": "unassigned",
        "h": (elapsed / _FREQUENCY).astype(int),
        "y": np.exp(forecasts["y"]),
        "last": forecasts["last"],
        "regime_driver": forecasts["regime_driver"],
        "origin_sigma": forecasts["origin_sigma"],
        "origin_momentum": forecasts["origin_momentum"],
        "q10": np.exp(forecasts[quantiles["q10"]]),
        "q50": np.exp(forecasts[quantiles["q50"]]),
        "q90": np.exp(forecasts[quantiles["q90"]]),
    })
    validate_tft_forecasts(frame, expected_horizons=tuple(sorted(frame.h.unique())))
    return frame[REQUIRED]


def validate_tft_forecasts(frame, expected_horizons):
    missing = [column for column in REQUIRED if column not in frame]
    if missing:
        raise ValueError(f"missing forecast columns: {missing}")
    if frame.empty:
        raise ValueError("forecast frame is empty")
    if frame[REQUIRED].isna().any().any():
        raise ValueError("forecast missing values")
    for column in _NUMERIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError("forecast numeric values must be finite")
    for column in ("origin", "target_time"):
        timestamps = pd.to_datetime(frame[column], errors="coerce")
        if timestamps.isna().any() or getattr(timestamps.dt, "tz", None) is None:
            raise ValueError(f"{column} must be timezone-aware UTC")
        if str(timestamps.dt.tz) not in ("UTC", "UTC+00:00"):
            raise ValueError(f"{column} must be timezone-aware UTC")
    if frame[KEY].isna().any().any():
        raise ValueError("forecast keys must not be missing")
    if frame.duplicated(KEY).any():
        raise ValueError("duplicate forecast keys")
    expected = set(expected_horizons)
    if not expected or set(frame.h) - expected:
        raise ValueError("unexpected forecast horizons")
    if (frame[["y", "last", "q10", "q50", "q90"]] <= 0).any().any():
        raise ValueError("price-space values must be strictly positive")
    grouped_horizons = frame.groupby(["model", "asset", "origin"], sort=False).h.agg(set)
    if not grouped_horizons.map(lambda horizons: horizons == expected).all():
        raise ValueError("complete horizons required for every forecast origin")
    if (frame.q10 > frame.q50).any() or (frame.q50 > frame.q90).any():
        raise ValueError("crossed quantiles")
    if not (frame.target_time == frame.origin + frame.h * _FREQUENCY).all():
        raise ValueError("target times must match forecast horizons")

    models = list(frame.model.drop_duplicates())
    reference = frame.loc[frame.model == models[0], [*FORECAST_KEY, *_CONTEXT_COLUMNS]]
    reference = reference.sort_values(FORECAST_KEY).reset_index(drop=True)
    reference_keys = set(map(tuple, reference[FORECAST_KEY].to_numpy()))
    for model in models[1:]:
        candidate = frame.loc[frame.model == model, [*FORECAST_KEY, *_CONTEXT_COLUMNS]]
        candidate = candidate.sort_values(FORECAST_KEY).reset_index(drop=True)
        if set(map(tuple, candidate[FORECAST_KEY].to_numpy())) != reference_keys:
            raise ValueError("models must use identical forecast key sets")
        for column in _CONTEXT_COLUMNS:
            if not candidate[column].equals(reference[column]):
                raise ValueError(f"models must share {column}")


def split_calibration_test(frame, eligible_origins, n_calibration=219, n_test=146):
    ordered = pd.DatetimeIndex(sorted(set(eligible_origins)))
    required = n_calibration + n_test
    if len(ordered) < required:
        raise ValueError(f"need {required} eligible daily origins, found {len(ordered)}")
    selected = ordered[-required:]
    cal_origins, test_origins = selected[:n_calibration], selected[n_calibration:]
    calibration = frame[frame.origin.isin(cal_origins)].copy()
    test = frame[frame.origin.isin(test_origins)].copy()
    calibration["split"], test["split"] = "calibration", "test"
    if test.origin.nunique() < 100:
        raise ValueError("fewer than 100 valid test origins")
    return calibration, test


def make_tft_baselines(reference):
    expected_horizons = tuple(sorted(reference.h.unique()))
    validate_tft_forecasts(reference, expected_horizons)
    baseline = reference[REQUIRED].copy()
    root_h = np.sqrt(baseline.h.to_numpy())
    origin_log = np.log(baseline["last"].to_numpy())
    band = 1.2815515655 * baseline.origin_sigma.to_numpy() * root_h
    mids = {
        "persistence_vol": origin_log,
        "momentum_vol": origin_log + (
            baseline.h.to_numpy() / 96
        ) * baseline.origin_momentum.to_numpy(),
    }
    frames = []
    for model, mid in mids.items():
        forecast = baseline.copy()
        forecast["model"] = model
        forecast["q10"] = np.exp(mid - band)
        forecast["q50"] = np.exp(mid)
        forecast["q90"] = np.exp(mid + band)
        frames.append(forecast)
    baselines = pd.concat(frames, ignore_index=True)
    validate_tft_forecasts(baselines, expected_horizons)
    return baselines
