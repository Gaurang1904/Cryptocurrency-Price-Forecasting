import numpy as np
import pandas as pd

from crypto.backtest import score


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
    required = n_calibration + n_test
    scheduled = pd.DatetimeIndex(pd.to_datetime(frame.origin.unique(), utc=True)).sort_values()
    if len(scheduled) < required:
        raise ValueError(f"need {required} scheduled daily origins, found {len(scheduled)}")
    selected = scheduled[-required:]
    eligible = pd.DatetimeIndex(pd.to_datetime(eligible_origins, utc=True)).unique()
    ineligible = selected.difference(eligible)
    if len(ineligible):
        raise ValueError(
            f"{len(ineligible)} ineligible scheduled origins; need exactly "
            f"{n_calibration} calibration and {n_test} test origins"
        )
    cal_origins, test_origins = selected[:n_calibration], selected[n_calibration:]
    calibration = frame[frame.origin.isin(cal_origins)].copy()
    test = frame[frame.origin.isin(test_origins)].copy()
    if calibration.origin.nunique() != n_calibration or test.origin.nunique() != n_test:
        raise ValueError(
            f"need exactly {n_calibration} calibration and {n_test} test origins"
        )
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


def _higher_quantile(values, level):
    return float(np.quantile(np.asarray(values), min(level, 1.0), method="higher"))


def calibrate_tft_intervals(calibration, test, alpha=0.20):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    validate_tft_forecasts(calibration, expected_horizons=sorted(calibration.h.unique()))
    validate_tft_forecasts(test, expected_horizons=sorted(test.h.unique()))
    calibration_models = calibration.model.drop_duplicates().tolist()
    test_models = test.model.drop_duplicates().tolist()
    if len(calibration_models) != 1 or calibration_models != test_models:
        raise ValueError("calibration and test must use the same single model")

    missing = sorted(set(test.h.unique()) - set(calibration.h.unique()))
    if missing:
        raise ValueError(f"calibration rows missing for horizon {missing[0]}")

    adjusted = test.copy()
    for horizon in sorted(test.h.unique()):
        cal_h = calibration.loc[calibration.h.eq(horizon)]
        n = len(cal_h)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha / 2)) / n)
        lower_scores = np.log(cal_h.q10) - np.log(cal_h.y)
        upper_scores = np.log(cal_h.y) - np.log(cal_h.q90)
        lower = max(0.0, _higher_quantile(lower_scores, level))
        upper = max(0.0, _higher_quantile(upper_scores, level))
        mask = adjusted.h.eq(horizon)
        adjusted.loc[mask, "q10"] *= np.exp(-lower)
        adjusted.loc[mask, "q90"] *= np.exp(upper)
    adjusted["model"] = "tft_calibrated"
    validate_tft_forecasts(adjusted, expected_horizons=sorted(test.h.unique()))
    return adjusted


def fit_regime_cutpoints(calibration):
    required = ["asset", "origin", "regime_driver"]
    missing = [column for column in required if column not in calibration]
    if missing:
        raise ValueError(f"missing calibration columns: {missing}")
    drivers = calibration[required].drop_duplicates()
    values = pd.to_numeric(drivers.regime_driver, errors="coerce")
    if values.empty or values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("calibration regime_driver values must be finite")
    low, high = values.quantile([1 / 3, 2 / 3])
    return float(low), float(high)


def apply_regimes(frame, cutpoints):
    low, high = cutpoints
    labelled = frame.copy()
    labelled["regime"] = np.select(
        [labelled.regime_driver <= low, labelled.regime_driver <= high],
        ["low", "medium"], default="high",
    )
    return labelled


def _direction_interval(group, bootstrap_samples, seed):
    origins = group.origin.drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    hits = np.sign(group.q50 / group["last"] - 1) == np.sign(
        group.y / group["last"] - 1
    )
    clusters = pd.DataFrame({"origin": group.origin, "hit": hits}).groupby(
        "origin", sort=False,
    ).hit.agg(["sum", "count"]).reindex(origins)
    sampled = rng.choice(
        len(origins), size=(bootstrap_samples, len(origins)), replace=True,
    )
    estimates = (
        clusters["sum"].to_numpy()[sampled].sum(axis=1)
        / clusters["count"].to_numpy()[sampled].sum(axis=1)
        * 100
    )
    return np.quantile(estimates, [0.025, 0.975])


def headline_metrics(frame, bootstrap_samples=2000, seed=42):
    final = frame.loc[frame.h.eq(96)].copy()
    if final.empty:
        raise ValueError("headline metrics require h=96 rows")
    diagnostics = []
    for model, group in final.groupby("model", sort=True):
        actual = group.y.to_numpy() / group["last"].to_numpy() - 1
        predicted = group.q50.to_numpy() / group["last"].to_numpy() - 1
        residual = np.sum((actual - predicted) ** 2)
        total = np.sum((actual - actual.mean()) ** 2)
        low, high = _direction_interval(group, bootstrap_samples, seed)
        diagnostics.append({
            "model": model,
            "return_mae": np.mean(np.abs(actual - predicted)),
            "return_r2": 1 - residual / total if total > 0 else np.nan,
            "return_correlation": np.corrcoef(actual, predicted)[0, 1],
            "direction_accuracy": np.mean(
                np.sign(predicted) == np.sign(actual)
            ) * 100,
            "direction_ci_low": low,
            "direction_ci_high": high,
        })
    return score(final).join(pd.DataFrame(diagnostics).set_index("model"))


def grouped_interval_metrics(frame, keys):
    rows = []
    for values, group in frame.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        metrics = score(group).reset_index().iloc[0].to_dict()
        rows.append(dict(zip(keys, values)) | metrics)
    return pd.DataFrame(rows).set_index(keys)


def grouped_headline_metrics(frame, keys):
    rows = []
    for values, group in frame.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        metrics = headline_metrics(group).reset_index().iloc[0].to_dict()
        rows.append(dict(zip(keys, values)) | metrics)
    return pd.DataFrame(rows).set_index(keys)


def tft_metric_tables(calibration, test):
    cutpoints = fit_regime_cutpoints(calibration)
    return {
        "overall": headline_metrics(test),
        "by_horizon": grouped_interval_metrics(test, ["model", "h"]),
        "by_asset": grouped_headline_metrics(test, ["model", "asset"]),
        "by_regime": grouped_headline_metrics(
            apply_regimes(test, cutpoints), ["model", "regime"],
        ),
    }
