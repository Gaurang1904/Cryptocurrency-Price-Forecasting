import json
import re
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from crypto.backtest import score


REQUIRED = [
    "model", "asset", "origin", "fold", "h", "y", "last", "sigma", "rv",
    "regime_driver", "q10", "q50", "q90",
]
KEY = ["model", "asset", "origin", "h"]
FORECAST_KEY = ["asset", "origin", "h"]
SHARED = ["fold", "y", "last", "rv", "regime_driver"]
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def default_run_dir(tag, data_end, run_id, root=Path("artifacts/evaluation")):
    stamp = pd.Timestamp(data_end).strftime("%Y%m%d")
    return Path(root) / f"daily-{tag}-{stamp}-{run_id}"


def _validated_run_id(run_id):
    run_id = uuid4().hex[:12] if run_id is None else run_id
    if (
        not isinstance(run_id, str)
        or not _RUN_ID.fullmatch(run_id)
        or run_id.startswith(".")
        or run_id.endswith(".")
        or "/" in run_id
        or "\\" in run_id
    ):
        raise ValueError("run_id must be one safe, non-dot path component")
    return run_id


def reserve_run_dir(tag, data_end, run_id=None, root=Path("artifacts/evaluation")):
    """Atomically reserve an empty, contained run directory before fitting."""
    run_id = _validated_run_id(run_id)
    root = Path(root).resolve()
    output_dir = default_run_dir(tag, data_end, run_id, root).resolve()
    try:
        output_dir.relative_to(root)
    except ValueError:
        raise ValueError(f"run_id resolves outside output root: {run_id}") from None
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def validate_predictions(frame):
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"missing prediction columns: {missing}")
    if not pd.api.types.is_numeric_dtype(frame.regime_driver):
        raise ValueError("non-numeric regime_driver")
    if frame[REQUIRED].isna().any().any():
        raise ValueError("NaN predictions")
    if frame.duplicated(KEY).any():
        raise ValueError("duplicate forecast keys")
    if ((frame.q10 > frame.q50) | (frame.q50 > frame.q90)).any():
        raise ValueError("crossed quantiles")
    if not frame.h.between(1, 7).all():
        raise ValueError("mismatched horizons")


def _metadata_count(metadata, name, actual, bundle_index):
    try:
        declared = int(metadata[name])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"bundle {bundle_index} metadata {name} is missing or invalid") from None
    if declared != actual:
        raise ValueError(
            f"bundle {bundle_index} metadata {name}={declared} does not match rows={actual}"
        )


def validate_comparable_predictions(frames, metadata):
    """Require complete, identical grids and shared outcomes before publication."""
    if not frames or len(frames) != len(metadata):
        raise ValueError("prediction frames and metadata must be non-empty and aligned")

    model_views = {}
    for bundle_index, (frame, details) in enumerate(zip(frames, metadata), start=1):
        validate_predictions(frame)
        models = sorted(frame.model.unique().tolist())
        if not models:
            raise ValueError(f"bundle {bundle_index} contains no models")
        if "models" in details and sorted(details["models"]) != models:
            raise ValueError(f"bundle {bundle_index} metadata models do not match rows")

        try:
            horizon_count = int(details["horizons"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"bundle {bundle_index} metadata horizons is missing or invalid"
            ) from None
        expected_horizons = set(range(1, horizon_count + 1))
        complete = frame.groupby(["model", "asset", "origin"])["h"].agg(
            lambda values: set(int(value) for value in values) == expected_horizons
        )
        if not complete.all():
            raise ValueError(f"bundle {bundle_index} does not contain complete horizons")

        origins = frame[["asset", "origin"]].drop_duplicates().shape[0]
        _metadata_count(details, "horizons", len(frame.h.unique()), bundle_index)
        _metadata_count(details, "folds", frame.fold.nunique(), bundle_index)
        _metadata_count(details, "origins", origins, bundle_index)
        expected_rows = len(models) * origins * horizon_count
        if len(frame) != expected_rows:
            raise ValueError(
                f"bundle {bundle_index} row count {len(frame)} does not match "
                f"models*origins*horizons={expected_rows}"
            )

        for model, group in frame.groupby("model", sort=False):
            if model in model_views:
                raise ValueError(f"model {model!r} appears in multiple input bundles")
            model_views[model] = group.sort_values(FORECAST_KEY).reset_index(drop=True)

    reference_name, reference = next(iter(model_views.items()))
    reference_keys = pd.MultiIndex.from_frame(reference[FORECAST_KEY])
    for model, view in model_views.items():
        keys = pd.MultiIndex.from_frame(view[FORECAST_KEY])
        if not keys.equals(reference_keys):
            raise ValueError(
                f"models {reference_name!r} and {model!r} do not have identical forecast key sets"
            )
        for column in SHARED:
            if not view[column].reset_index(drop=True).equals(
                reference[column].reset_index(drop=True)
            ):
                raise ValueError(
                    f"models {reference_name!r} and {model!r} do not have shared {column} values"
                )


def save_predictions(frame, output_dir, metadata, reserved=False):
    validate_predictions(frame)
    output_dir = Path(output_dir)
    if reserved:
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(
                f"reserved run directory is unavailable or not empty: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    return output_dir


def _group_score(frame, keys):
    rows = []
    for values, group in frame.groupby(keys, sort=False):
        values = values if isinstance(values, tuple) else (values,)
        row = score(group).reset_index().iloc[0].to_dict()
        rows.append(dict(zip(keys, values)) | row)
    return pd.DataFrame(rows).set_index(keys)


def _with_regime_labels(frame):
    """Label each test fold by its observed origin-time volatility terciles.

    Labels are descriptive only: their cutpoints use the fold's persisted
    predictors and never enter fitting, calibration, or model selection.
    """
    labelled = frame.copy()
    for _, group in labelled.groupby("fold", sort=False):
        drivers = group[["asset", "origin", "regime_driver"]].drop_duplicates()
        low, high = drivers.regime_driver.quantile([1 / 3, 2 / 3])
        labelled.loc[group.index, "regime"] = np.select(
            [group.regime_driver <= low, group.regime_driver <= high],
            ["low", "medium"], default="high",
        )
    return labelled


def metric_tables(frame):
    validate_predictions(frame)
    return {
        "overall": score(frame),
        "by_horizon": _group_score(frame, ["model", "h"]),
        "by_asset": _group_score(frame, ["model", "asset"]),
        "by_fold": _group_score(frame, ["model", "fold"]),
        "by_regime": _group_score(
            _with_regime_labels(frame), ["model", "fold", "regime"]
        ),
    }
