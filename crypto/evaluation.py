import json
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


def default_run_dir(tag, data_end, run_id, root=Path("artifacts/evaluation")):
    stamp = pd.Timestamp(data_end).strftime("%Y%m%d")
    return Path(root) / f"daily-{tag}-{stamp}-{run_id}"


def new_run_dir(tag, data_end, run_id=None, root=Path("artifacts/evaluation")):
    run_id = run_id or uuid4().hex[:12]
    output_dir = default_run_dir(tag, data_end, run_id, root)
    if output_dir.exists():
        raise FileExistsError(f"run directory already exists: {output_dir}")
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


def save_predictions(frame, output_dir, metadata):
    validate_predictions(frame)
    output_dir = Path(output_dir)
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
