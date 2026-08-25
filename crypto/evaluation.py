import json
from pathlib import Path

import pandas as pd


REQUIRED = [
    "model", "asset", "origin", "fold", "h", "y", "last", "sigma", "rv",
    "q10", "q50", "q90",
]
KEY = ["model", "asset", "origin", "h"]


def default_run_dir(tag, data_end, run_id, root=Path("artifacts/evaluation")):
    stamp = pd.Timestamp(data_end).strftime("%Y%m%d")
    return Path(root) / f"daily-{tag}-{stamp}-{run_id}"


def validate_predictions(frame):
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"missing prediction columns: {missing}")
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
