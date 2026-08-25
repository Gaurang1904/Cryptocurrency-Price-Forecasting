"""Calibration-selected convex blends of aligned prediction frames."""

import numpy as np

from crypto.backtest import score
from crypto.evaluation import validate_predictions


KEY = ["asset", "origin", "fold", "h"]
MATCH = ["y", "last", "rv", "regime_driver"]
BLEND = ["sigma", "q10", "q50", "q90"]


def blend_predictions(left, right, weight, model="tree_blend"):
    """Blend aligned forecasts, with ``weight`` applied to ``left``."""
    if not np.isscalar(weight) or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("weight must be between 0 and 1")

    for name, frame in (("left", left), ("right", right)):
        missing = sorted(set(KEY) - set(frame.columns))
        if missing:
            raise ValueError(f"missing {name} forecast keys: {missing}")
        if frame.duplicated(KEY).any():
            raise ValueError(f"duplicate {name} forecast keys")
        validate_predictions(frame)

    left_index = left.set_index(KEY)
    right_index = right.set_index(KEY)
    if set(left_index.index) != set(right_index.index):
        raise ValueError("forecast keys do not match")
    right_index = right_index.reindex(left_index.index)

    for column in MATCH:
        if not np.array_equal(left_index[column].to_numpy(), right_index[column].to_numpy()):
            raise ValueError(f"mismatched {column}")

    result = left_index.reset_index().copy()
    result["model"] = model
    for column in BLEND:
        result[column] = (
            weight * left_index[column].to_numpy()
            + (1 - weight) * right_index[column].to_numpy()
        )
    validate_predictions(result)
    return result


def select_weight(left, right, grid=None):
    """Choose a blend weight from only the supplied calibration forecasts."""
    weights = [float(weight) for weight in (np.linspace(0, 1, 21) if grid is None else grid)]
    if not weights:
        raise ValueError("weight grid must not be empty")
    losses = {
        weight: score(blend_predictions(left, right, weight)).loc["tree_blend", "pinball_%"]
        for weight in weights
    }
    best = min(losses.values())
    tied = [weight for weight, loss in losses.items() if loss == best]
    return min(tied, key=lambda weight: (abs(weight - 0.5), weight))
