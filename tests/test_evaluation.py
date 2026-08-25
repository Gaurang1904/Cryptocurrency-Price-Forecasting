import tempfile
import unittest
from pathlib import Path

import pandas as pd

from crypto.evaluation import save_predictions, validate_predictions


def valid_frame():
    return pd.DataFrame({
        "model": ["xgb"], "asset": ["BTC"],
        "origin": [pd.Timestamp("2025-01-01", tz="UTC")],
        "fold": [pd.Timestamp("2025-01-01", tz="UTC")],
        "h": [1], "y": [101.0], "last": [100.0],
        "sigma": [0.02], "rv": [0.018],
        "q10": [95.0], "q50": [100.0], "q90": [105.0],
    })


class PredictionValidationTests(unittest.TestCase):
    def test_valid_predictions_are_accepted(self):
        validate_predictions(valid_frame())

    def test_crossed_quantiles_are_rejected(self):
        frame = valid_frame().assign(q10=106.0)
        with self.assertRaisesRegex(ValueError, "crossed quantiles"):
            validate_predictions(frame)

    def test_duplicate_forecast_keys_are_rejected(self):
        frame = pd.concat([valid_frame(), valid_frame()], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate forecast keys"):
            validate_predictions(frame)

    def test_save_writes_predictions_and_metadata_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = save_predictions(valid_frame(), Path(tmp) / "run-1", {"pipeline": "daily"})
            self.assertTrue((out / "predictions.parquet").exists())
            self.assertTrue((out / "metadata.json").exists())
            with self.assertRaises(FileExistsError):
                save_predictions(valid_frame(), Path(tmp) / "run-1", {"pipeline": "daily"})
