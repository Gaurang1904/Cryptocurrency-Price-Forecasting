import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import patch

from crypto.evaluation import (
    default_run_dir,
    metric_tables,
    save_predictions,
    validate_predictions,
)


def valid_frame():
    return pd.DataFrame({
        "model": ["xgb"], "asset": ["BTC"],
        "origin": [pd.Timestamp("2025-01-01", tz="UTC")],
        "fold": [pd.Timestamp("2025-01-01", tz="UTC")],
        "h": [1], "y": [101.0], "last": [100.0],
        "sigma": [0.02], "rv": [0.018], "regime_driver": [0.02],
        "q10": [95.0], "q50": [100.0], "q90": [105.0],
    })


class PredictionValidationTests(unittest.TestCase):
    def test_run_directory_contains_pipeline_and_data_cutoff(self):
        got = default_run_dir(
            "tree", pd.Timestamp("2026-07-23", tz="UTC"), "baseline", Path("out")
        )
        self.assertEqual(got, Path("out/daily-tree-20260723-baseline"))

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

    def test_missing_regime_driver_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "regime_driver"):
            validate_predictions(valid_frame().drop(columns="regime_driver"))

    def test_non_numeric_regime_driver_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "numeric regime_driver"):
            validate_predictions(valid_frame().assign(regime_driver="high"))

    def test_metric_tables_report_coverage_by_horizon_and_asset(self):
        frame = pd.concat([
            valid_frame(),
            valid_frame().assign(
                asset="ETH", y=110.0,
                origin=pd.Timestamp("2025-01-02", tz="UTC"),
            ),
        ], ignore_index=True)
        tables = metric_tables(frame)
        self.assertEqual(tables["overall"].loc["xgb", "coverage"], 50.0)
        self.assertEqual(tables["by_horizon"].loc[("xgb", 1), "coverage"], 50.0)
        self.assertEqual(tables["by_fold"].loc[(
            "xgb", pd.Timestamp("2025-01-01", tz="UTC")
        ), "coverage"], 50.0)
        self.assertEqual(tables["by_asset"].loc[("xgb", "BTC"), "coverage"], 100.0)
        self.assertEqual(tables["by_asset"].loc[("xgb", "ETH"), "coverage"], 0.0)

    def test_metric_tables_label_regimes_from_fold_drivers_not_outcomes(self):
        frame = pd.concat([
            valid_frame().assign(asset="BTC", origin=pd.Timestamp("2025-01-01", tz="UTC"), regime_driver=0.01),
            valid_frame().assign(asset="ETH", origin=pd.Timestamp("2025-01-02", tz="UTC"), regime_driver=0.02, y=110.0),
            valid_frame().assign(asset="SOL", origin=pd.Timestamp("2025-01-03", tz="UTC"), regime_driver=0.03),
        ], ignore_index=True)
        tables = metric_tables(frame)
        labels = tables["by_regime"].index.get_level_values("regime").tolist()
        self.assertEqual(labels, ["low", "medium", "high"])
        self.assertEqual(tables["by_regime"].loc[("xgb", pd.Timestamp("2025-01-01", tz="UTC"), "medium"), "coverage"], 0.0)

        changed_outcomes = frame.assign(y=[110.0, 101.0, 110.0])
        changed = metric_tables(changed_outcomes)
        self.assertEqual(changed["by_regime"].index.tolist(), tables["by_regime"].index.tolist())

    def test_neural_backtest_keeps_test_provenance_aligned_with_sigma_and_rv(self):
        from neural import core

        columns = {"asset": ["BTC", "ETH"], "date": pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True),
                   "close": [100.0, 200.0], "vol_21d": [0.21, 0.29]}
        for h in range(1, core.H + 1):
            columns[f"y{h}"] = [0.01, 0.02]
            columns[f"rv{h}"] = [0.011 * h, 0.012 * h]
        test = pd.DataFrame(columns)
        train = pd.concat([test, test], ignore_index=True)
        fold = pd.Timestamp("2026-01-01", tz="UTC")
        cal_prediction = np.log(np.full((2, core.H), 0.1))
        test_prediction = np.log(np.array([np.arange(0.21, 0.21 + core.H * 0.01, 0.01), np.arange(0.29, 0.29 + core.H * 0.001, 0.001)]))

        with (patch.object(core, "build", return_value=(test, [])),
              patch.object(core.pd, "read_parquet", return_value=test),
              patch.object(core, "channel_windows", return_value=(np.zeros((2, 1, 1)), test[["asset", "date"]])),
              patch.object(core, "run_folds", return_value=[(train, test, fold)]),
              patch.object(core, "split_calibration", return_value=(train, test)),
              patch.object(core, "_lookup", side_effect=lambda _win, _idx, rows: (np.zeros((len(rows), 1, 1)), np.ones(len(rows), dtype=bool))),
              patch.object(core, "train_net", side_effect=lambda model, *_args: model),
              patch.object(core, "_predict", side_effect=[cal_prediction, test_prediction]),
              patch.object(core, "calibrate", return_value={0.1: 1.0, 0.5: 1.0, 0.9: 1.0}),
              patch.object(core, "save_predictions"),
              patch.object(core, "bands", side_effect=lambda _z, last, _sigma, _h: {0.1: last * 0.9, 0.5: last, 0.9: last * 1.1})):
            res, _ = core.backtest({"test-net": lambda _channels: object()})

        h1 = res[res.h == 1]
        self.assertListEqual(h1.origin.tolist(), test.date.tolist())
        self.assertListEqual(h1.fold.tolist(), [fold, fold])
        self.assertListEqual(h1.sigma.tolist(), [0.21, 0.29])
        self.assertListEqual(h1.rv.tolist(), [0.011, 0.012])
        self.assertListEqual(h1.regime_driver.tolist(), [0.21, 0.29])

    def test_tree_backtest_keeps_test_provenance_aligned_with_sigma_and_rv(self):
        from tree import run

        columns = {"asset": ["BTC", "ETH"], "date": pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True),
                   "close": [100.0, 200.0], "vol_21d": [0.21, 0.29]}
        for h in range(1, run.H + 1):
            columns[f"y{h}"] = [0.01, 0.02]
            columns[f"rv{h}"] = [0.011 * h, 0.012 * h]
        test = pd.DataFrame(columns)
        train = pd.concat([test, test], ignore_index=True)
        fold = pd.Timestamp("2026-01-01", tz="UTC")

        with (patch.object(run, "build", return_value=(test, [])),
              patch.object(run.pd, "read_parquet", return_value=test),
              patch.object(run, "run_folds", return_value=[(train, test, fold)]),
              patch.object(run, "split_calibration", return_value=(train, test)),
              patch.object(run, "model_sigma", return_value=(np.full(2, 0.1), np.array([0.41, 0.49]))),
              patch.object(run, "calibrate", return_value={0.1: 1.0, 0.5: 1.0, 0.9: 1.0}),
              patch.object(run, "save_predictions"),
              patch.object(run, "bands", side_effect=lambda _z, last, _sigma, _h: {0.1: last * 0.9, 0.5: last, 0.9: last * 1.1})):
            res, _ = run.backtest({"test-tree": lambda *_args: object()})

        h1 = res[(res.model == "test-tree") & (res.h == 1)]
        self.assertListEqual(h1.origin.tolist(), test.date.tolist())
        self.assertListEqual(h1.fold.tolist(), [fold, fold])
        self.assertListEqual(h1.sigma.tolist(), [0.41, 0.49])
        self.assertListEqual(h1.rv.tolist(), [0.011, 0.012])
        self.assertListEqual(h1.regime_driver.tolist(), [0.21, 0.29])

    def test_save_writes_predictions_and_metadata_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = save_predictions(valid_frame(), Path(tmp) / "run-1", {"pipeline": "daily"})
            self.assertTrue((out / "predictions.parquet").exists())
            self.assertTrue((out / "metadata.json").exists())
            with self.assertRaises(FileExistsError):
                save_predictions(valid_frame(), Path(tmp) / "run-1", {"pipeline": "daily"})
