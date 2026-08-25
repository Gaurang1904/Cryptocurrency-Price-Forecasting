import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import patch

from crypto.ensemble import blend_predictions, select_weight
from tests.test_evaluation import valid_frame


class EnsembleTests(unittest.TestCase):
    def test_blend_aligns_keys_and_preserves_prediction_semantics(self):
        left = pd.concat([
            valid_frame().assign(model="xgb", asset="BTC", q10=90.0, q50=100.0, q90=110.0,
                                 sigma=0.02),
            valid_frame().assign(model="xgb", asset="ETH", q10=180.0, q50=200.0, q90=220.0,
                                 y=202.0, last=200.0, sigma=0.04, rv=0.036,
                                 regime_driver=0.04),
        ], ignore_index=True)
        right = pd.concat([
            valid_frame().assign(model="lgbm", asset="ETH", q10=188.0, q50=204.0, q90=228.0,
                                 y=202.0, last=200.0, sigma=0.06, rv=0.036,
                                 regime_driver=0.04),
            valid_frame().assign(model="lgbm", asset="BTC", q10=94.0, q50=102.0, q90=114.0,
                                 sigma=0.04),
        ], ignore_index=True)

        got = blend_predictions(left, right, 0.25)

        self.assertListEqual(got.asset.tolist(), ["BTC", "ETH"])
        self.assertListEqual(got.model.tolist(), ["tree_blend", "tree_blend"])
        self.assertListEqual(got.q50.tolist(), [101.5, 203.0])
        self.assertAlmostEqual(got.sigma.iloc[0], 0.035)
        self.assertAlmostEqual(got.sigma.iloc[1], 0.055)
        self.assertListEqual(got.rv.tolist(), [0.018, 0.036])
        self.assertListEqual(got.regime_driver.tolist(), [0.02, 0.04])
        self.assertTrue((got.q10 <= got.q50).all() and (got.q50 <= got.q90).all())

    def test_blend_rejects_unaligned_or_duplicate_keys(self):
        left = valid_frame().assign(model="xgb")
        right = valid_frame().assign(model="lgbm", asset="ETH")
        with self.assertRaisesRegex(ValueError, "forecast keys"):
            blend_predictions(left, right, 0.5)

        duplicate = pd.concat([left, left], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            blend_predictions(duplicate, valid_frame().assign(model="lgbm"), 0.5)

    def test_blend_rejects_mismatched_shared_values(self):
        left = valid_frame().assign(model="xgb")
        for column in ("y", "last", "rv", "regime_driver"):
            right = valid_frame().assign(model="lgbm", **{column: [999.0]})
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, column):
                    blend_predictions(left, right, 0.5)

    def test_blend_rejects_invalid_weight(self):
        left = valid_frame().assign(model="xgb")
        right = valid_frame().assign(model="lgbm")
        with self.assertRaisesRegex(ValueError, "weight"):
            blend_predictions(left, right, 1.1)

    def test_weight_selection_uses_only_supplied_calibration_rows(self):
        left = valid_frame().assign(model="xgb", q10=99.0, q50=101.0, q90=103.0)
        right = valid_frame().assign(model="lgbm", q10=108.0, q50=110.0, q90=112.0)
        self.assertEqual(select_weight(left, right, grid=[0.0, 1.0]), 1.0)

    def test_weight_selection_breaks_equal_loss_ties_toward_half(self):
        left = valid_frame().assign(model="xgb", y=100.0, q10=99.0, q50=99.0, q90=99.0)
        right = valid_frame().assign(model="lgbm", y=100.0, q10=101.0, q50=101.0, q90=101.0)
        self.assertEqual(select_weight(left, right, grid=[0.0, 0.25, 0.75, 1.0]), 0.25)

    def test_tree_backtest_selects_on_calibration_and_persists_only_test_blend(self):
        from tree import run

        def rows(dates, closes):
            data = {
                "asset": ["BTC", "ETH"], "date": pd.to_datetime(dates, utc=True),
                "close": closes, "vol_21d": [0.03, 0.04],
            }
            for h in range(1, run.H + 1):
                data[f"y{h}"] = [0.01, 0.02]
                data[f"rv{h}"] = [0.011 * h, 0.012 * h]
            return pd.DataFrame(data)

        fit = rows(["2025-01-01", "2025-01-02"], [80.0, 160.0])
        cal = rows(["2025-02-01", "2025-02-02"], [90.0, 180.0])
        test = rows(["2026-01-02", "2026-01-03"], [100.0, 200.0])
        train = pd.concat([fit, cal], ignore_index=True)
        fold = pd.Timestamp("2026-01-01", tz="UTC")
        fitted = {"xgb": [], "lgbm": []}

        class ConstantModel:
            def __init__(self, sigma):
                self.sigma = sigma

            def predict(self, frame):
                return np.full(len(frame), np.log(self.sigma))

        def fitter(name, sigma):
            def fit_one(_fit, _cols, h):
                fitted[name].append(h)
                return ConstantModel(sigma)
            return fit_one

        selection_origins = []

        def choose(left, right):
            selection_origins.append((set(left.origin), set(right.origin)))
            return 0.25

        saved = {}

        def save(frame, _output_dir, metadata, reserved=False):
            saved["frame"] = frame.copy()
            saved["metadata"] = metadata

        with (tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp,
              patch.object(run, "build", return_value=(test, [])),
              patch.object(run.pd, "read_parquet", return_value=test),
              patch.object(run, "run_folds", return_value=[(train, test, fold)]),
              patch.object(run, "split_calibration", return_value=(fit, cal)),
              patch.object(run, "select_weight", side_effect=choose),
              patch.object(run, "save_predictions", side_effect=save)):
            result, vol_err = run.backtest(
                {
                    "xgb": fitter("xgb", 0.1),
                    "lgbm": fitter("lgbm", 0.2),
                },
                output_root=Path(tmp),
            )

        self.assertEqual(fitted, {"xgb": list(range(1, 8)), "lgbm": list(range(1, 8))})
        self.assertEqual(len(selection_origins), run.H)
        self.assertTrue(all(origins == (set(cal.date), set(cal.date)) for origins in selection_origins))
        self.assertSetEqual(set(result.model), {"xgb", "lgbm", "vol_21d", "tree_blend"})
        self.assertSetEqual(set(result.origin), set(test.date))
        self.assertTrue(np.allclose(result[result.model == "tree_blend"].sigma, 0.175))
        self.assertSetEqual(set(vol_err.model), {"xgb", "lgbm", "vol_21d", "tree_blend"})
        self.assertSetEqual(set(saved["frame"].origin), set(test.date))
        self.assertEqual(
            saved["metadata"]["blend_weights"],
            [{"fold": fold, "h": h, "weight": 0.25} for h in range(1, 8)],
        )


if __name__ == "__main__":
    unittest.main()
