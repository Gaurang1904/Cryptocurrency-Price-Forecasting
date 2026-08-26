import unittest

import numpy as np
import pandas as pd

from neural.tft_evaluation import (
    make_tft_baselines, split_calibration_test,
    to_tft_forecasts, validate_tft_forecasts,
)


def cv_fixture(origins=2, horizons=3, assets=("BTC", "ETH")):
    cutoffs = pd.date_range("2025-01-01 23:45", periods=origins, freq="D")
    rows, context = [], []
    for asset_index, asset in enumerate(assets, start=1):
        asset_offset = asset_index * 100.0
        for origin_index, cutoff in enumerate(cutoffs):
            last = asset_offset + origin_index
            context.append({
                "asset": asset, "ds": cutoff, "close": last,
                "rv_672": 0.02, "sigma_672": 0.001,
                "momentum_96": 0.01, "missing_bar": 0,
            })
            for h in range(1, horizons + 1):
                target_log = np.log(last * (1 + 0.001 * h))
                rows.append({
                    "index": len(rows), "unique_id": asset,
                    "ds": cutoff + pd.Timedelta("15min") * h,
                    "cutoff": cutoff,
                    "TFT-lo-80.0": target_log - 0.02,
                    "TFT-median": target_log,
                    "TFT-hi-80.0": target_log + 0.02,
                    "y": target_log,
                })
    return pd.DataFrame(rows), pd.DataFrame(context)


def forecast_fixture(origins=4, horizons=3, assets=("BTC", "ETH")):
    origin_index = pd.date_range("2025-01-01", periods=origins, freq="D", tz="UTC")
    rows = []
    for asset_index, asset in enumerate(assets, start=1):
        for date_index, origin in enumerate(origin_index):
            last = asset_index * 100.0 + date_index
            for h in range(1, horizons + 1):
                actual = last * (1 + 0.0015 * h)
                predicted = last * (1 + 0.001 * h)
                rows.append({
                    "model": "tft_raw", "asset": asset, "origin": origin,
                    "target_time": origin + pd.Timedelta("15min") * h,
                    "split": "unassigned", "h": h, "y": actual, "last": last,
                    "regime_driver": 0.02 + date_index * 0.0001,
                    "origin_sigma": 0.001, "origin_momentum": 0.01,
                    "q10": predicted * 0.98, "q50": predicted,
                    "q90": predicted * 1.02,
                })
    return pd.DataFrame(rows), origin_index


class TftForecastFrameTests(unittest.TestCase):
    def test_conversion_uses_candle_close_times_and_price_space(self):
        cv, context = cv_fixture(origins=2, horizons=3, assets=("BTC", "ETH"))
        frame = to_tft_forecasts(cv, context, model="tft_raw")
        first = frame.sort_values(["asset", "origin", "h"]).iloc[0]
        self.assertEqual(first.origin.minute, 0)
        self.assertEqual(first.target_time, first.origin + pd.Timedelta("15min"))
        self.assertEqual(first.h, 1)
        self.assertAlmostEqual(first.q50, np.exp(cv.iloc[0]["TFT-median"]))
        validate_tft_forecasts(frame, expected_horizons=(1, 2, 3))

    def test_split_uses_219_calibration_and_146_untouched_origins(self):
        frame, origins = forecast_fixture(origins=365, horizons=2)
        calibration, test = split_calibration_test(
            frame, origins, n_calibration=219, n_test=146,
        )
        self.assertEqual(calibration.origin.nunique(), 219)
        self.assertEqual(test.origin.nunique(), 146)
        self.assertLess(calibration.origin.max(), test.origin.min())
        self.assertEqual(set(calibration.split), {"calibration"})
        self.assertEqual(set(test.split), {"test"})

    def test_baselines_preserve_keys_truth_and_quantile_order(self):
        reference, _ = forecast_fixture(origins=4, horizons=3)
        baselines = make_tft_baselines(reference)
        self.assertEqual(set(baselines.model), {"persistence_vol", "momentum_vol"})
        self.assertFalse(baselines.duplicated(["model", "asset", "origin", "h"]).any())
        self.assertTrue((baselines.q10 <= baselines.q50).all())
        self.assertTrue((baselines.q50 <= baselines.q90).all())
        for _, group in baselines.groupby("model"):
            pd.testing.assert_frame_equal(
                group[["asset", "origin", "h", "y", "last"]].reset_index(drop=True),
                reference[["asset", "origin", "h", "y", "last"]].reset_index(drop=True),
            )

    def test_validation_rejects_missing_schema_values(self):
        frame, _ = forecast_fixture()
        with self.assertRaisesRegex(ValueError, "missing values"):
            validate_tft_forecasts(frame.assign(split=np.nan), expected_horizons=(1, 2, 3))
