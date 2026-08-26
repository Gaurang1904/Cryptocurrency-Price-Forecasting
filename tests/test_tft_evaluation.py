import unittest

import numpy as np
import pandas as pd

from neural.tft_evaluation import (
    apply_regimes, calibrate_tft_intervals, fit_regime_cutpoints,
    headline_metrics, make_tft_baselines, split_calibration_test,
    tft_metric_tables,
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


def asymmetric_interval_fixture():
    frame, _ = forecast_fixture(origins=8, horizons=2, assets=("BTC",))
    ordered = sorted(frame.origin.unique())
    calibration = frame[frame.origin.isin(ordered[:5])].copy().assign(split="calibration")
    test = frame[frame.origin.isin(ordered[5:])].copy().assign(split="test")
    calibration.loc[:, "y"] = 100.0
    calibration.loc[:, "q10"] = 101.0
    calibration.loc[:, "q50"] = 102.0
    calibration.loc[:, "q90"] = 104.0
    return calibration, test


def metric_fixture_with_price_level_trend():
    origins = pd.date_range("2025-01-01", periods=30, freq="D", tz="UTC")
    rows = []
    for model in ("good_returns", "bad_returns"):
        for index, origin in enumerate(origins):
            last = 1000.0 + index * 10
            actual_return = 0.01 if index % 2 == 0 else -0.01
            predicted_return = actual_return if model == "good_returns" else -actual_return
            actual, median = last * (1 + actual_return), last * (1 + predicted_return)
            rows.append({
                "model": model, "asset": "BTC", "origin": origin,
                "target_time": origin + pd.Timedelta("1D"), "split": "test",
                "h": 96, "y": actual, "last": last, "regime_driver": 0.02,
                "origin_sigma": 0.001, "origin_momentum": 0.0,
                "q10": median * 0.98, "q50": median, "q90": median * 1.02,
            })
    return pd.DataFrame(rows)


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

    def test_conversion_preserves_horizons_for_non_default_cv_indices(self):
        cv, context = cv_fixture()
        cv.index = np.arange(10, 10 + len(cv))
        frame = to_tft_forecasts(cv, context, model="tft_raw")
        self.assertEqual(sorted(frame.h.unique()), [1, 2, 3])
        self.assertFalse(frame.h.isna().any())

    def test_validation_rejects_non_positive_price_space_fields(self):
        reference, _ = forecast_fixture()
        for column in ("y", "last", "q10", "q50", "q90"):
            for value in (-1.0, 0.0):
                with self.subTest(column=column, value=value):
                    frame = reference.copy()
                    frame.loc[frame.index[0], column] = value
                    with self.assertRaisesRegex(ValueError, "strictly positive"):
                        validate_tft_forecasts(frame, expected_horizons=(1, 2, 3))


class TftCalibrationMetricTests(unittest.TestCase):
    def test_calibration_uses_only_calibration_rows_and_never_tightens(self):
        calibration, test = asymmetric_interval_fixture()
        first = calibrate_tft_intervals(calibration, test, alpha=0.20)
        corrupted = test.assign(y=test.y * 100)
        second = calibrate_tft_intervals(calibration, corrupted, alpha=0.20)
        np.testing.assert_allclose(first.q10, second.q10)
        np.testing.assert_allclose(first.q90, second.q90)
        self.assertTrue((first.q10 <= test.q10).all())
        self.assertTrue((first.q90 >= test.q90).all())
        self.assertNotEqual(
            np.log(test.q10.iloc[0]) - np.log(first.q10.iloc[0]),
            np.log(first.q90.iloc[0]) - np.log(test.q90.iloc[0]),
        )

    def test_calibration_preserves_median_and_requires_matching_model(self):
        calibration, test = asymmetric_interval_fixture()
        calibrated = calibrate_tft_intervals(calibration, test, alpha=0.20)
        np.testing.assert_allclose(calibrated.q50, test.q50)
        with self.assertRaisesRegex(ValueError, "same single model"):
            calibrate_tft_intervals(
                calibration.assign(model="other"), test, alpha=0.20,
            )

    def test_calibration_requires_rows_for_every_test_horizon(self):
        calibration, test = asymmetric_interval_fixture()
        with self.assertRaisesRegex(ValueError, "horizon 2"):
            calibrate_tft_intervals(
                calibration.loc[calibration.h.eq(1)], test, alpha=0.20,
            )

    def test_headline_metrics_use_only_horizon_96_and_return_r_squared(self):
        frame = metric_fixture_with_price_level_trend()
        metrics = headline_metrics(frame, bootstrap_samples=200, seed=42)
        self.assertEqual(set(metrics.index), set(frame.model))
        self.assertIn("pinball_%", metrics)
        self.assertIn("direction_accuracy", metrics)
        self.assertIn("direction_ci_low", metrics)
        self.assertIn("return_r2", metrics)
        self.assertLess(metrics.loc["bad_returns", "return_r2"], 0)

    def test_headline_metrics_reject_frames_without_next_day_rows(self):
        frame = metric_fixture_with_price_level_trend().assign(h=95)
        with self.assertRaisesRegex(ValueError, "h=96"):
            headline_metrics(frame)

    def test_direction_interval_resamples_complete_multi_asset_origins(self):
        origins = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
        rows = []
        for index, origin in enumerate(origins):
            for asset in ("BTC", "ETH"):
                actual = 101.0 if asset == "BTC" else 102.0
                if index < 2:
                    median = 100.5 if asset == "BTC" else 101.5
                else:
                    median = 99.0 if asset == "BTC" else 98.5
                rows.append({
                    "model": "m", "asset": asset, "origin": origin, "h": 96,
                    "y": actual, "last": 100.0, "q10": 90.0,
                    "q50": median, "q90": 110.0,
                })
        metrics = headline_metrics(
            pd.DataFrame(rows), bootstrap_samples=200, seed=7,
        )
        self.assertEqual(metrics.loc["m", "direction_ci_low"], 0.0)
        self.assertEqual(metrics.loc["m", "direction_ci_high"], 100.0)

    def test_regimes_use_unique_calibration_origins_and_fixed_cutpoints(self):
        calibration = pd.DataFrame({
            "asset": ["BTC", "BTC", "BTC", "BTC", "BTC", "ETH", "SOL"],
            "origin": pd.to_datetime([
                "2025-01-01", "2025-01-01", "2025-01-01", "2025-01-01",
                "2025-01-01", "2025-01-02", "2025-01-03",
            ], utc=True),
            "regime_driver": [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03],
        })
        cutpoints = fit_regime_cutpoints(calibration)
        test = pd.DataFrame({"regime_driver": [0.01, 0.02, 0.03], "y": [1, 2, 3]})
        labelled = apply_regimes(test, cutpoints)
        self.assertEqual(labelled.regime.tolist(), ["low", "medium", "high"])
        changed = apply_regimes(test.assign(y=[300, 200, 100]), cutpoints)
        self.assertEqual(changed.regime.tolist(), labelled.regime.tolist())

    def test_metric_tables_include_fixed_regime_and_subgroup_diagnostics(self):
        test = metric_fixture_with_price_level_trend()
        test["regime_driver"] = np.tile(
            np.linspace(0.01, 0.03, 30), test.model.nunique(),
        )
        calibration = test.loc[test.model.eq("good_returns")].copy()
        tables = tft_metric_tables(calibration, test)
        self.assertEqual(
            set(tables), {"overall", "by_horizon", "by_asset", "by_regime"},
        )
        self.assertIn("return_r2", tables["by_asset"])
        self.assertIn("coverage", tables["by_horizon"])
        self.assertEqual(
            set(tables["by_regime"].index.get_level_values("regime")),
            {"low", "medium", "high"},
        )
