import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from neural.tft_evaluation import make_tft_baselines
from neural.tft_plots import render_tft_report
from tests.test_tft_evaluation import forecast_fixture


def report_fixture():
    frame, origins = forecast_fixture(origins=6, horizons=96, assets=("BTC", "ETH"))
    final = frame.h.eq(96)
    origin_steps = frame.origin.factorize()[0]
    asset_steps = frame.asset.map({asset: index for index, asset in enumerate(frame.asset.unique())})
    variation = 1 + (origin_steps + asset_steps) * 0.0001
    frame.loc[final, "y"] *= variation[final]
    frame.loc[final, ["q10", "q50", "q90"]] = (
        frame.loc[final, ["q10", "q50", "q90"]].to_numpy() * variation[final].to_numpy()[:, None]
    )
    calibration = frame[frame.origin.isin(origins[:2])].copy().assign(split="calibration")
    raw_test = frame[frame.origin.isin(origins[2:])].copy().assign(split="test")
    calibrated = raw_test.assign(model="tft_calibrated")
    baselines = make_tft_baselines(raw_test)
    test = pd.concat([raw_test, calibrated, baselines], ignore_index=True)
    baseline_rows = test.model.isin({"persistence_vol", "momentum_vol"}) & test.h.eq(96)
    test_origin_steps = test.origin.factorize()[0]
    test_asset_steps = test.asset.map({asset: index for index, asset in enumerate(test.asset.unique())})
    test_variation = 1 + (test_origin_steps + test_asset_steps) * 0.0001
    test.loc[baseline_rows, ["q10", "q50", "q90"]] = (
        test.loc[baseline_rows, ["q10", "q50", "q90"]].to_numpy()
        * test_variation[baseline_rows].to_numpy()[:, None]
    )
    return calibration, test


class TftPlotTests(unittest.TestCase):
    def test_report_creates_exact_tables_and_graphs_and_closes_figures(self):
        calibration, test = report_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            paths = render_tft_report(calibration, test, Path(tmp))
            names = {path.name for path in paths}
            self.assertEqual(names, {
                "metrics_overall.csv", "metrics_by_horizon.csv",
                "metrics_by_asset.csv", "metrics_by_regime.csv",
                "forecast_path.png", "next_day_forecasts.png",
                "returns_scatter.png", "performance_by_asset.png",
                "performance_by_regime.png", "calibration_comparison.png",
            })
            self.assertEqual([], plt.get_fignums())

    def test_headline_graphs_reject_non_test_rows_and_missing_baselines(self):
        calibration, test = report_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "test"):
                render_tft_report(calibration, test.assign(split="calibration"), Path(tmp))
            with self.assertRaisesRegex(ValueError, "baseline"):
                render_tft_report(
                    calibration, test[test.model.str.startswith("tft")], Path(tmp),
                )
