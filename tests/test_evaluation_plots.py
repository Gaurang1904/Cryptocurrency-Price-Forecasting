import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from crypto.evaluation_plots import render_bundle
from evaluate import main
from tests.test_evaluation import valid_frame


def evaluation_frame():
    base = valid_frame()
    origins = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    rows = []
    for model, sigma in (("vol_21d", 0.02), ("xgb", 0.018)):
        for i, origin in enumerate(origins):
            rows.append(base.assign(
                model=model, asset="BTC" if i < 2 else "ETH", origin=origin,
                fold=pd.Timestamp("2025-01-01", tz="UTC"), h=i % 2 + 1,
                y=101.0 + i, last=100.0, sigma=sigma, rv=0.017 + i * 0.001,
                regime_driver=0.01 + i * 0.01, q10=95.0 + i,
                q50=100.0 + i, q90=105.0 + i,
            ))
    return pd.concat(rows, ignore_index=True)


class EvaluationPlotTests(unittest.TestCase):
    def test_render_bundle_creates_all_expected_charts_and_closes_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            render_bundle(evaluation_frame(), out)
            self.assertEqual({
                "forecast_bands.png", "coverage_by_horizon.png",
                "pinball_by_model.png", "volatility_fit.png",
                "performance_by_asset.png", "performance_by_regime.png",
            }, {path.name for path in out.glob("*.png")})
            self.assertEqual([], plt.get_fignums())

    def test_render_bundle_rejects_artifacts_without_volatility_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "vol_21d baseline"):
                render_bundle(valid_frame(), Path(tmp))

    def test_cli_writes_five_metric_tables_and_never_overwrites_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.parquet"
            evaluation_frame().to_parquet(predictions, index=False)
            out = root / "report"
            main([str(predictions), "--out", str(out)])
            self.assertEqual({
                "metrics_overall.csv", "metrics_by_horizon.csv",
                "metrics_by_asset.csv", "metrics_by_fold.csv",
                "metrics_by_regime.csv",
            }, {path.name for path in out.glob("metrics_*.csv")})
            with self.assertRaises(FileExistsError):
                main([str(predictions), "--out", str(out)])
