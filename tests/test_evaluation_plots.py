import hashlib
import json
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

    def test_cli_writes_hashed_manifest_and_never_overwrites_output(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            tree_dir, neural_dir = root / "tree", root / "neural"
            tree_dir.mkdir()
            neural_dir.mkdir()
            tree_predictions = tree_dir / "predictions.parquet"
            neural_predictions = neural_dir / "predictions.parquet"
            tree_frame = evaluation_frame()
            neural_frame = tree_frame[tree_frame.model == "xgb"].assign(model="lstm")
            tree_frame.to_parquet(tree_predictions, index=False)
            neural_frame.to_parquet(neural_predictions, index=False)
            metadata = {
                "pipeline": "daily", "data_end": "2025-01-10 00:00:00+00:00",
                "horizons": 2, "folds": 1, "origins": 3,
            }
            (tree_dir / "metadata.json").write_text(
                json.dumps(metadata | {"family": "tree", "features": ["vol_21d"]})
            )
            (neural_dir / "metadata.json").write_text(
                json.dumps(metadata | {"family": "neural", "lookback": 30})
            )

            out = root / "report"
            main([str(tree_predictions), str(neural_predictions), "--out", str(out)])

            csvs = {
                "metrics_overall.csv", "metrics_by_horizon.csv",
                "metrics_by_asset.csv", "metrics_by_fold.csv",
                "metrics_by_regime.csv",
            }
            pngs = {
                "forecast_bands.png", "coverage_by_horizon.png",
                "pinball_by_model.png", "volatility_fit.png",
                "performance_by_asset.png", "performance_by_regime.png",
            }
            self.assertEqual(csvs, {path.name for path in out.glob("metrics_*.csv")})
            manifest_path = out / "manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["manifest_version"], 1)
            self.assertEqual(manifest["hash_algorithm"], "sha256")
            self.assertEqual(manifest["evaluation"], {
                "data_cutoff": "2025-01-10T00:00:00+00:00",
                "oos_start": "2025-01-01T00:00:00+00:00",
                "oos_end": "2025-01-03T00:00:00+00:00",
                "folds": ["2025-01-01T00:00:00+00:00"],
                "fold_count": 1,
                "distinct_origins": 3,
                "row_count": 9,
                "models": ["lstm", "vol_21d", "xgb"],
                "horizons": [1, 2],
            })
            self.assertEqual(
                {Path(item["path"]).name for item in manifest["inputs"]},
                {"predictions.parquet"},
            )
            self.assertTrue(all(not Path(item["path"]).is_absolute()
                                for item in manifest["inputs"]))
            self.assertEqual(
                {item["metadata"]["family"] for item in manifest["inputs"]},
                {"tree", "neural"},
            )
            for item, expected in zip(
                manifest["inputs"], [tree_predictions, neural_predictions]
            ):
                self.assertEqual(
                    item["sha256"], hashlib.sha256(expected.read_bytes()).hexdigest()
                )
            self.assertEqual(
                {Path(item["path"]).name for item in manifest["outputs"]},
                csvs | pngs,
            )
            for item in manifest["outputs"]:
                artifact = Path.cwd() / item["path"]
                self.assertEqual(
                    item["sha256"], hashlib.sha256(artifact.read_bytes()).hexdigest()
                )
            with self.assertRaises(FileExistsError):
                main([str(tree_predictions), str(neural_predictions),
                      "--out", str(out)])
