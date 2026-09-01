import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd

from crypto.evaluation_plots import _forecast_bands, render_bundle
from evaluate import main
from tests.test_evaluation import valid_frame


def evaluation_frame():
    base = valid_frame()
    origins = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    rows = []
    models = (
        ("lstm", 0.017), ("dlinear", 0.0175), ("tree_blend", 0.018),
        ("xgb", 0.0185), ("vol_21d", 0.02), ("lgbm", 0.019),
    )
    for model, sigma in models:
        for i, origin in enumerate(origins):
            for h in range(1, 8):
                rows.append(base.assign(
                    model=model, asset="BTC" if i < 2 else "ETH", origin=origin,
                    fold=pd.Timestamp("2025-01-01", tz="UTC"), h=h,
                    y=101.0 + i + h, last=100.0, sigma=sigma,
                    rv=0.017 + i * 0.001 + h * 0.0001,
                    regime_driver=0.01 + i * 0.01, q10=95.0 + i,
                    q50=100.0 + i, q90=105.0 + i,
                ))
    return pd.concat(rows, ignore_index=True)


def forecast_panel_frame():
    base = valid_frame()
    origins = pd.date_range("2024-01-01", periods=289, freq="7D", tz="UTC")
    rows = []
    for offset, model in enumerate(
        ("lstm", "dlinear", "tree_blend", "xgb", "vol_21d", "lgbm")
    ):
        for i, origin in enumerate(origins):
            price = 100.0 + i * 0.1
            rows.append(base.assign(
                model=model, asset="BTC", origin=origin,
                fold=pd.Timestamp("2024-01-01", tz="UTC"), h=7,
                y=price + 1, last=price, sigma=0.02, rv=0.018,
                regime_driver=0.02, q10=price - 5 - offset * 0.1,
                q50=price + 0.5 + offset * 0.1,
                q90=price + 5 + offset * 0.1,
            ))
    return pd.concat(rows, ignore_index=True)


class EvaluationPlotTests(unittest.TestCase):
    def test_forecast_bands_requires_seven_day_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "7-day horizon"):
                _forecast_bands(forecast_panel_frame().assign(h=6), Path(tmp))

    def test_forecast_bands_requires_all_six_models(self):
        frame = forecast_panel_frame()
        frame = frame[frame.model != "dlinear"]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "six daily models"):
                _forecast_bands(frame, Path(tmp))

    def test_forecast_bands_has_six_panels_and_asset_caption(self):
        figures = []
        with patch("crypto.evaluation_plots._save",
                   side_effect=lambda fig, *_: figures.append(fig)):
            _forecast_bands(forecast_panel_frame(), Path("unused"))
        self.assertEqual(1, len(figures))
        fig = figures[0]
        self.assertEqual(
            {"lstm", "dlinear", "tree blend", "xgb", "vol 21d", "lgbm"},
            {ax.get_title() for ax in fig.axes},
        )
        self.assertIn("BTC 7-day forecast bands", fig._suptitle.get_text())
        self.assertIn("289 origins", fig._suptitle.get_text())
        plt.close(fig)

    def test_render_bundle_creates_all_expected_charts_and_closes_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            render_bundle(evaluation_frame(), out)
            self.assertEqual({
                "forecast_bands_btc.png", "forecast_bands_eth.png",
                "coverage_by_horizon.png",
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
            frame = evaluation_frame()
            tree_frame = frame[frame.model.isin(
                ["vol_21d", "xgb", "lgbm", "tree_blend"]
            )]
            neural_frame = frame[frame.model.isin(["lstm", "dlinear"])]
            tree_frame.to_parquet(tree_predictions, index=False)
            neural_frame.to_parquet(neural_predictions, index=False)
            metadata = {
                "pipeline": "daily", "data_end": "2025-01-10 00:00:00+00:00",
                "horizons": 7, "folds": 1, "origins": 3,
            }
            (tree_dir / "metadata.json").write_text(json.dumps(metadata | {
                "family": "tree", "features": ["vol_21d"],
                "output_dir": str(tree_dir.resolve()),
                "config": {"cache_path": str((tree_dir / "cache").resolve())},
            }))
            (neural_dir / "metadata.json").write_text(json.dumps(metadata | {
                "family": "neural", "lookback": 30,
                "output_dir": str(neural_dir.resolve()),
                "config": {"cache_path": str((neural_dir / "cache").resolve())},
            }))

            out = root / "report"
            main([
                str(tree_predictions), str(neural_predictions),
                "--out", str(out), "--provenance-root", str(root),
            ])

            csvs = {
                "metrics_overall.csv", "metrics_by_horizon.csv",
                "metrics_by_asset.csv", "metrics_by_fold.csv",
                "metrics_by_regime.csv",
            }
            pngs = {
                "forecast_bands_btc.png", "forecast_bands_eth.png",
                "coverage_by_horizon.png",
                "pinball_by_model.png", "volatility_fit.png",
                "performance_by_asset.png", "performance_by_regime.png",
            }
            self.assertEqual(csvs, {path.name for path in out.glob("metrics_*.csv")})
            manifest_path = out / "manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["manifest_version"], 1)
            self.assertEqual(manifest["hash_algorithm"], "sha256")
            self.assertEqual(manifest["provenance"], {
                "root": ".",
                "path_format": "provenance-root-relative-posix",
            })
            encoded = json.dumps(manifest)
            self.assertNotIn("absolute_path", encoded)
            self.assertNotRegex(encoded, r"[A-Za-z]:[/\\]")
            self.assertEqual(manifest["evaluation"], {
                "data_cutoff": "2025-01-10T00:00:00+00:00",
                "oos_start": "2025-01-01T00:00:00+00:00",
                "oos_end": "2025-01-03T00:00:00+00:00",
                "folds": ["2025-01-01T00:00:00+00:00"],
                "fold_count": 1,
                "distinct_origins": 3,
                "row_count": 126,
                "models": [
                    "dlinear", "lgbm", "lstm", "tree_blend", "vol_21d", "xgb",
                ],
                "horizons": [1, 2, 3, 4, 5, 6, 7],
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
            self.assertEqual(
                {item["metadata"]["output_dir"] for item in manifest["inputs"]},
                {"tree", "neural"},
            )
            self.assertEqual(
                {
                    item["metadata"]["config"]["cache_path"]
                    for item in manifest["inputs"]
                },
                {"tree/cache", "neural/cache"},
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
                artifact = root / item["path"]
                self.assertEqual(
                    item["sha256"], hashlib.sha256(artifact.read_bytes()).hexdigest()
                )
            with self.assertRaises(FileExistsError):
                main([
                    str(tree_predictions), str(neural_predictions),
                    "--out", str(out), "--provenance-root", str(root),
                ])

    def test_cli_rejects_input_outside_provenance_root(self):
        with (tempfile.TemporaryDirectory() as external_tmp,
              tempfile.TemporaryDirectory(dir=Path.cwd()) as root_tmp):
            external = Path(external_tmp)
            root = Path(root_tmp)
            predictions = external / "predictions.parquet"
            evaluation_frame().to_parquet(predictions, index=False)
            (external / "metadata.json").write_text(json.dumps({
                "pipeline": "daily", "family": "tree",
                "data_end": "2025-01-10 00:00:00+00:00",
                "horizons": 2, "folds": 1, "origins": 3,
            }), encoding="utf-8")
            out = root / "report"

            with self.assertRaisesRegex(ValueError, "outside provenance root"):
                main([
                    str(predictions), "--out", str(out),
                    "--provenance-root", str(root),
                ])
            self.assertFalse(out.exists())

    def test_cli_rejects_output_outside_provenance_root(self):
        with (tempfile.TemporaryDirectory() as external_tmp,
              tempfile.TemporaryDirectory(dir=Path.cwd()) as root_tmp):
            external = Path(external_tmp)
            root = Path(root_tmp)
            bundle = root / "tree"
            bundle.mkdir()
            predictions = bundle / "predictions.parquet"
            evaluation_frame().to_parquet(predictions, index=False)
            (bundle / "metadata.json").write_text(json.dumps({
                "pipeline": "daily", "family": "tree",
                "data_end": "2025-01-10 00:00:00+00:00",
                "horizons": 2, "folds": 1, "origins": 3,
            }), encoding="utf-8")
            out = external / "report"

            with self.assertRaisesRegex(ValueError, "outside provenance root"):
                main([
                    str(predictions), "--out", str(out),
                    "--provenance-root", str(root),
                ])
            self.assertFalse(out.exists())

    def test_cli_rejects_incomparable_bundle_before_creating_output(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            tree_dir, neural_dir = root / "tree", root / "neural"
            tree_dir.mkdir()
            neural_dir.mkdir()
            tree_predictions = tree_dir / "predictions.parquet"
            neural_predictions = neural_dir / "predictions.parquet"
            frame = evaluation_frame()
            tree_frame = frame[frame.model.isin(
                ["vol_21d", "xgb", "lgbm", "tree_blend"]
            )]
            neural_frame = frame[
                frame.model.isin(["lstm", "dlinear"])
                & frame.origin.ne(frame.origin.max())
            ]
            tree_frame.to_parquet(tree_predictions, index=False)
            neural_frame.to_parquet(neural_predictions, index=False)
            metadata = {
                "pipeline": "daily", "data_end": "2025-01-10 00:00:00+00:00",
                "horizons": 7, "folds": 1, "origins": 3,
            }
            (tree_dir / "metadata.json").write_text(json.dumps(metadata | {"family": "tree"}))
            (neural_dir / "metadata.json").write_text(json.dumps(metadata | {
                "family": "neural", "origins": 2,
            }))
            out = root / "report"

            with self.assertRaisesRegex(ValueError, "identical forecast key sets"):
                main([
                    str(tree_predictions), str(neural_predictions),
                    "--out", str(out), "--provenance-root", str(root),
                ])
            self.assertFalse(out.exists())
