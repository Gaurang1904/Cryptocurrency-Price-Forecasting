import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import call, patch

import numpy as np
import pandas as pd
import torch

REAL_READ_PARQUET = pd.read_parquet

from neural.tft_runner import TFTExperimentConfig, build_metadata, build_tft, run_tft
from tests.test_tft_evaluation import cv_fixture


def fixture_parquet_loader(path, columns=None):
    periods = 732 * 96
    dates = pd.date_range("2024-01-01", periods=periods, freq="15min", tz="UTC")
    frames = []
    for index, asset in enumerate(("BTC", "ETH"), start=1):
        close = index * 100 + np.arange(periods) * 0.0001
        frames.append(pd.DataFrame({
            "asset": asset, "date": dates, "open": close,
            "high": close * 1.001, "low": close * 0.999,
            "close": close, "volume": 10 + np.arange(periods) % 7,
        }))
    frame = pd.concat(frames, ignore_index=True)
    return frame if columns is None else frame.loc[:, columns]


class FakeNF:
    instances = []
    cv_origins = 365

    def __init__(self, models, freq):
        self.models = models
        self.freq = freq
        self.cross_validation_calls = []
        self.saved = False
        self.__class__.instances.append(self)

    def cross_validation(self, frame, **kwargs):
        self.cross_validation_calls.append((frame, kwargs))
        return cv_fixture(
            origins=self.cv_origins, horizons=96, assets=("BTC", "ETH"),
        )[0]

    def save(self, path, save_dataset=True, overwrite=False):
        self.saved = True
        target = Path(path)
        target.mkdir()
        (target / "fixture.ckpt").write_bytes(b"fake")


class TftRunnerTests(unittest.TestCase):
    def setUp(self):
        FakeNF.instances.clear()
        FakeNF.cv_origins = 365

    def test_model_construction_uses_fixed_daily_quantiles_and_exogenous_features(self):
        config = TFTExperimentConfig(run_id="fixture")
        loss, valid_loss = object(), object()
        with patch("neural.tft_runner.MQLoss", side_effect=[loss, valid_loss]) as mq_loss, patch(
            "neural.tft_runner.TFT"
        ) as tft:
            model = build_tft(config)

        self.assertIs(model, tft.return_value)
        self.assertEqual(
            mq_loss.call_args_list,
            [
                call(quantiles=[0.1, 0.5, 0.9]),
                call(quantiles=[0.1, 0.5, 0.9]),
            ],
        )
        self.assertEqual(tft.call_args.args, ())
        self.assertEqual(
            tft.call_args.kwargs,
            {
                "h": 96,
                "input_size": 672,
                "hist_exog_list": [
                    "log_return", "abs_return", "range_pct", "log_volume",
                    "log_volume_change", "rv_96", "rv_672", "momentum_96",
                    "momentum_672", "volume_z96", "missing_bar",
                ],
                "futr_exog_list": ["tod_sin", "tod_cos", "dow_sin", "dow_cos"],
                "hidden_size": 128,
                "loss": loss,
                "valid_loss": valid_loss,
                "scaler_type": "robust",
                "max_steps": 4000,
                "early_stop_patience_steps": 5,
                "val_check_steps": 100,
                "batch_size": 32,
                "random_seed": 42,
                "accelerator": "auto",
                "enable_progress_bar": True,
                "logger": False,
            },
        )

    def test_model_construction_is_seeded_before_tft_initialization(self):
        config = TFTExperimentConfig(run_id="fixture", assets=("BTC", "ETH"))
        first = build_tft(config)
        torch.manual_seed(999)
        second = build_tft(config)
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]))

    @patch("neural.tft_runner.render_tft_report", return_value=[])
    @patch("neural.tft_runner.pd.read_parquet", side_effect=fixture_parquet_loader)
    def test_runner_uses_one_fixed_daily_cv_without_real_training(self, _, __):
        with tempfile.TemporaryDirectory() as tmp:
            config = TFTExperimentConfig(
                run_id="fixture", output_root=Path(tmp), assets=("BTC", "ETH"),
                max_steps=1,
            )
            output_dir = run_tft(config, nf_class=FakeNF)
            self.assertEqual(len(FakeNF.instances[-1].cross_validation_calls), 1)
            frame, call = FakeNF.instances[-1].cross_validation_calls[-1]
            self.assertEqual(FakeNF.instances[-1].freq, "15min")
            self.assertEqual(call, {
                "n_windows": 365, "step_size": 96,
                "val_size": 28 * 96, "refit": False,
            })
            self.assertEqual(set(frame.unique_id), {"BTC", "ETH"})
            self.assertTrue(FakeNF.instances[-1].saved)
            self.assertEqual(
                json.loads((output_dir / "status.json").read_text())["state"],
                "complete",
            )
            self.assertTrue((output_dir / "raw_cv.parquet").is_file())
            self.assertTrue((output_dir / "failure.json").exists() is False)

    @patch("neural.tft_runner.pd.read_parquet", side_effect=fixture_parquet_loader)
    def test_runner_preserves_incomplete_reservation_and_failure_evidence(self, _):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "neural.tft_runner.prepare_tft_data", side_effect=ValueError("bad source")
        ):
            config = TFTExperimentConfig(
                run_id="failure", output_root=Path(tmp), assets=("BTC", "ETH"),
            )
            with self.assertRaisesRegex(ValueError, "bad source"):
                run_tft(config, nf_class=FakeNF)
            output_dir = next(Path(tmp).iterdir())
            self.assertEqual(
                json.loads((output_dir / "status.json").read_text())["state"],
                "incomplete",
            )
            failure = json.loads((output_dir / "failure.json").read_text())
            self.assertEqual(failure["error_type"], "ValueError")
            self.assertEqual(failure["message"], "bad source")

    @patch("neural.tft_runner.pd.read_parquet", side_effect=fixture_parquet_loader)
    def test_runner_persists_raw_cv_before_forecast_conversion_failure(self, _):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "neural.tft_runner.to_tft_forecasts",
            side_effect=RuntimeError("conversion failed"),
        ):
            config = TFTExperimentConfig(
                run_id="post-cv-failure", output_root=Path(tmp), assets=("BTC", "ETH"),
            )
            with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                run_tft(config, nf_class=FakeNF)

            output_dir = next(Path(tmp).iterdir())
            self.assertTrue((output_dir / "raw_cv.parquet").is_file())
            self.assertEqual(len(FakeNF.instances[-1].cross_validation_calls), 1)
            self.assertEqual(
                json.loads((output_dir / "status.json").read_text())["state"],
                "incomplete",
            )
            self.assertEqual(
                json.loads((output_dir / "failure.json").read_text()),
                {"error_type": "RuntimeError", "message": "conversion failed"},
            )

    @patch("neural.tft_runner.render_tft_report", return_value=[])
    @patch("neural.tft_runner.pd.read_parquet", side_effect=fixture_parquet_loader)
    def test_runner_splits_only_the_eligible_daily_origins(self, _, __):
        FakeNF.cv_origins = 366
        eligible = pd.date_range(
            "2025-01-02", periods=366, freq="D", tz="UTC",
        )
        synthetic_origin = pd.Timestamp("2025-01-02", tz="UTC")
        with tempfile.TemporaryDirectory() as tmp, patch(
            "neural.tft_runner.eligible_daily_origins", return_value=eligible,
        ) as origins, patch(
            "neural.tft_runner.split_calibration_test",
            wraps=__import__("neural.tft_runner", fromlist=["split_calibration_test"]).split_calibration_test,
        ) as split:
            config = TFTExperimentConfig(
                run_id="eligible-only", output_root=Path(tmp), assets=("BTC", "ETH"),
                max_steps=1,
            )
            output_dir = run_tft(config, nf_class=FakeNF)
            raw_test_origins = set(pd.to_datetime(
                REAL_READ_PARQUET(output_dir / "predictions_raw_test.parquet")["origin"], utc=True,
            ))

        origins.assert_called_once()
        self.assertEqual(origins.call_args.args[1], 96)
        split.assert_called_once()
        pd.testing.assert_index_equal(split.call_args.args[1], eligible)
        self.assertEqual(split.call_args.args[2:], (219, 146))
        self.assertNotIn(synthetic_origin, raw_test_origins)

    def test_cli_forwards_operational_tft_options(self):
        from neural.nf_run import main

        with patch("neural.nf_run.run_tft") as run:
            main([
                "--model", "tft", "--run-id", "manual-1",
                "--output-root", "runs", "--accelerator", "gpu",
                "--batch-size", "16",
            ])
        config = run.call_args.args[0]
        self.assertEqual(config.run_id, "manual-1")
        self.assertEqual(config.output_root, Path("runs"))
        self.assertEqual(config.accelerator, "gpu")
        self.assertEqual(config.batch_size, 16)

    @patch("neural.tft_runner._git_commit", return_value="fixture")
    @patch("neural.tft_runner.version", return_value="fixture")
    def test_metadata_train_and_validation_boundaries_are_exact(self, _, __):
        calibration_origin = pd.Timestamp("2026-01-31 00:00", tz="UTC")
        metadata = build_metadata(
            TFTExperimentConfig(run_id="fixture"),
            SimpleNamespace(
                model_frame=pd.DataFrame({"ds": [pd.Timestamp("2024-01-01")]}),
                data_end=pd.Timestamp("2026-02-01", tz="UTC"), gap_stats={},
            ),
            pd.DataFrame({"origin": [calibration_origin]}),
            pd.DataFrame({"origin": [calibration_origin + pd.Timedelta(days=219)]}),
            elapsed_seconds=1.0,
        )
        validation_end = calibration_origin - pd.Timedelta("15min")
        validation_start = validation_end - pd.Timedelta(minutes=15 * 2687)
        self.assertEqual(pd.Timestamp(metadata["validation_end"]), validation_end)
        self.assertEqual(pd.Timestamp(metadata["validation_start"]), validation_start)
        self.assertEqual(
            pd.Timestamp(metadata["train_end"]), validation_start - pd.Timedelta("15min"),
        )

    def test_cli_rejects_tft_options_for_other_models(self):
        from neural.nf_run import main

        with self.assertRaisesRegex(SystemExit, "2"):
            main(["--model", "lstm", "--run-id", "manual-1"])
