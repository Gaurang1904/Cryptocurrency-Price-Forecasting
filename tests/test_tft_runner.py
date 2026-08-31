import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from neural.tft_runner import TFTExperimentConfig, build_tft, run_tft
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

    def __init__(self, models, freq):
        self.models = models
        self.freq = freq
        self.cross_validation_calls = []
        self.saved = False
        self.__class__.instances.append(self)

    def cross_validation(self, frame, **kwargs):
        self.cross_validation_calls.append((frame, kwargs))
        return cv_fixture(origins=365, horizons=96, assets=("BTC", "ETH"))[0]

    def save(self, path, save_dataset=True, overwrite=False):
        self.saved = True
        target = Path(path)
        target.mkdir()
        (target / "fixture.ckpt").write_bytes(b"fake")


class TftRunnerTests(unittest.TestCase):
    def setUp(self):
        FakeNF.instances.clear()

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

    def test_cli_rejects_tft_options_for_other_models(self):
        from neural.nf_run import main

        with self.assertRaisesRegex(SystemExit, "2"):
            main(["--model", "lstm", "--run-id", "manual-1"])
