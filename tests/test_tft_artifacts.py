import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from neural.tft_artifacts import (
    finalize_tft_run,
    reserve_tft_run,
    save_tft_core,
    verify_tft_manifest,
    write_raw_cv,
)
from tests.test_tft_evaluation import forecast_fixture


def raw_cv_frame():
    return pd.DataFrame({
        "unique_id": ["BTC"],
        "cutoff": [pd.Timestamp("2025-01-01 23:45")],
        "ds": [pd.Timestamp("2025-01-02 00:00")],
        "y": [4.6],
        "TFT-lo-80.0": [4.5],
        "TFT-median": [4.6],
        "TFT-hi-80.0": [4.7],
    })


def raw_test_frame():
    frame, _ = forecast_fixture(origins=2, horizons=2, assets=("BTC",))
    return frame.assign(split="test")


def calibrated_test_frame():
    return raw_test_frame().assign(model="tft_calibrated")


def complete_metadata():
    return {
        "run_id": "fixture", "pipeline": "hf15m", "family": "tft",
        "data_path": "data/ohlcv_15m.parquet", "data_start": "2021-01-01",
        "data_end": "2026-07-31", "assets": ["BTC"], "rows": 1000,
        "gap_stats": {}, "features": ["log_return"],
        "future_features": ["tod_sin"], "config": {"horizon": 96},
        "train_end": "2024-07-31", "validation_start": "2024-07-04",
        "validation_end": "2024-07-31", "calibration_start": "2024-08-01",
        "calibration_end": "2025-03-07", "test_start": "2025-03-08",
        "test_end": "2025-07-31",
        "package_versions": {"neuralforecast": "3.2.0"},
        "git_commit": "fixture", "elapsed_seconds": 1.0, "device": "cpu",
    }


class FakeNeuralForecast:
    def save(self, path, save_dataset=True, overwrite=False):
        target = Path(path)
        target.mkdir()
        (target / "weights.ckpt").write_bytes(b"weights")


def populate_run(root):
    out = reserve_tft_run("2026-07-31", "fixture", root)
    write_raw_cv(out, raw_cv_frame())
    save_tft_core(
        out, FakeNeuralForecast(), raw_test_frame(), calibrated_test_frame()
    )
    return out


class TftArtifactTests(unittest.TestCase):
    def test_complete_run_hashes_checkpoint_predictions_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = reserve_tft_run("2026-07-31", "fixture", Path(tmp))
            self.assertEqual(
                json.loads((out / "status.json").read_text())["state"], "incomplete"
            )
            write_raw_cv(out, raw_cv_frame())
            save_tft_core(
                out, FakeNeuralForecast(), raw_test_frame(), calibrated_test_frame()
            )
            finalize_tft_run(out, complete_metadata(), extra_paths=[])
            verify_tft_manifest(out)
            self.assertEqual(
                json.loads((out / "status.json").read_text())["state"], "complete"
            )
            self.assertTrue((out / "model" / "weights.ckpt").exists())
            self.assertTrue((out / "raw_cv.parquet").exists())
            self.assertTrue((out / "manifest.json").exists())

    def test_collision_and_unsafe_run_ids_fail_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            reserve_tft_run("2026-07-31", "fixture", Path(tmp))
            with self.assertRaises(FileExistsError):
                reserve_tft_run("2026-07-31", "fixture", Path(tmp))
            with self.assertRaisesRegex(ValueError, "run_id"):
                reserve_tft_run("2026-07-31", "../escape", Path(tmp))

    def test_raw_cv_is_persisted_only_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = reserve_tft_run("2026-07-31", "fixture", Path(tmp))
            write_raw_cv(out, raw_cv_frame())
            original = (out / "raw_cv.parquet").read_bytes()
            with self.assertRaises(FileExistsError):
                write_raw_cv(out, raw_cv_frame().assign(y=99.0))
            self.assertEqual((out / "raw_cv.parquet").read_bytes(), original)

    def test_core_artifacts_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = populate_run(Path(tmp))
            raw_path = out / "predictions_raw_test.parquet"
            original = raw_path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_tft_core(
                    out, FakeNeuralForecast(), raw_test_frame().assign(q50=99.0),
                    calibrated_test_frame(),
                )
            self.assertEqual(raw_path.read_bytes(), original)

    def test_finalize_rejects_missing_required_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = populate_run(Path(tmp))
            metadata = complete_metadata()
            del metadata["device"]
            with self.assertRaisesRegex(ValueError, "device"):
                finalize_tft_run(out, metadata)
            self.assertFalse((out / "metadata.json").exists())
            self.assertEqual(
                json.loads((out / "status.json").read_text())["state"], "incomplete"
            )

    def test_metadata_paths_are_portable_within_provenance_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            data_path = root / "data" / "ohlcv_15m.parquet"
            data_path.parent.mkdir()
            data_path.write_bytes(b"source")
            out = populate_run(root)
            metadata = complete_metadata() | {"data_path": str(data_path)}
            finalize_tft_run(out, metadata, provenance_root=root)
            saved = json.loads((out / "metadata.json").read_text())
            self.assertEqual(saved["data_path"], "data/ohlcv_15m.parquet")

    def test_metadata_paths_outside_provenance_root_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as external:
            root = Path(tmp).resolve()
            out = populate_run(root)
            metadata = complete_metadata() | {
                "data_path": str(Path(external).resolve() / "ohlcv.parquet")
            }
            with self.assertRaisesRegex(ValueError, "provenance root"):
                finalize_tft_run(out, metadata, provenance_root=root)
            self.assertFalse((out / "metadata.json").exists())

    def test_manifest_hashes_extra_evidence_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = populate_run(Path(tmp))
            table = out / "metrics.csv"
            graph = out / "forecast.png"
            table.write_text("metric,value\nmae,1\n", encoding="utf-8")
            graph.write_bytes(b"png")
            finalize_tft_run(out, complete_metadata(), extra_paths=[table, graph])
            paths = {
                record["path"]
                for record in json.loads((out / "manifest.json").read_text())["files"]
            }
            self.assertTrue({"metrics.csv", "forecast.png"}.issubset(paths))

    def test_manifest_hashes_nested_checkpoint_lifecycle_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = populate_run(Path(tmp))
            (out / "model" / "status.json").write_text(
                "checkpoint metadata", encoding="utf-8"
            )
            finalize_tft_run(out, complete_metadata())
            paths = {
                record["path"]
                for record in json.loads((out / "manifest.json").read_text())["files"]
            }
            self.assertIn("model/status.json", paths)

    def test_manifest_verification_rejects_changed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = populate_run(Path(tmp))
            finalize_tft_run(out, complete_metadata())
            (out / "raw_cv.parquet").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                verify_tft_manifest(out)

    def test_manifest_verification_rejects_missing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = populate_run(Path(tmp))
            finalize_tft_run(out, complete_metadata())
            (out / "predictions_raw_test.parquet").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                verify_tft_manifest(out)

    def test_manifest_verification_rejects_extra_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = populate_run(Path(tmp))
            finalize_tft_run(out, complete_metadata())
            (out / "unrecorded.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "extra"):
                verify_tft_manifest(out)


if __name__ == "__main__":
    unittest.main()
