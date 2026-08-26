import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import patch

from crypto.evaluation import (
    default_run_dir,
    metric_tables,
    save_predictions,
    validate_predictions,
)


def valid_frame():
    return pd.DataFrame({
        "model": ["xgb"], "asset": ["BTC"],
        "origin": [pd.Timestamp("2025-01-01", tz="UTC")],
        "fold": [pd.Timestamp("2025-01-01", tz="UTC")],
        "h": [1], "y": [101.0], "last": [100.0],
        "sigma": [0.02], "rv": [0.018], "regime_driver": [0.02],
        "q10": [95.0], "q50": [100.0], "q90": [105.0],
    })


def comparable_frame(model="xgb", horizons=(1, 2)):
    rows = []
    for i, origin in enumerate(pd.date_range("2025-01-01", periods=2, tz="UTC")):
        for h in horizons:
            rows.append(valid_frame().assign(
                model=model, asset="BTC" if i == 0 else "ETH", origin=origin,
                fold=pd.Timestamp("2025-01-01", tz="UTC"), h=h,
                y=100.0 + i + h, last=100.0 + i,
                sigma=0.02, rv=0.01 + h / 1000,
                regime_driver=0.02 + i / 100,
                q10=95.0 + i, q50=100.0 + i, q90=105.0 + i,
            ))
    return pd.concat(rows, ignore_index=True)


class PredictionValidationTests(unittest.TestCase):
    def test_run_directory_contains_pipeline_and_data_cutoff(self):
        got = default_run_dir(
            "tree", pd.Timestamp("2026-07-23", tz="UTC"), "baseline", Path("out")
        )
        self.assertEqual(got, Path("out/daily-tree-20260723-baseline"))

    def test_reserve_run_directory_is_atomic_and_rejects_collisions(self):
        from crypto import evaluation

        self.assertTrue(hasattr(evaluation, "reserve_run_dir"))
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            cutoff = pd.Timestamp("2026-07-23", tz="UTC")
            first = evaluation.reserve_run_dir("tree", cutoff, root=root)
            second = evaluation.reserve_run_dir("tree", cutoff, root=root)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertEqual(list(first.iterdir()), [])
            taken = evaluation.reserve_run_dir(
                "tree", cutoff, run_id="taken", root=root
            )
            self.assertTrue(taken.is_dir())
            with self.assertRaises(FileExistsError):
                evaluation.reserve_run_dir(
                    "tree", cutoff, run_id="taken", root=root
                )

    def test_valid_predictions_are_accepted(self):
        validate_predictions(valid_frame())

    def test_crossed_quantiles_are_rejected(self):
        frame = valid_frame().assign(q10=106.0)
        with self.assertRaisesRegex(ValueError, "crossed quantiles"):
            validate_predictions(frame)

    def test_duplicate_forecast_keys_are_rejected(self):
        frame = pd.concat([valid_frame(), valid_frame()], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate forecast keys"):
            validate_predictions(frame)

    def test_missing_regime_driver_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "regime_driver"):
            validate_predictions(valid_frame().drop(columns="regime_driver"))

    def test_non_numeric_regime_driver_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "numeric regime_driver"):
            validate_predictions(valid_frame().assign(regime_driver="high"))

    def test_metric_tables_report_coverage_by_horizon_and_asset(self):
        frame = pd.concat([
            valid_frame(),
            valid_frame().assign(
                asset="ETH", y=110.0,
                origin=pd.Timestamp("2025-01-02", tz="UTC"),
            ),
        ], ignore_index=True)
        tables = metric_tables(frame)
        self.assertEqual(tables["overall"].loc["xgb", "coverage"], 50.0)
        self.assertEqual(tables["by_horizon"].loc[("xgb", 1), "coverage"], 50.0)
        self.assertEqual(tables["by_fold"].loc[(
            "xgb", pd.Timestamp("2025-01-01", tz="UTC")
        ), "coverage"], 50.0)
        self.assertEqual(tables["by_asset"].loc[("xgb", "BTC"), "coverage"], 100.0)
        self.assertEqual(tables["by_asset"].loc[("xgb", "ETH"), "coverage"], 0.0)

    def test_metric_tables_label_regimes_from_fold_drivers_not_outcomes(self):
        frame = pd.concat([
            valid_frame().assign(asset="BTC", origin=pd.Timestamp("2025-01-01", tz="UTC"), regime_driver=0.01),
            valid_frame().assign(asset="ETH", origin=pd.Timestamp("2025-01-02", tz="UTC"), regime_driver=0.02, y=110.0),
            valid_frame().assign(asset="SOL", origin=pd.Timestamp("2025-01-03", tz="UTC"), regime_driver=0.03),
        ], ignore_index=True)
        tables = metric_tables(frame)
        labels = tables["by_regime"].index.get_level_values("regime").tolist()
        self.assertEqual(labels, ["low", "medium", "high"])
        self.assertEqual(tables["by_regime"].loc[("xgb", pd.Timestamp("2025-01-01", tz="UTC"), "medium"), "coverage"], 0.0)

        changed_outcomes = frame.assign(y=[110.0, 101.0, 110.0])
        changed = metric_tables(changed_outcomes)
        self.assertEqual(changed["by_regime"].index.tolist(), tables["by_regime"].index.tolist())

    def test_neural_backtest_keeps_test_provenance_aligned_with_sigma_and_rv(self):
        from neural import core

        columns = {"asset": ["BTC", "ETH"], "date": pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True),
                   "close": [100.0, 200.0], "vol_21d": [0.21, 0.29]}
        for h in range(1, core.H + 1):
            columns[f"y{h}"] = [0.01, 0.02]
            columns[f"rv{h}"] = [0.011 * h, 0.012 * h]
        test = pd.DataFrame(columns)
        train = pd.concat([test, test], ignore_index=True)
        fold = pd.Timestamp("2026-01-01", tz="UTC")
        cal_prediction = np.log(np.full((2, core.H), 0.1))
        test_prediction = np.log(np.array([np.arange(0.21, 0.21 + core.H * 0.01, 0.01), np.arange(0.29, 0.29 + core.H * 0.001, 0.001)]))

        with (tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp,
              patch.object(core, "build", return_value=(test, [])),
              patch.object(core.pd, "read_parquet", return_value=test),
              patch.object(core, "channel_windows", return_value=(np.zeros((2, 1, 1)), test[["asset", "date"]])),
              patch.object(core, "run_folds", return_value=[(train, test, fold)]),
              patch.object(core, "split_calibration", return_value=(train, test)),
              patch.object(core, "_lookup", side_effect=lambda _win, _idx, rows: (np.zeros((len(rows), 1, 1)), np.ones(len(rows), dtype=bool))),
              patch.object(core, "train_net", side_effect=lambda model, *_args: model),
              patch.object(core, "_predict", side_effect=[cal_prediction, test_prediction]),
              patch.object(core, "calibrate", return_value={0.1: 1.0, 0.5: 1.0, 0.9: 1.0}),
              patch.object(core, "save_predictions"),
              patch.object(core, "bands", side_effect=lambda _z, last, _sigma, _h: {0.1: last * 0.9, 0.5: last, 0.9: last * 1.1})):
            res, _ = core.backtest(
                {"test-net": lambda _channels: object()},
                output_root=Path(tmp),
            )

        h1 = res[res.h == 1]
        self.assertListEqual(h1.origin.tolist(), test.date.tolist())
        self.assertListEqual(h1.fold.tolist(), [fold, fold])
        self.assertListEqual(h1.sigma.tolist(), [0.21, 0.29])
        self.assertListEqual(h1.rv.tolist(), [0.011, 0.012])
        self.assertListEqual(h1.regime_driver.tolist(), [0.21, 0.29])

    def test_tree_backtest_keeps_test_provenance_aligned_with_sigma_and_rv(self):
        from tree import run

        columns = {"asset": ["BTC", "ETH"], "date": pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True),
                   "close": [100.0, 200.0], "vol_21d": [0.21, 0.29]}
        for h in range(1, run.H + 1):
            columns[f"y{h}"] = [0.01, 0.02]
            columns[f"rv{h}"] = [0.011 * h, 0.012 * h]
        test = pd.DataFrame(columns)
        train = pd.concat([test, test], ignore_index=True)
        fold = pd.Timestamp("2026-01-01", tz="UTC")

        with (tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp,
              patch.object(run, "build", return_value=(test, [])),
              patch.object(run.pd, "read_parquet", return_value=test),
              patch.object(run, "run_folds", return_value=[(train, test, fold)]),
              patch.object(run, "split_calibration", return_value=(train, test)),
              patch.object(run, "model_sigma", return_value=(np.full(2, 0.1), np.array([0.41, 0.49]))),
              patch.object(run, "calibrate", return_value={0.1: 1.0, 0.5: 1.0, 0.9: 1.0}),
              patch.object(run, "save_predictions"),
              patch.object(run, "bands", side_effect=lambda _z, last, _sigma, _h: {0.1: last * 0.9, 0.5: last, 0.9: last * 1.1})):
            res, _ = run.backtest(
                {"test-tree": lambda *_args: object()},
                output_root=Path(tmp),
            )

        h1 = res[(res.model == "test-tree") & (res.h == 1)]
        self.assertListEqual(h1.origin.tolist(), test.date.tolist())
        self.assertListEqual(h1.fold.tolist(), [fold, fold])
        self.assertListEqual(h1.sigma.tolist(), [0.41, 0.49])
        self.assertListEqual(h1.rv.tolist(), [0.011, 0.012])
        self.assertListEqual(h1.regime_driver.tolist(), [0.21, 0.29])

    def test_runner_clis_accept_run_id_and_output_root(self):
        from neural import run as neural_run
        from tree import run as tree_run

        for module in (tree_run, neural_run):
            self.assertTrue(hasattr(module, "parse_args"))
            args = module.parse_args([
                "--run-id", "review-1", "--output-root", "custom-output",
            ])
            self.assertEqual(args.run_id, "review-1")
            self.assertEqual(args.output_root, Path("custom-output"))

    def test_runners_reject_output_collisions_before_expensive_work(self):
        from neural import core
        from tree import run

        for module in (run, core):
            self.assertIn("run_id", inspect.signature(module.backtest).parameters)
            self.assertIn("output_root", inspect.signature(module.backtest).parameters)

        cutoff = pd.Timestamp("2026-07-23", tz="UTC")
        feat = pd.DataFrame({"date": [cutoff]})
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            for tag, module, expensive in (
                ("tree", run, "run_folds"),
                ("neural", core, "channel_windows"),
            ):
                with self.subTest(tag=tag):
                    default_run_dir(tag, cutoff, "taken", root).mkdir()
                    with (patch.object(module, "build", return_value=(feat, [])),
                          patch.object(module.pd, "read_parquet", return_value=feat),
                          patch.object(module, expensive,
                                       side_effect=AssertionError("expensive work started"))):
                        with self.assertRaises(FileExistsError):
                            module.backtest({}, run_id="taken", output_root=root)

    def test_runners_leave_atomic_reservation_before_expensive_work(self):
        from neural import core
        from tree import run

        cutoff = pd.Timestamp("2026-07-23", tz="UTC")
        feat = pd.DataFrame({"date": [cutoff]})
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            for tag, module, expensive in (
                ("tree", run, "run_folds"),
                ("neural", core, "channel_windows"),
            ):
                with self.subTest(tag=tag):
                    with (patch.object(module, "build", return_value=(feat, [])),
                          patch.object(module.pd, "read_parquet", return_value=feat),
                          patch.object(
                              module, expensive,
                              side_effect=RuntimeError("expensive work started"),
                          )):
                        with self.assertRaisesRegex(
                            RuntimeError, "expensive work started"
                        ):
                            module.backtest(
                                {}, run_id="orphan", output_root=root
                            )
                    reserved = default_run_dir(
                        tag, cutoff, "orphan", root
                    )
                    self.assertTrue(reserved.is_dir())
                    self.assertEqual(list(reserved.iterdir()), [])

    def test_save_writes_predictions_and_metadata_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = save_predictions(valid_frame(), Path(tmp) / "run-1", {"pipeline": "daily"})
            self.assertTrue((out / "predictions.parquet").exists())
            self.assertTrue((out / "metadata.json").exists())
            with self.assertRaises(FileExistsError):
                save_predictions(valid_frame(), Path(tmp) / "run-1", {"pipeline": "daily"})
            existing_empty = Path(tmp) / "existing-empty"
            existing_empty.mkdir()
            with self.assertRaises(FileExistsError):
                save_predictions(
                    valid_frame(), existing_empty, {"pipeline": "daily"}
                )

    def test_save_consumes_reserved_empty_directory_once(self):
        from crypto.evaluation import reserve_run_dir

        with tempfile.TemporaryDirectory() as tmp:
            reserved = reserve_run_dir(
                "tree", pd.Timestamp("2026-07-23", tz="UTC"),
                run_id="reserved", root=Path(tmp),
            )
            out = save_predictions(
                valid_frame(), reserved, {"pipeline": "daily"}, reserved=True
            )
            self.assertEqual(out, reserved)
            self.assertEqual(
                {path.name for path in reserved.iterdir()},
                {"predictions.parquet", "metadata.json"},
            )
            with self.assertRaises(FileExistsError):
                save_predictions(
                    valid_frame(), reserved, {"pipeline": "daily"}, reserved=True
                )

    def test_save_refuses_populated_reserved_directory(self):
        from crypto.evaluation import reserve_run_dir

        with tempfile.TemporaryDirectory() as tmp:
            reserved = reserve_run_dir(
                "tree", pd.Timestamp("2026-07-23", tz="UTC"),
                run_id="reserved", root=Path(tmp),
            )
            (reserved / "orphan.txt").write_text("occupied", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                save_predictions(
                    valid_frame(), reserved, {"pipeline": "daily"}, reserved=True
                )

    def test_run_ids_are_single_safe_path_components(self):
        from crypto.evaluation import reserve_run_dir

        invalid = ["", ".", "..", ".hidden", "../escape", "a/b", "a\\b", "trail."]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run_id in invalid:
                with self.subTest(run_id=run_id):
                    with self.assertRaisesRegex(ValueError, "run_id"):
                        reserve_run_dir(
                            "tree", pd.Timestamp("2026-07-23", tz="UTC"),
                            run_id=run_id, root=root,
                        )

    def test_comparable_bundles_require_identical_model_keys(self):
        from crypto.evaluation import validate_comparable_predictions

        left = comparable_frame("xgb")
        right = comparable_frame("lstm").iloc[:-2]
        metadata = [
            {"horizons": 2, "folds": 1, "origins": 2},
            {"horizons": 2, "folds": 1, "origins": 1},
        ]
        with self.assertRaisesRegex(ValueError, "identical forecast key sets"):
            validate_comparable_predictions([left, right], metadata)

    def test_comparable_bundles_require_shared_truth_and_context(self):
        from crypto.evaluation import validate_comparable_predictions

        left = comparable_frame("xgb")
        right = comparable_frame("lstm")
        right.loc[0, "y"] += 1
        metadata = [
            {"horizons": 2, "folds": 1, "origins": 2},
            {"horizons": 2, "folds": 1, "origins": 2},
        ]
        with self.assertRaisesRegex(ValueError, "shared y"):
            validate_comparable_predictions([left, right], metadata)

    def test_comparable_bundles_require_complete_horizons(self):
        from crypto.evaluation import validate_comparable_predictions

        frames = [comparable_frame("xgb", horizons=(1,)), comparable_frame("lstm", horizons=(1,))]
        metadata = [
            {"horizons": 2, "folds": 1, "origins": 2},
            {"horizons": 2, "folds": 1, "origins": 2},
        ]
        with self.assertRaisesRegex(ValueError, "complete horizons"):
            validate_comparable_predictions(frames, metadata)

    def test_comparable_bundles_require_metadata_counts_to_match_rows(self):
        from crypto.evaluation import validate_comparable_predictions

        frames = [comparable_frame("xgb"), comparable_frame("lstm")]
        metadata = [
            {"horizons": 2, "folds": 1, "origins": 3},
            {"horizons": 2, "folds": 1, "origins": 2},
        ]
        with self.assertRaisesRegex(ValueError, "metadata origins"):
            validate_comparable_predictions(frames, metadata)

    def test_neural_model_construction_is_cold_start_reproducible(self):
        import torch
        from neural import core

        build = lambda channels: torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(channels * 2, core.H)
        )
        first = core.construct_model(build, channels=2)
        torch.manual_seed(999)
        second = core.construct_model(build, channels=2)

        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]))
        sample = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        np.testing.assert_array_equal(core._predict(first, sample), core._predict(second, sample))
