"""User-operated next-day Temporal Fusion Transformer experiment runner."""

from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import TFT

from neural.tft_artifacts import (
    finalize_tft_run,
    reserve_tft_run,
    save_tft_core,
    verify_tft_manifest,
    write_raw_cv,
)
from neural.tft_data import (
    FUTR_EXOG,
    HIST_EXOG,
    HORIZON,
    INPUT_SIZE,
    REQUIRED_ASSETS,
    eligible_daily_origins,
    prepare_tft_data,
)
from neural.tft_evaluation import (
    calibrate_tft_intervals,
    make_tft_baselines,
    split_calibration_test,
    to_tft_forecasts,
)
from neural.tft_plots import render_tft_report


@dataclass(frozen=True)
class TFTExperimentConfig:
    run_id: str
    data_path: Path = Path("data/ohlcv_15m.parquet")
    output_root: Path = Path("artifacts/evaluation")
    assets: tuple[str, ...] = REQUIRED_ASSETS
    input_size: int = INPUT_SIZE
    horizon: int = HORIZON
    n_windows: int = 365
    calibration_origins: int = 219
    test_origins: int = 146
    validation_bars: int = 28 * HORIZON
    hidden_size: int = 128
    max_steps: int = 4000
    batch_size: int = 32
    accelerator: str = "auto"
    seed: int = 42


def build_tft(config):
    """Build the fixed q10/q50/q90 TFT without fitting it."""
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    loss = MQLoss(quantiles=[0.1, 0.5, 0.9])
    return TFT(
        h=config.horizon,
        input_size=config.input_size,
        hist_exog_list=list(HIST_EXOG),
        futr_exog_list=list(FUTR_EXOG),
        hidden_size=config.hidden_size,
        loss=loss,
        valid_loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
        scaler_type="robust",
        max_steps=config.max_steps,
        early_stop_patience_steps=5,
        val_check_steps=100,
        batch_size=config.batch_size,
        random_seed=config.seed,
        accelerator=config.accelerator,
        enable_progress_bar=True,
        logger=False,
    )


def read_data_end(data_path):
    """Read the minimum parquet projection needed to name the reserved run."""
    dates = pd.read_parquet(data_path, columns=["date"])["date"]
    dates = pd.to_datetime(dates, utc=True, errors="coerce")
    if dates.empty or dates.isna().any():
        raise ValueError("source parquet has no valid dates")
    return dates.max()


def _timestamp(value):
    return pd.Timestamp(value).isoformat()


def _git_commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _metadata_config(config):
    details = asdict(config)
    details.pop("output_root")
    return {
        key: (list(value) if isinstance(value, tuple) else value.as_posix() if isinstance(value, Path) else value)
        for key, value in details.items()
    }


def build_metadata(config, prepared, calibration, raw_test, elapsed_seconds):
    validation_end = pd.Timestamp(calibration.origin.min()) - pd.Timedelta("15min")
    validation_start = validation_end - pd.Timedelta(minutes=15 * (config.validation_bars - 1))
    return {
        "run_id": config.run_id,
        "pipeline": "hf15m",
        "family": "tft",
        "data_path": config.data_path.as_posix(),
        "data_start": _timestamp(prepared.model_frame.ds.min()),
        "data_end": _timestamp(prepared.data_end),
        "assets": list(config.assets),
        "rows": int(len(prepared.model_frame)),
        "gap_stats": {asset: asdict(stats) for asset, stats in prepared.gap_stats.items()},
        "features": list(HIST_EXOG),
        "future_features": list(FUTR_EXOG),
        "config": _metadata_config(config),
        "train_end": _timestamp(validation_end),
        "validation_start": _timestamp(validation_start),
        "validation_end": _timestamp(validation_end),
        "calibration_start": _timestamp(calibration.origin.min()),
        "calibration_end": _timestamp(calibration.origin.max()),
        "test_start": _timestamp(raw_test.origin.min()),
        "test_end": _timestamp(raw_test.origin.max()),
        "package_versions": {
            package: version(package)
            for package in ("neuralforecast", "numpy", "pandas", "torch")
        },
        "git_commit": _git_commit(),
        "elapsed_seconds": float(elapsed_seconds),
        "device": config.accelerator,
    }


def write_failure(output_dir, error):
    """Leave a reserved run explicitly incomplete with concise diagnostics."""
    output_dir = Path(output_dir)
    message = str(error).splitlines()[0][:500]
    (output_dir / "failure.json").write_text(
        json.dumps({"error_type": type(error).__name__, "message": message}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "status.json").write_text(
        json.dumps({"state": "incomplete"}) + "\n", encoding="utf-8"
    )


def run_tft(config, nf_class=NeuralForecast):
    """Run one explicitly requested daily-origin TFT cross-validation experiment."""
    data_end = read_data_end(config.data_path)
    output_dir = reserve_tft_run(data_end, config.run_id, config.output_root)
    started = time.monotonic()
    try:
        raw = pd.read_parquet(config.data_path)
        prepared = prepare_tft_data(raw, assets=config.assets)
        eligible_origins = eligible_daily_origins(prepared.context_frame, config.horizon)
        nf = nf_class(models=[build_tft(config)], freq="15min")
        raw_cv = nf.cross_validation(
            prepared.model_frame,
            n_windows=config.n_windows,
            step_size=config.horizon,
            val_size=config.validation_bars,
            refit=False,
        )
        write_raw_cv(output_dir, raw_cv)
        forecasts = to_tft_forecasts(raw_cv, prepared.context_frame, model="tft_raw")
        calibration, raw_test = split_calibration_test(
            forecasts,
            eligible_origins,
            config.calibration_origins,
            config.test_origins,
        )
        calibrated = calibrate_tft_intervals(calibration, raw_test, alpha=0.20)
        baselines = make_tft_baselines(raw_test)
        comparison = pd.concat([raw_test, calibrated, baselines], ignore_index=True)
        save_tft_core(output_dir, nf, raw_test, calibrated)
        report_paths = render_tft_report(calibration, comparison, output_dir)
        metadata = build_metadata(
            config,
            prepared,
            calibration,
            raw_test,
            elapsed_seconds=time.monotonic() - started,
        )
        finalize_tft_run(
            output_dir,
            metadata,
            extra_paths=report_paths,
            provenance_root=config.data_path.resolve().parents[1],
        )
        verify_tft_manifest(output_dir)
        return output_dir
    except Exception as error:
        write_failure(output_dir, error)
        raise
