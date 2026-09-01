"""High-frequency (15-min) quantile forecasters via neuralforecast.

Four architectures, one harness, one command each:
    python -m neural.nf_run --model lstm
    python -m neural.nf_run --model gru
    python -m neural.nf_run --model nhits
    python -m neural.nf_run --model tft --run-id <safe-run-id> --output-root artifacts/evaluation --accelerator gpu --batch-size 16

All share MQLoss quantile output, the same nf_core reshape/grid/calibration, and
the same scoring. TFT additionally gets the known-future calendar covariates
(time-of-day, day-of-week) - its edge on intraday data. LSTM/GRU are near-identical
(GRU is a lighter LSTM); both included for completeness.

CPU works but a GPU is strongly preferred. TFT is the memory-heavy one - if it
OOMs on 8 GB, drop batch_size.
"""

import argparse
from pathlib import Path

import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, GRU, NHITS
from neuralforecast.losses.pytorch import MQLoss

from neural.nf_core import (H_HF, INPUT_HF, FREQ, HIST_EXOG, FUTR_EXOG, QUANTILES,
                            cross_val_score, ensemble)
from neural.tft_runner import TFTExperimentConfig, run_tft

MAX_STEPS = 4000  # was 1000 - loss was still falling, models were undertrained
torch.set_float32_matmul_precision("high")  # Tensor Cores on the 4060

COMMON = dict(h=H_HF, input_size=INPUT_HF, loss=MQLoss(quantiles=QUANTILES),
              hist_exog_list=HIST_EXOG, scaler_type="robust", max_steps=MAX_STEPS,
              val_check_steps=100, random_seed=42, enable_progress_bar=True, logger=False)


def lstm():
    return LSTM(**COMMON, encoder_hidden_size=128)   # was 64 - more capacity


def gru():
    return GRU(**COMMON, encoder_hidden_size=128)


def nhits():
    return NHITS(
        **COMMON,
        n_pool_kernel_size=[4, 2, 1],
        n_freq_downsample=[8, 4, 1],
        batch_size=4,
        valid_batch_size=4,
        windows_batch_size=128,
        inference_windows_batch_size=128,
    )


BUILDERS = {"lstm": lstm, "gru": gru, "nhits": nhits}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*BUILDERS, "tft", "ensemble"], required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root")
    parser.add_argument("--accelerator")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args(argv)
    tft_options = (args.run_id, args.output_root, args.accelerator, args.batch_size)
    if args.model != "tft" and any(option is not None for option in tft_options):
        parser.error("TFT operational options require --model tft")
    if args.model == "tft":
        if args.run_id is None:
            parser.error("--run-id is required for --model tft")
        options = {"run_id": args.run_id}
        if args.output_root is not None:
            options["output_root"] = Path(args.output_root)
        if args.accelerator is not None:
            options["accelerator"] = args.accelerator
        if args.batch_size is not None:
            options["batch_size"] = args.batch_size
        return run_tft(TFTExperimentConfig(**options))
    if args.model == "ensemble":
        return ensemble()
    return cross_val_score(
        NeuralForecast(models=[BUILDERS[args.model]()], freq=FREQ), args.model
    )


if __name__ == "__main__":
    main()
