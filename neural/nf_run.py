"""High-frequency (15-min) quantile forecasters via neuralforecast.

Four architectures, one harness, one command each:
    python -m neural.nf_run --model lstm
    python -m neural.nf_run --model gru
    python -m neural.nf_run --model nhits
    python -m neural.nf_run --model tft

All share MQLoss quantile output, the same nf_core reshape/grid/calibration, and
the same scoring. TFT additionally gets the known-future calendar covariates
(time-of-day, day-of-week) - its edge on intraday data. LSTM/GRU are near-identical
(GRU is a lighter LSTM); both included for completeness.

CPU works but a GPU is strongly preferred. TFT is the memory-heavy one - if it
OOMs on 8 GB, drop batch_size.
"""

import argparse

import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, GRU, NHITS, TFT
from neuralforecast.losses.pytorch import MQLoss

from neural.nf_core import (H_HF, INPUT_HF, FREQ, HIST_EXOG, FUTR_EXOG, QUANTILES,
                            cross_val_score, ensemble)

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
    return NHITS(**COMMON, n_pool_kernel_size=[4, 2, 1], n_freq_downsample=[8, 4, 1])


def tft():
    # TFT alone uses the known-future calendar covariates - intraday patterns.
    return TFT(**COMMON, futr_exog_list=FUTR_EXOG, hidden_size=128)


BUILDERS = {"lstm": lstm, "gru": gru, "nhits": nhits, "tft": tft}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(BUILDERS) + ["ensemble"], required=True)
    args = ap.parse_args()
    if args.model == "ensemble":       # averages saved cvs, no training
        ensemble()
    else:
        cross_val_score(NeuralForecast(models=[BUILDERS[args.model]()], freq=FREQ), args.model)
