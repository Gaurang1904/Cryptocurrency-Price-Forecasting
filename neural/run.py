"""Backtest the neural family: DLinear vs LSTM. python -m neural.run

Both sequence models scored on the same origins as tree/ and linear/, so pinball
and coverage compare across all three families. CPU-only, a few minutes.
"""

from neural.core import backtest, report_and_log
from neural.dlinear import build as dlinear
from neural.lstm import build as lstm

BUILDERS = {"dlinear": dlinear, "lstm": lstm}

if __name__ == "__main__":
    res, vol_err = backtest(BUILDERS)
    report_and_log(res, vol_err, "neural")
