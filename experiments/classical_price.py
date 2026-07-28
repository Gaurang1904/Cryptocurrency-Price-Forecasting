"""Classical price-level forecasters vs the naive baselines. python -m experiments.classical_price

ARIMA, SARIMA, Linear Regression all predict the PRICE (a level), not volatility.
Scored with report() - MASE/MAPE/DirAcc - against flat/drift/seasonal on the same
origins. This is the arena where flat has been unbeaten; the test is whether any
classical model changes that.

Runtime: fits ARIMA/SARIMA at every origin, so it is slow (minutes). History is
capped at WINDOW bars for speed and convergence.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from crypto.backtest import H, log, naive, report, walk_forward
from crypto.data import OHLCV_OUT

warnings.filterwarnings("ignore")  # statsmodels floods convergence warnings
WINDOW = 400  # bars of history each classical model sees (expanding is slower, not better)


def _arima(hist, order, seasonal=(0, 0, 0, 0)):
    try:
        f = ARIMA(hist[-WINDOW:], order=order, seasonal_order=seasonal).fit()
        return np.asarray(f.forecast(H))
    except Exception:
        return np.full(H, hist[-1])  # convergence failure -> fall back to flat


def _linreg(hist, n=30):
    """OLS line through the last n closes, extrapolated H steps. Literally
    linear regression on price."""
    y = hist[-n:]
    slope, intercept = np.polyfit(np.arange(n), y, 1)
    return intercept + slope * (n + np.arange(H))


def predict(hist):
    out = naive(hist)  # flat, drift, seasonal - the bar to beat
    out["linreg"] = _linreg(hist)
    out["arima"] = _arima(hist, (1, 1, 1))
    out["sarima"] = _arima(hist, (1, 1, 1), (1, 0, 1, 7))  # weekly seasonality
    return out


if __name__ == "__main__":
    df = pd.read_parquet(OHLCV_OUT)
    print(f"fitting classical models at every origin over {df.asset.nunique()} coins "
          f"(slow)...\n")
    res = walk_forward(df, predict=predict)
    summary, per_h = report(res)

    print(f"{len(res) // (len(res.model.unique()) * H)} origins\n")
    print("PRICE FORECAST (lower MAE/MAPE/MASE better, DirAcc vs 50%)")
    print(summary.round(2).to_string())
    print("\nMAPE % by horizon day")
    print(per_h.round(2).to_string())
    log("classical_price", summary, origins=len(res) // (len(res.model.unique()) * H),
        assets=df.asset.nunique())
