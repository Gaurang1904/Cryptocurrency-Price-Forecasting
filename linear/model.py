"""HAR-RV, Ridge, and GARCH(1,1) volatility models. All per-series."""

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.linear_model import Ridge

from crypto.model import clip_sigma
from linear.adapter import HAR_COLS, ridge_pipeline


def fit_har(fit, h):
    """HAR-RV: log realised vol on its own 1d/5d/22d averages. The vol benchmark."""
    return Ridge(alpha=0.0).fit(fit[HAR_COLS], np.log(fit[f"rv{h}"]))


def fit_ridge(fit, cols, h):
    """Linear control. If a tree cannot beat this, the nonlinearity is noise."""
    return ridge_pipeline().fit(fit[cols].fillna(0), np.log(fit[f"rv{h}"]))


def _garch(returns):
    return arch_model(returns * 100, vol="Garch", p=1, q=1, mean="Zero")


def fit_garch(returns):
    return _garch(returns).fit(disp="off")


def garch_sigma(returns, params, h):
    """Conditional h-day sigma at EVERY point, using parameters fit elsewhere.

    Refilters the full series with fixed parameters, so each origin gets its own
    conditional variance. Forecasting once per fold collapses GARCH to a constant.
    """
    var = _garch(returns).fix(params).forecast(horizon=h, reindex=True).variance
    return pd.Series(clip_sigma(np.sqrt(var.mean(axis=1)).to_numpy() / 100), index=returns.index)
