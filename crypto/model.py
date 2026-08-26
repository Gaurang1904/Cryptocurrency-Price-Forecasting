"""Shared sigma -> price-interval calibration. Family-agnostic.

Every model family predicts one thing: daily volatility. This module turns a
sigma estimate into a price interval, and it does so identically no matter
which family produced the sigma. That is what lets the families be compared
on one scale. The family-specific fitters live in tree/, linear/, neural/.
"""

import numpy as np
from scipy.stats import norm

from crypto.backtest import purge_forward_labels

QUANTILES = [0.1, 0.5, 0.9]
CAL_FRAC = 0.2  # tail of the training window reserved for calibration
SIGMA_MIN, SIGMA_MAX = 1e-4, 0.30  # daily vol bounds; 30%/day is already extreme


def split_calibration(train, frac=CAL_FRAC):
    """Split fit/calibration and purge labels that reach past the cut."""
    cut = train.date.quantile(1 - frac)
    fit = purge_forward_labels(train[train.date < cut], cut)
    return fit, train[train.date >= cut]


def clip_sigma(s):
    """exp() of a linear prediction on outlier features can reach 1e120."""
    return np.clip(np.nan_to_num(s, nan=SIGMA_MIN), SIGMA_MIN, SIGMA_MAX)


def calibrate(y, sigma, h, quantiles=QUANTILES):
    """Empirical quantiles of h-day returns standardised by THIS sigma estimate."""
    s = np.asarray(y) / (np.asarray(sigma) * np.sqrt(h))
    return {q: float(np.nanquantile(s, q)) for q in quantiles}


def bands(z, last, sigma, h):
    return {q: last * np.exp(zq * sigma * np.sqrt(h)) for q, zq in z.items()}


def rw_band(last, sigma, h, quantiles=QUANTILES):
    """Random-walk interval assuming normal returns. The baseline to beat."""
    return {q: last * np.exp(norm.ppf(q) * sigma * np.sqrt(h)) for q in quantiles}


def fit_all(train, cols, fit_one, quantiles=QUANTILES):
    """Fit every horizon plus its calibration table, given a family's fitter.

    fit_one(fit_df, cols, h) -> object with .predict, and predict_sigma below
    handles turning that into sigma. Returns (models, z, n_fit, n_cal).
    """
    from crypto.features import H

    fit, cal = split_calibration(train)
    models, z = {}, {}
    for h in range(1, H + 1):
        m = fit_one(fit, cols, h)
        sigma = clip_sigma(np.exp(m.predict(cal[cols])))
        models[h] = m
        z[h] = calibrate(cal[f"y{h}"].to_numpy(), sigma, h, quantiles)
        assert list(z[h].values()) == sorted(z[h].values()), f"h={h}: quantiles crossed"
    return models, z, len(fit), len(cal)
