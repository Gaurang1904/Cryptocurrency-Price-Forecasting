"""Shared neuralforecast plumbing for the high-frequency (15-min) experiments.

Different from neural/core.py: neuralforecast wants its own long format
(unique_id, ds, y) on a GAPLESS regular grid, and it does windowing, scaling,
and quantile output internally. So this module only reshapes + fills the grid,
runs cross_validation, and maps the quantile output back into the repo's
score() so pinball/coverage stay comparable across models.

Horizon here is 72 hours (3 days). These numbers are NOT comparable to the
daily models - intraday forecasts live on their own scale.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from crypto.backtest import score, coverage_by_h, log, point_metrics
from crypto.data import OHLCV_15M_OUT

# High-frequency default: 15-min bars, 2-hour (8-bar) horizon, 24-hour lookback.
FREQ = "15min"
PATH = OHLCV_15M_OUT
H_HF = 8           # 2 hours ahead
INPUT_HF = 192     # 48-hour lookback (more context per forecast)
QUANTILES = [0.1, 0.5, 0.9]
HIST_EXOG = ["lret", "aret", "rng", "lvol"]              # past-only, all models
FUTR_EXOG = ["tod_sin", "tod_cos", "dow_sin", "dow_cos"]  # known-future calendar, TFT

# Many non-overlapping origins so conformal calibration has enough windows.
N_WINDOWS = 40
STEP = H_HF * 6    # 48 bars (12h) between origins, non-overlapping (horizon is 8)


def to_nf(path=PATH, freq=FREQ):
    """Reshape OHLCV to neuralforecast format on a gapless grid.

    y = log close. Exogenous features are recomputed AFTER gap-filling so a
    forward-filled bar does not create a fake return spike. Calendar covariates
    (time-of-day, day-of-week) are known-future - deterministic from ds - so TFT
    can use them to exploit intraday patterns. ds is tz-naive UTC.
    """
    df = pd.read_parquet(path)
    bars_per_day = int(pd.Timedelta("1D") / pd.Timedelta(freq))
    out = []
    for asset, g in df.groupby("asset", sort=False):
        g = g.set_index("date").sort_index()
        grid = pd.date_range(g.index.min(), g.index.max(), freq=freq, tz="UTC")
        g = g.reindex(grid).ffill()  # fill exchange-downtime holes
        lr = np.log(g.close).diff()
        tod = (grid.hour * 60 + grid.minute) / (24 * 60)  # time-of-day fraction
        dow = grid.dayofweek / 7
        out.append(pd.DataFrame({
            "unique_id": asset,
            "ds": grid.tz_convert("UTC").tz_localize(None),
            "y": np.log(g.close).to_numpy(),
            "lret": lr.to_numpy(),
            "aret": lr.abs().to_numpy(),
            "rng": ((g.high - g.low) / g.close).to_numpy(),
            "lvol": np.log(g.volume.replace(0, np.nan)).ffill().diff().to_numpy(),
            "tod_sin": np.sin(2 * np.pi * tod), "tod_cos": np.cos(2 * np.pi * tod),
            "dow_sin": np.sin(2 * np.pi * dow), "dow_cos": np.cos(2 * np.pi * dow),
        }))
    return pd.concat(out, ignore_index=True).dropna().reset_index(drop=True)


def _quantile_cols(cv):
    """Map q10/q50/q90 to MQLoss output columns by suffix (nf: -lo-80.0 / -median
    / -hi-80.0). Suffix-based, so it survives the model-name prefix and the
    80 vs 80.0 spelling."""
    return {
        "q10": next(c for c in cv.columns if "-lo-" in c),
        "q50": next(c for c in cv.columns if c.endswith("-median")),
        "q90": next(c for c in cv.columns if "-hi-" in c),
    }


def to_score_frame(cv, nf_df, model):
    """Map a cross_validation result to score()'s columns, in PRICE space.

    cv rows: unique_id, ds, cutoff, y, <3 quantile cols>. y and predictions are
    log-close, so everything is exp()'d back to price. `last` is the close at the
    origin (ds == cutoff), pulled from the input series.
    """
    qc = _quantile_cols(cv)
    last = (nf_df.rename(columns={"ds": "cutoff", "y": "last_log"})[["unique_id", "cutoff", "last_log"]])
    m = cv.merge(last, on=["unique_id", "cutoff"], how="left")
    m["h"] = ((m.ds - m.cutoff) / pd.Timedelta(FREQ)).round().astype(int)  # horizon in BARS

    out = pd.DataFrame({
        "model": model, "asset": m.unique_id.to_numpy(), "h": m.h.to_numpy(),
        "ds": m.ds.to_numpy(), "cutoff": m.cutoff.to_numpy(),
        "y": np.exp(m.y.to_numpy()), "last": np.exp(m["last_log"].to_numpy()),
    })
    for tag, c in qc.items():
        out[tag] = np.exp(m[c].to_numpy())
    return out.dropna()


def plot_last_forecast(res, nf_df, name, context=96):
    """Recent price history + forecast continuing off the end, per coin - PNG.

    Only the last `context` bars are shown so the short forecast is visible.
    jieyima-style: blue history, orange forecast, shaded 80% band.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coins = sorted(res.asset.unique())
    cols = 2
    rows = (len(coins) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3 * rows))
    axes = axes.flatten()
    for ax, coin in zip(axes, coins):
        cutoff = res[res.asset == coin].cutoff.max()
        hist = (nf_df[(nf_df.unique_id == coin) & (nf_df.ds <= cutoff)]
                .assign(price=lambda d: np.exp(d.y)).tail(context))
        fc = res[(res.asset == coin) & (res.cutoff == cutoff)].sort_values("h")
        fx = pd.to_datetime(np.r_[hist.ds.iloc[-1], fc.ds.to_numpy()])  # keep dtype consistent
        j = lambda a: np.concatenate([[hist.price.iloc[-1]], a])  # connect to history
        ax.plot(hist.ds, hist.price, color="#1f77b4", lw=1.2, label="Historical Price")
        ax.fill_between(fx, j(fc.q10.to_numpy()), j(fc.q90.to_numpy()),
                        color="#ff7f0e", alpha=0.2, label="80% band")
        ax.plot(fx, j(fc.q50.to_numpy()), color="#ff7f0e", lw=1.5, label="Predicted Price")
        ax.set_title(f"{name.upper()} forecast on {coin}", fontsize=10)
        ax.set_ylabel("Price in USD", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
    for ax in axes[len(coins):]:
        ax.axis("off")
    fig.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = Path("plots") / f"forecast_{name}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def cross_val_score(nf, name):
    """Run cross_validation on a fitted-config NeuralForecast, score, plot, log."""
    import time

    print(f"=== {name} ===")
    nf_df = to_nf()
    print(f"data: {len(nf_df):,} rows, {nf_df.unique_id.nunique()} coins, "
          f"{nf_df.ds.min()} .. {nf_df.ds.max()}")
    print(f"config: horizon {H_HF} bars, lookback {INPUT_HF} bars, {N_WINDOWS} folds, "
          f"step {STEP}h, exog {HIST_EXOG}")
    print("training (Lightning progress bar below)...\n")

    t0 = time.time()
    cv = nf.cross_validation(nf_df, n_windows=N_WINDOWS, step_size=STEP).reset_index()
    print(f"\ntrained + cross-validated in {(time.time() - t0) / 60:.1f} min")

    # Save the raw cross-validation BEFORE scoring - training is expensive, a
    # scoring bug must never cost a retrain. Re-score later with score_saved().
    cv_path = Path("data") / f"_cv_{name}.parquet"
    cv.to_parquet(cv_path, index=False)
    print(f"saved raw cv -> {cv_path}")

    return _score_cv(cv, nf_df, name)


def _conformal(res, cal_frac=0.6):
    """Widen each horizon's band until it covers, per horizon, on a time-split
    calibration set. k_h is forced non-decreasing in h - uncertainty should grow
    with the horizon, not jump around from thin-sample noise. Returns the
    calibrated TEST slice (model tag suffixed '_cal')."""
    model = res.model.iloc[0]
    cutoffs = np.sort(res.cutoff.unique())
    split = cutoffs[int(len(cutoffs) * cal_frac)]
    cal, test = res[res.cutoff < split].copy(), res[res.cutoff >= split].copy()

    half = (cal.q90 - cal.q50).clip(lower=1e-9)
    cal["ratio"] = (cal.y - cal.q50).abs() / half
    # Per-horizon widening: 80th percentile of |error|/half-band covers 80%.
    # Rolling-median smooth (not cummax - that over-inflated long horizons to 100%).
    # clip(lower=1) so calibration only WIDENS, never tightens: at short horizons
    # the native quantiles are already well-calibrated and shrinking them undercovers.
    k = (cal.groupby("h")["ratio"].quantile(0.80)
         .sort_index().rolling(5, center=True, min_periods=1).median().clip(lower=1.0))

    kh = test.h.map(k).to_numpy()
    hw = (test.q90 - test.q50).to_numpy()
    test["q10"], test["q90"] = test.q50 - kh * hw, test.q50 + kh * hw
    test["model"] = f"{model}_cal"
    return test, res[res.cutoff >= split]


def _score_cv(cv, nf_df, name):
    model = [c for c in cv.columns if c.endswith("-median")][0].replace("-median", "")
    res = to_score_frame(cv, nf_df, model)
    cal_test, raw_test = _conformal(res)

    hmax = int(res.h.max())
    print(f"\nRESULTS ({hmax}-bar horizon, {res.asset.nunique()} coins, "
          f"test = last {cal_test.cutoff.nunique()} origins)")
    print("\nPOINT metrics (median; R2 on RETURNS, not price):")
    print(point_metrics(cal_test).round(4).to_string())
    print("\nINTERVAL metrics, raw (native quantiles) vs calibrated:")
    print(pd.concat([score(raw_test.assign(model=model)), score(cal_test)]).round(2).to_string())
    print("\ncalibrated coverage % by horizon (target 80):")
    cov = coverage_by_h(cal_test)
    keep = [c for c in cov.columns if c in (1, 2, 4, 6, 8, hmax)]
    print(cov[keep].round(1).to_string())

    log(f"hf_{name}_cal", score(cal_test), horizon=hmax)
    plot_last_forecast(cal_test, nf_df, name)  # history + forecast continuation
    return cal_test


def score_saved(name):
    """Re-score a saved cross_validation without retraining. Use after fixing
    anything in the scoring/plotting path: python -c "from neural.nf_core import
    score_saved; score_saved('nhits')"."""
    cv = pd.read_parquet(Path("data") / f"_cv_{name}.parquet")
    return _score_cv(cv, to_nf(), name)


def calibrate_conformal(name, cal_frac=0.6):
    """Re-run conformal calibration on a saved cv without retraining. Same logic
    cross_val_score already applies - use this only to re-check an old cv."""
    cv = pd.read_parquet(Path("data") / f"_cv_{name}.parquet")
    return _score_cv(cv, to_nf(), name)


def ensemble(names=("lstm", "gru", "nhits")):
    """Average the quantile forecasts of already-trained models. Reads their saved
    cvs - no training. Ensembling near-independent models almost always beats any
    single one, so this is free accuracy on top of what you already ran."""
    keys = ["unique_id", "ds", "cutoff"]
    parts = []
    for n in names:
        cv = pd.read_parquet(Path("data") / f"_cv_{n}.parquet")
        qc = _quantile_cols(cv)
        d = cv[keys + ["y"]].copy()
        for tag, c in qc.items():
            d[tag] = cv[c].to_numpy()      # log-space quantiles
        parts.append(d.set_index(keys))
    stacked = pd.concat(parts)
    avg = stacked.groupby(level=keys).mean().reset_index()   # mean across models
    avg = avg.rename(columns={"q10": "ENS-lo-80.0", "q50": "ENS-median", "q90": "ENS-hi-80.0"})
    return _score_cv(avg, to_nf(), "ensemble")
