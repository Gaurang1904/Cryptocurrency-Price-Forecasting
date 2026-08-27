"""Static, next-day TFT forecast diagnostics for the untouched test split."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neural.tft_evaluation import apply_regimes, fit_regime_cutpoints, tft_metric_tables


BASELINES = {"persistence_vol", "momentum_vol"}
_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9")
_STYLES = ("-", "--", ":", "-.", (0, (3, 1, 1, 1)))
_MARKERS = ("o", "s", "^", "D", "P")


def _caption(test):
    origins = pd.to_datetime(test.origin)
    return (
        f"Test split | OOS {origins.min():%Y-%m-%d} to {origins.max():%Y-%m-%d} "
        f"| {origins.nunique()} distinct daily origins"
    )


def _title(ax, label, test):
    ax.set_title(f"{label}\n{_caption(test)}")


def _model_style(model, models):
    index = list(models).index(model)
    return _COLORS[index % len(_COLORS)], _STYLES[index % len(_STYLES)], _MARKERS[index % len(_MARKERS)]


def _save(fig, output_dir, name):
    path = Path(output_dir) / name
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return path
    finally:
        plt.close(fig)


def _forecast_path(_, test, output_dir):
    latest = test.origin.max()
    rows = test.loc[(test.model == "tft_calibrated") & test.origin.eq(latest)]
    assets = sorted(rows.asset.unique())
    fig, axes = plt.subplots(len(assets), 1, figsize=(10, 3.5 * len(assets)), squeeze=False)
    for ax, asset in zip(axes[:, 0], assets):
        group = rows.loc[rows.asset.eq(asset)].sort_values("h")
        ax.plot(group.h, group.y, color="#222222", marker="o", linewidth=1.8, label="actual")
        ax.plot(group.h, group.q50, color="#0072B2", linestyle="-", marker="s", markevery=12,
                linewidth=1.8, label="calibrated TFT q50")
        ax.fill_between(group.h, group.q10, group.q90, color="#0072B2", alpha=0.20,
                        label="calibrated TFT q10–q90")
        ax.set(xlabel="15-minute horizon", ylabel="price", title=f"{asset}: latest test origin {latest:%Y-%m-%d}")
        ax.legend(fontsize="small")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(_caption(test), y=1.02)
    return _save(fig, output_dir, "forecast_path.png")


def _next_day_forecasts(_, test, output_dir):
    rows = test.loc[test.h.eq(96)].copy()
    assets, models = sorted(rows.asset.unique()), sorted(rows.model.unique())
    fig, axes = plt.subplots(len(assets), 1, figsize=(11, 3.5 * len(assets)), sharex=True, squeeze=False)
    for ax, asset in zip(axes[:, 0], assets):
        group = rows.loc[rows.asset.eq(asset)]
        actual = group.drop_duplicates("origin").sort_values("origin")
        ax.plot(actual.origin, actual.y, color="#222222", marker="o", linewidth=1.8, label="actual")
        for model in models:
            prediction = group.loc[group.model.eq(model)].sort_values("origin")
            color, style, marker = _model_style(model, models)
            ax.plot(prediction.origin, prediction.q50, color=color, linestyle=style, marker=marker,
                    markersize=4, linewidth=1.5, label=f"{model} q50")
        ax.set(ylabel="next-day price", title=asset)
        ax.legend(fontsize="x-small", ncol=2)
        ax.grid(axis="y", alpha=0.25)
    axes[-1, 0].set_xlabel("test origin")
    fig.suptitle(f"Next-day (h=96) forecasts\n{_caption(test)}", y=1.02)
    return _save(fig, output_dir, "next_day_forecasts.png")


def _returns_scatter(_, test, output_dir):
    rows = test.loc[test.h.eq(96)].copy()
    models = sorted(rows.model.unique())
    fig, ax = plt.subplots(figsize=(7, 6))
    extent = 0.0
    for model in models:
        group = rows.loc[rows.model.eq(model)]
        actual = group.y / group["last"] - 1
        predicted = group.q50 / group["last"] - 1
        extent = max(extent, float(np.abs(np.r_[actual.to_numpy(), predicted.to_numpy()]).max()))
        color, _, marker = _model_style(model, models)
        ax.scatter(actual, predicted, color=color, marker=marker, alpha=0.7, label=model)
    limit = max(extent * 1.05, 0.001)
    ax.axhline(0, color="#555555", linestyle=":", linewidth=1, label="zero return")
    ax.axvline(0, color="#555555", linestyle=":", linewidth=1)
    ax.plot([-limit, limit], [-limit, limit], color="#222222", linestyle="--", label="identity")
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="realized return", ylabel="predicted return")
    _title(ax, "Next-day predicted versus realized returns (h=96)", test)
    ax.legend(fontsize="small")
    ax.grid(alpha=0.25)
    return _save(fig, output_dir, "returns_scatter.png")


def _performance(table, group, label, test, output_dir, name):
    frame = table.reset_index()
    models, groups = sorted(frame.model.unique()), sorted(frame[group].unique())
    x = np.arange(len(groups))
    width = 0.8 / len(models)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for index, model in enumerate(models):
        values = frame.loc[frame.model.eq(model)].set_index(group).reindex(groups)
        color, _, _ = _model_style(model, models)
        offset = (index - (len(models) - 1) / 2) * width
        axes[0].bar(x + offset, values["pinball_%"], width, color=color, hatch=("", "//", "xx", "..", "\\\\")[index % 5], label=model)
        axes[1].bar(x + offset, values["direction_accuracy"], width, color=color, hatch=("", "//", "xx", "..", "\\\\")[index % 5], label=model)
    axes[0].set_ylabel("pinball (% of price)")
    axes[1].set_ylabel("direction accuracy (%)")
    axes[1].set_xlabel(label.lower())
    axes[1].set_xticks(x, groups)
    axes[1].set_ylim(0, 100)
    axes[0].legend(title="model", fontsize="small")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Next-day performance by {label.lower()} (h=96)\n{_caption(test)}", y=0.98)
    return _save(fig, output_dir, name)


def _performance_by_asset(calibration, test, output_dir):
    return _performance(tft_metric_tables(calibration, test)["by_asset"], "asset", "Asset", test, output_dir,
                        "performance_by_asset.png")


def _performance_by_regime(calibration, test, output_dir):
    return _performance(tft_metric_tables(calibration, test)["by_regime"], "regime", "Fixed regime", test, output_dir,
                        "performance_by_regime.png")


def _calibration_comparison(calibration, test, output_dir):
    raw = test.loc[test.model.eq("tft_raw")]
    calibrated = test.loc[test.model.eq("tft_calibrated")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, frame, color, style, marker in (
        ("raw TFT", raw, "#D55E00", "--", "s"),
        ("calibrated TFT", calibrated, "#0072B2", "-", "o"),
    ):
        coverage = ((frame.y >= frame.q10) & (frame.y <= frame.q90)).groupby(frame.h).mean().mul(100)
        ax.plot(coverage.index, coverage.values, color=color, linestyle=style, marker=marker, label=label)
    ax.axhline(80, color="#222222", linestyle=":", label="80% reference")
    ax.set(xlabel="15-minute horizon", ylabel="coverage (%)", ylim=(0, 100))
    _title(ax, "Raw versus calibrated interval coverage", test)
    ax.legend(fontsize="small")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, output_dir, "calibration_comparison.png")


REPORT_RENDERERS = (
    _forecast_path,
    _next_day_forecasts,
    _returns_scatter,
    _performance_by_asset,
    _performance_by_regime,
    _calibration_comparison,
)


def _validate_report_inputs(calibration, test):
    if set(test.split) != {"test"}:
        raise ValueError("headline graphs require only test rows")
    if not BASELINES.issubset(set(test.model)):
        raise ValueError("both baseline models are required")
    if not {"tft_raw", "tft_calibrated"}.issubset(set(test.model)):
        raise ValueError("raw and calibrated TFT models are required")
    if set(calibration.split) != {"calibration"}:
        raise ValueError("calibration rows must use the calibration split")


def render_tft_report(calibration, test, output_dir):
    """Write TFT metric tables and static diagnostics from calibration and test rows."""
    _validate_report_inputs(calibration, test)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for name, table in tft_metric_tables(calibration, test).items():
            path = output_dir / f"metrics_{name}.csv"
            table.to_csv(path)
            paths.append(path)
        for renderer in REPORT_RENDERERS:
            paths.append(renderer(calibration, test, output_dir))
        return paths
    finally:
        plt.close("all")
