from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crypto.backtest import coverage_by_h, score
from crypto.evaluation import metric_tables, validate_predictions


BASELINE = "vol_21d"


def require_volatility_baseline(frame):
    if BASELINE not in set(frame.model):
        raise ValueError(
            "evaluation charts require the vol_21d baseline; regenerate artifacts "
            "with the baseline included"
        )


def _caption(frame):
    dates = frame.origin.dt.strftime("%Y-%m-%d")
    origins = frame[["asset", "origin"]].drop_duplicates().shape[0]
    return f"OOS {dates.min()} to {dates.max()} | {origins} origins"


def _save(fig, output_dir, name):
    fig.tight_layout()
    fig.savefig(Path(output_dir) / name, dpi=150)
    plt.close(fig)


def _title(ax, label, frame):
    ax.set_title(f"{label}\n{_caption(frame)}")


def _forecast_bands(frame, output_dir):
    asset = frame.asset.iloc[0]
    h = int(frame.h.min())
    rows = frame[(frame.asset == asset) & (frame.h == h)].sort_values("origin")
    fig, ax = plt.subplots(figsize=(9, 4))
    actual = rows.drop_duplicates("origin")
    ax.plot(actual.origin, actual.y, color="black", linewidth=1.5, label="actual")
    for model, group in rows.groupby("model", sort=False):
        style = "--" if model == BASELINE else "-"
        ax.plot(group.origin, group.q50, linestyle=style, label=f"{model} median")
        ax.fill_between(group.origin, group.q10, group.q90, alpha=0.12)
    ax.set_ylabel("price")
    ax.legend(fontsize="small", ncol=2)
    _title(ax, f"{asset} {h}-day forecast bands", frame)
    _save(fig, output_dir, "forecast_bands.png")


def _coverage(frame, output_dir):
    table = coverage_by_h(frame)
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, row in table.iterrows():
        ax.plot(row.index, row.values, marker="o", label=model)
    ax.axhline(80, color="black", linestyle="--", label="80% target")
    ax.set(xlabel="horizon (days)", ylabel="coverage (%)", ylim=(0, 100))
    ax.legend(fontsize="small")
    _title(ax, "Interval coverage by horizon", frame)
    _save(fig, output_dir, "coverage_by_horizon.png")


def _pinball_by_model(frame, output_dir):
    table = score(frame).sort_values("pinball_%")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table.index, table["pinball_%"])
    ax.set(ylabel="pinball (% of price)")
    _title(ax, "Normalized pinball loss by model", frame)
    _save(fig, output_dir, "pinball_by_model.png")


def _volatility_fit(frame, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    limit = float(max(frame.sigma.max(), frame.rv.max()))
    for model, group in frame.groupby("model", sort=False):
        ax.scatter(group.rv, group.sigma, alpha=0.65, label=model)
    ax.plot([0, limit], [0, limit], color="black", linestyle="--", label="perfect fit")
    ax.set(xlabel="realized volatility", ylabel="predicted volatility")
    ax.legend(fontsize="small")
    _title(ax, "Predicted versus realized volatility", frame)
    _save(fig, output_dir, "volatility_fit.png")


def _grouped_pinball(table, group, label, frame, output_dir, name):
    values = (table.reset_index().groupby(["model", group], sort=False)["pinball_%"]
              .mean().unstack(group))
    fig, ax = plt.subplots(figsize=(8, 4))
    values.T.plot.bar(ax=ax)
    ax.set(xlabel=group.replace("_", " "), ylabel="pinball (% of price)")
    ax.legend(title="model", fontsize="small")
    _title(ax, label, frame)
    _save(fig, output_dir, name)


def render_bundle(frame, output_dir):
    """Render daily out-of-sample diagnostics from validated saved predictions."""
    validate_predictions(frame)
    require_volatility_baseline(frame)
    _forecast_bands(frame, output_dir)
    _coverage(frame, output_dir)
    _pinball_by_model(frame, output_dir)
    _volatility_fit(frame, output_dir)
    tables = metric_tables(frame)
    _grouped_pinball(
        tables["by_asset"], "asset", "Normalized pinball loss by asset", frame,
        output_dir, "performance_by_asset.png",
    )
    _grouped_pinball(
        tables["by_regime"], "regime", "Normalized pinball loss by regime", frame,
        output_dir, "performance_by_regime.png",
    )

