"""Exploratory plots as matplotlib PNGs - jieyima-notebook style.

  python -m crypto.eda            # daily dataset
  python -m crypto.eda --tf 15m   # 15-min dataset

Writes to plots/:
  eda_prices.png       per-coin OHLC price history grid
  eda_returns.png      cumulative return of every coin, overlaid
  eda_correlation.png  return-correlation heatmap
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto.data import OHLCV_OUT, OHLCV_1H_OUT, OHLCV_15M_OUT

PATHS = {"1d": OHLCV_OUT, "1h": OHLCV_1H_OUT, "15m": OHLCV_15M_OUT}
PLOTS = Path("plots")


def _save(fig, name):
    PLOTS.mkdir(exist_ok=True)
    out = PLOTS / name
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def price_grid(df):
    assets = sorted(df.asset.unique())
    start = pd.to_datetime(df.date).min().year
    cols = 2
    rows = (len(assets) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3 * rows))
    axes = axes.flatten()
    for ax, a in zip(axes, assets):
        g = df[df.asset == a].sort_values("date")
        for field, color in [("open", "#1f77b4"), ("high", "#ff7f0e"),
                             ("low", "#2ca02c"), ("close", "#d62728")]:
            ax.plot(g.date, g[field], color=color, lw=0.8, label=field)
        ax.set_title(f"Historical price of {a} since {start}", fontsize=10)
        ax.set_ylabel("price in USD", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
    for ax in axes[len(assets):]:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, "eda_prices.png")


def return_overlay(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    for a in sorted(df.asset.unique()):
        g = df[df.asset == a].sort_values("date")
        ax.plot(g.date, g.close / g.close.iloc[0], lw=1.3, label=a)
    ax.set_title("Cumulative return of each cryptocurrency")
    ax.set_ylabel("return ratio")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "eda_returns.png")


def correlation_heatmap(df):
    wide = (df.assign(ret=df.groupby("asset").close.transform(lambda s: np.log(s).diff()))
              .pivot_table(index="date", columns="asset", values="ret"))
    corr = wide.corr()
    fig, ax = plt.subplots(figsize=(1.2 * len(corr) + 2, 1.2 * len(corr) + 1))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), corr.columns)
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Return correlation across coins")
    fig.colorbar(im, ax=ax, shrink=0.8, label="corr")
    return _save(fig, "eda_correlation.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", choices=PATHS, default="1d")
    args = ap.parse_args()
    df = pd.read_parquet(PATHS[args.tf])
    print(f"{len(df):,} rows, {df.asset.nunique()} coins, {args.tf}")
    price_grid(df)
    return_overlay(df)
    correlation_heatmap(df)
