# Results

All numbers are out-of-sample, walk-forward. Features are verified causal
(`crypto/features.check_causal` + a corruption test on the 15-min pipeline).
Daily and 15-min are **separate pipelines** — their numbers are not comparable
to each other and are reported apart.

---

## The question and the finding

> Can we predict cryptocurrency price?

| tested | result |
|---|---|
| Price **direction** (up/down) | **No** — ~50% across every model, incl. Transformer |
| Price **level** | **No** — nothing beats a flat "price stays put" baseline |
| **Volatility** | **Yes, mildly** — beats a rolling-average baseline, but simple methods capture most of it |

Scaling the models (4× training steps, 2× capacity, 2× lookback) produced **zero
accuracy gain** — evidence the ceiling is the data, not the model.

---

## Daily volatility — 7-day interval forecast (~1,445 origins, 5 assets)

Predict daily volatility, turn it into a calibrated price interval. Lower pinball
better; coverage target is 80%.

| model | family | pinball | coverage % |
|---|---|---|---|
| LSTM | neural | **162.97** | 80.0 |
| DLinear | neural | 163.22 | 79.7 |
| XGBoost | tree | 166.98 | 79.0 |
| LightGBM | tree | 167.92 | 78.7 |
| vol_21d (baseline) | — | 167.99 | 79.3 |

All models cluster within ~3% — near-interchangeable. The neural edge is the
sequence input, not depth (DLinear ≈ LSTM).

---

## 15-min high-frequency — 2-hour forecast (~1M rows, 16 test origins)

Same config across all four (1000 steps, 96-bar lookback). MAPE is the median
forecast error; DirAcc is up/down accuracy on the 2-hour move; R² is on **returns**
(price-R² would be a meaningless ~0.99).

| model | MAPE % | R²(returns) | DirAcc % | raw coverage % |
|---|---|---|---|---|
| N-HiTS | **0.27** | +0.03 | 66.2 | 75.8 |
| GRU | 0.34 | -0.28 | 58.8 | 80.8 |
| LSTM | 0.35 | -0.31 | 63.8 | 81.3 |
| TFT | 0.30 | -0.10 | **48.8** | 65.0 |

**The DirAcc numbers are a test-window artifact, not signal.** Three momentum-
extrapolating models scored 59-66% by riding a trending 8-day test window; TFT —
the only model that learns patterns rather than extrapolating — scored **below 50%**.
Real edge would appear in the most capable model, not vanish from it. A better-
trained LSTM (4000 steps) also dropped to 52.5%, confirming it.

---

## Method

- **Walk-forward** validation, expanding window — never a single split.
- **Baselines first** — flat / drift / seasonal; no model is trusted until it beats them.
- **Causal features** — a corruption test proves no look-ahead (the leak that
  invalidated the original version of this project).
- **Conformal calibration** — neural native quantiles are overconfident; empirical
  per-horizon widening restores ~80% coverage.

---

## Plots

Return correlation across coins (why panel training works — they move together):

![correlation](docs/eda_correlation.png)

Cumulative returns:

![returns](docs/eda_returns.png)

LSTM 2-hour forecast with 80% interval:

![forecast](docs/forecast_lstm.png)

---

## Honest caveats

- 15-min coverage/DirAcc come from only 16 test origins — thin. The direction
  finding is robust (confirmed by TFT and the scaled LSTM); the exact coverage % is not.
- 0.27% MAPE at a 2-hour horizon is low mostly because price barely moves over 2
  hours — it is honest, but close to a naive-lag baseline. It is not evidence of edge.
- This is a research finding, not a trading system. No live P&L is claimed.
