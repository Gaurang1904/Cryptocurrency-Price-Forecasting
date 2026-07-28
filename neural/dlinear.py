"""DLinear: series decomposition + linear layers. The neural CONTROL.

The DLinear paper's point: a plain linear model on a decomposed look-back window
beats most transformers on forecasting. If this ties LightGBM, neural depth is
adding nothing on this data - which is itself the finding.
"""

import torch
import torch.nn as nn

from neural.core import H, LOOKBACK


class DLinear(nn.Module):
    def __init__(self, channels, lookback=LOOKBACK, horizon=H, kernel=25):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel, stride=1, padding=kernel // 2)
        flat = lookback * channels
        self.trend = nn.Linear(flat, horizon)     # low-frequency component
        self.seasonal = nn.Linear(flat, horizon)  # residual

    def forward(self, x):                          # x: (B, L, C)
        t = x.transpose(1, 2)                      # (B, C, L)
        trend = self.pool(t).transpose(1, 2)       # moving average back to (B, L, C)
        trend = trend[:, : x.size(1), :]           # padding can overshoot by one
        seasonal = x - trend
        return self.trend(trend.flatten(1)) + self.seasonal(seasonal.flatten(1))


def build(channels):
    return DLinear(channels)


if __name__ == "__main__":
    from neural.core import train_and_save
    train_and_save(build, "dlinear")
