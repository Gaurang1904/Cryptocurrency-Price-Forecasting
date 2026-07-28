"""LSTM: the classic sequence baseline. Reads the window step by step, carries
memory, predicts all H horizons from the final hidden state.
"""

import torch.nn as nn

from neural.core import H


class LSTM(nn.Module):
    def __init__(self, channels, hidden=32, horizon=H):
        super().__init__()
        self.lstm = nn.LSTM(channels, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, horizon))

    def forward(self, x):                # x: (B, L, C)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])     # last timestep -> H outputs


def build(channels):
    return LSTM(channels)


if __name__ == "__main__":
    from neural.core import train_and_save
    train_and_save(build, "lstm")
