import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from arch import arch_model

# === Load and prepare the data ===
df = pd.read_csv("BNB_USDT_1d.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

df.rename(columns={
    "open": "Open", "high": "High", "low": "Low",
    "close": "Close", "volume": "Volume"
}, inplace=True)

# === Technical Indicators ===
df["RSI_14"] = RSIIndicator(close=df["Close"], window=14).rsi()
df["MACD_diff"] = MACD(close=df["Close"]).macd_diff()
df["EMA_9"] = EMAIndicator(close=df["Close"], window=9).ema_indicator()
df["EMA_21"] = EMAIndicator(close=df["Close"], window=21).ema_indicator()
df["ATR_14"] = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"]).average_true_range()
bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
df["BB_upper"] = bb.bollinger_hband()
df["BB_lower"] = bb.bollinger_lband()
df["rolling_std_6h"] = df["Close"].rolling(window=6).std()
df["rolling_std_12h"] = df["Close"].rolling(window=12).std()

# === Lag Features ===
for lag in [1, 3, 6, 24]:
    df[f"lag_{lag}h"] = df["Close"].shift(lag)

# === GARCH Volatility ===
returns = 100 * df["Close"].pct_change().dropna()
garch_model = arch_model(returns, vol="Garch", p=1, q=1)
garch_fit = garch_model.fit(disp="off")
df.loc[returns.index, "garch_vol"] = garch_fit.conditional_volatility

# === Temporal Features ===
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

# === Target Variables ===
df["target_return_1h"] = df["Close"].pct_change(periods=1).shift(-1)
df["target_direction_1h"] = (df["target_return_1h"] > 0).astype(int)
df["is_spike"] = (df["target_return_1h"].abs() > 0.02).astype(int)  # 2% move threshold

# === Final cleanup ===
df = df.dropna().reset_index(drop=True)
df.to_csv("BNB_FE.csv", index=False)
print("✅ Feature engineering complete and saved to BNB_FE.csv")
