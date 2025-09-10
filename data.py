import ccxt
import pandas as pd
import time

# Setup
exchange = ccxt.binance()
symbol = 'BNB/USDT'
timeframe = '1d'
limit = 1000  # Binance max per fetch

# Step 1: Get earliest available candle
start = exchange.fetch_ohlcv(symbol, timeframe, exchange.parse8601('2015-01-01T00:00:00Z'), 1)
if not start:
    print("No data found.")
    exit()

since = start[0][0]  # first candle timestamp in ms
now = exchange.milliseconds()  # current time

all_ohlcv = []

# Step 2: Loop and fetch in batches
while since < now:
    try:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        if not candles:
            break
        all_ohlcv.extend(candles)

        print(f"Fetched: {pd.to_datetime(candles[0][0], unit='ms')} → {pd.to_datetime(candles[-1][0], unit='ms')}")

        since = candles[-1][0] + 1  # move forward
        time.sleep(exchange.rateLimit / 1000)

    except Exception as e:
        print(f"Error: {e}")
        break

# Step 3: Save to CSV
df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.to_csv('BNB_USDT_1d.csv', index=False)
print("✅ Saved as BNB_USDT_1d.csv")
