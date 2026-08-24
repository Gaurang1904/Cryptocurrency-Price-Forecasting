"""Fetch prices, then funding and open interest. Run this first.

  python fetch.py            daily OHLCV + funding + open interest
  python fetch.py --tf 1h    hourly OHLCV (2020->now)
  python fetch.py --tf 15m   15-min OHLCV (2021->now) - slow, ~1M rows
"""

import argparse
import sys

from crypto.data import fetch_ohlcv, fetch_ohlcv_1h, fetch_ohlcv_15m, fetch_derivatives

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ticker names aren't all cp1252

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", choices=["1d", "1h", "15m"], default="1d")
    args = ap.parse_args()

    if args.tf == "15m":
        fetch_ohlcv_15m()
    elif args.tf == "1h":
        fetch_ohlcv_1h()
    else:
        fetch_ohlcv()
        print()
        fetch_derivatives()  # funding/OI only meaningful on the daily run
