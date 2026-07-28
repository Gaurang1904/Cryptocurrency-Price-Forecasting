"""Fetch prices, then funding and open interest. Run this first."""

import sys

from crypto.data import fetch_ohlcv, fetch_derivatives

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ticker names aren't all cp1252

if __name__ == "__main__":
    fetch_ohlcv()
    print()
    fetch_derivatives()
