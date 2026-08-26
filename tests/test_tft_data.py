import unittest
import pandas as pd

from neural.tft_data import complete_ohlcv_grid, validate_ohlcv_15m


def tiny_raw():
    dates = pd.to_datetime([
        "2026-01-01 00:00Z", "2026-01-01 00:15Z", "2026-01-01 00:45Z",
    ])
    rows = []
    for asset, offset in (("BTC", 0.0), ("ETH", 100.0)):
        for i, date in enumerate(dates):
            close = 100.0 + offset + i
            rows.append({
                "asset": asset, "date": date, "open": close - 0.2,
                "high": close + 0.5, "low": close - 0.5,
                "close": close, "volume": 10.0 + i,
            })
    return pd.DataFrame(rows)


class TftDataContractTests(unittest.TestCase):
    def test_grid_completion_is_causal_and_marks_missing_bars(self):
        raw = validate_ohlcv_15m(
            tiny_raw(), assets=("BTC", "ETH"),
            as_of=pd.Timestamp("2026-01-01 01:00Z"),
        )
        grid, stats = complete_ohlcv_grid(raw, assets=("BTC", "ETH"))
        missing = grid[grid.date == pd.Timestamp("2026-01-01 00:30Z")]
        self.assertEqual(set(missing.missing_bar), {1})
        self.assertEqual(set(missing.volume), {0.0})
        self.assertEqual(missing.set_index("asset").loc["BTC", "close"], 101.0)
        self.assertEqual(stats["BTC"].synthetic_bars, 1)
        self.assertEqual(stats["BTC"].max_gap_bars, 1)

    def test_validation_rejects_duplicate_and_incomplete_bars(self):
        raw = tiny_raw()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_ohlcv_15m(
                pd.concat([raw, raw.iloc[[0]]]), assets=("BTC", "ETH"),
                as_of=pd.Timestamp("2026-01-01 01:00Z"),
            )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_ohlcv_15m(
                raw, assets=("BTC", "ETH"),
                as_of=pd.Timestamp("2026-01-01 00:50Z"),
            )

    def test_validation_rejects_invalid_ohlcv_and_missing_assets(self):
        raw = tiny_raw()
        raw.loc[0, "high"] = raw.loc[0, "low"] - 1
        with self.assertRaisesRegex(ValueError, "OHLC"):
            validate_ohlcv_15m(
                raw, assets=("BTC", "ETH"),
                as_of=pd.Timestamp("2026-01-01 01:00Z"),
            )
        with self.assertRaisesRegex(ValueError, "assets"):
            validate_ohlcv_15m(
                tiny_raw().query("asset == 'BTC'"), assets=("BTC", "ETH"),
                as_of=pd.Timestamp("2026-01-01 01:00Z"),
            )

    def test_validation_rejects_non_utc_or_misaligned_timestamps(self):
        raw = tiny_raw(); raw["date"] = raw.date.dt.tz_convert("Asia/Kolkata")
        with self.assertRaisesRegex(ValueError, "UTC"):
            validate_ohlcv_15m(raw, assets=("BTC", "ETH"), as_of=pd.Timestamp("2026-01-01 01:00Z"))
        raw = tiny_raw(); raw.loc[0, "date"] = pd.Timestamp("2026-01-01 00:07Z")
        with self.assertRaisesRegex(ValueError, "aligned"):
            validate_ohlcv_15m(raw, assets=("BTC", "ETH"), as_of=pd.Timestamp("2026-01-01 01:00Z"))

    def test_validation_rejects_insufficient_history(self):
        raw = tiny_raw().groupby("asset", as_index=False).head(1)
        with self.assertRaisesRegex(ValueError, "history"):
            validate_ohlcv_15m(raw, assets=("BTC", "ETH"), as_of=pd.Timestamp("2026-01-01 01:00Z"))

    def test_validation_rejects_unsupported_assets(self):
        raw = tiny_raw(); extra = raw.iloc[:3].copy(); extra["asset"] = "DOGE"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_ohlcv_15m(pd.concat([raw, extra]), assets=("BTC", "ETH"), as_of=pd.Timestamp("2026-01-01 01:00Z"))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_ohlcv_15m(raw, assets=("BTC", "DOGE"), as_of=pd.Timestamp("2026-01-01 01:00Z"))

    def test_validation_rejects_missing_or_nonfinite_ohlcv(self):
        for value in (None, float("nan"), float("inf"), float("-inf")):
            raw = tiny_raw(); raw.loc[0, "close"] = value
            with self.assertRaisesRegex(ValueError, "finite"):
                validate_ohlcv_15m(raw, assets=("BTC", "ETH"), as_of=pd.Timestamp("2026-01-01 01:00Z"))

    def test_validation_rejects_numeric_string_ohlcv(self):
        raw = tiny_raw()
        raw["close"] = raw["close"].astype(object)
        raw.loc[0, "close"] = "100.0"
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_ohlcv_15m(raw, assets=("BTC", "ETH"), as_of=pd.Timestamp("2026-01-01 01:00Z"))
