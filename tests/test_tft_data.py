import unittest

import numpy as np
import pandas as pd

from neural.tft_data import (
    HIST_EXOG,
    PreparedTFTData,
    assert_tft_feature_causality,
    complete_ohlcv_grid,
    eligible_daily_origins,
    prepare_tft_data,
    validate_ohlcv_15m,
)


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


def long_raw_fixture(assets=("BTC", "ETH"), periods=14 * 96):
    dates = pd.date_range("2025-12-20", periods=periods, freq="15min", tz="UTC")
    rows = []
    for index, asset in enumerate(assets, start=1):
        close = index * 100 + np.arange(periods) * 0.01
        rows.append(pd.DataFrame({
            "asset": asset, "date": dates, "open": close,
            "high": close * 1.001, "low": close * 0.999,
            "close": close, "volume": 10 + np.arange(periods) % 7,
        }))
    raw = pd.concat(rows, ignore_index=True)
    return raw, pd.Timestamp("2025-12-28 23:45Z"), dates[-1] + pd.Timedelta("15min")


def prepared_fixture_with_one_missing_target_bar():
    raw, _, as_of = long_raw_fixture()
    raw = raw[raw.date != pd.Timestamp("2026-01-02 12:00Z")].reset_index(drop=True)
    return prepare_tft_data(raw, assets=("BTC", "ETH"), as_of=as_of)


class TftFeatureTests(unittest.TestCase):
    def test_prepared_frame_has_exact_causal_features_and_daily_endpoint(self):
        dates = pd.date_range("2025-12-20", periods=800, freq="15min", tz="UTC")
        rows = []
        for asset, scale in (("BTC", 1.0), ("ETH", 2.0)):
            close = scale * (100 + np.arange(len(dates)) * 0.01)
            rows.append(pd.DataFrame({
                "asset": asset, "date": dates, "open": close,
                "high": close * 1.001, "low": close * 0.999,
                "close": close, "volume": 10 + np.arange(len(dates)) % 7,
            }))
        prepared = prepare_tft_data(
            pd.concat(rows), assets=("BTC", "ETH"),
            as_of=pd.Timestamp("2026-01-01 12:00Z"),
        )
        self.assertIsInstance(prepared, PreparedTFTData)
        self.assertEqual(set(HIST_EXOG) - set(prepared.model_frame), set())
        self.assertEqual(prepared.model_frame.ds.max().strftime("%H:%M"), "23:45")
        self.assertTrue(np.isfinite(prepared.model_frame[list(HIST_EXOG)]).all().all())

    def test_target_origins_crossing_synthetic_bars_are_excluded_for_all_assets(self):
        prepared = prepared_fixture_with_one_missing_target_bar()
        origins = eligible_daily_origins(prepared.context_frame, horizon=96)
        affected = pd.Timestamp("2026-01-02 00:00Z")
        self.assertNotIn(affected, origins)

    def test_empty_origins_keep_utc_timezone(self):
        context = pd.DataFrame(columns=("asset", "ds", "missing_bar"))
        origins = eligible_daily_origins(context, horizon=96)
        self.assertEqual(str(origins.tz), "UTC")

    def test_future_corruption_does_not_change_past_features(self):
        raw, cutoff, as_of = long_raw_fixture()
        assert_tft_feature_causality(
            raw, cutoff=cutoff, assets=("BTC", "ETH"), as_of=as_of,
        )
