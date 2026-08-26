import unittest
import warnings

import numpy as np
import pandas as pd

from crypto.features import H, check_causal, feature_cols, feature_groups, make_features


def market_frame(n=500):
    dates = pd.date_range("2024-01-01", periods=n, tz="UTC")
    rows = []
    for offset, asset in enumerate(("BTC", "ETH")):
        values = np.arange(n, dtype=float) + 100.0 + offset * 50
        rows.append(pd.DataFrame({
            "asset": asset, "date": dates,
            "open": values, "high": values + 2.0,
            "low": values - 1.0, "close": values + 1.0,
            "volume": np.arange(n) + 1000.0 + offset * 100,
        }))
    return pd.concat(rows, ignore_index=True)


class RegimeFeatureTests(unittest.TestCase):
    def test_regime_features_are_present_and_causal(self):
        df = market_frame()
        feat = make_features(df)
        for column in ["vol_regime", "drawdown_63d", "volume_z21"]:
            self.assertIn(column, feat.columns)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            check_causal(df, ["vol_regime", "drawdown_63d", "volume_z21"])

    def test_label_endpoints_follow_each_assets_actual_date_sequence(self):
        frame = market_frame(20)
        missing = frame[(frame.asset == "ETH") & (frame.date == frame.date.unique()[8])].index
        feat = make_features(frame.drop(index=missing))
        for _asset, group in feat.groupby("asset", sort=False):
            for h in range(1, H + 1):
                pd.testing.assert_series_equal(
                    group[f"label_end{h}"].reset_index(drop=True),
                    group.date.shift(-h).reset_index(drop=True),
                    check_names=False,
                )

    def test_causality_guard_checks_requested_labels_and_all_assets(self):

        with self.assertRaisesRegex(AssertionError, "look-ahead"):
            check_causal(market_frame(), ["y1"])

    def test_candidate_features_are_opt_in_model_columns(self):
        feat = make_features(market_frame(100))
        candidates = {"vol_regime", "drawdown_63d", "volume_z21"}

        self.assertTrue(candidates.isdisjoint(feature_cols(feat)))
        self.assertTrue(
            candidates.issubset(feature_cols(feat, include_candidates=True))
        )

    def test_feature_groups_cover_each_feature_once(self):
        columns = ["ret_lag1", "vol_21d", "range", "btc_ret", "fund_7d"]
        groups = feature_groups(columns)
        flattened = [column for members in groups.values() for column in members]
        self.assertCountEqual(flattened, columns)

    def test_feature_groups_cover_actual_features_without_targets(self):
        columns = feature_cols(make_features(market_frame(100)))
        flattened = [column for members in feature_groups(columns).values() for column in members]
        self.assertCountEqual(flattened, columns)
        for h in range(1, H + 1):
            self.assertNotIn(f"y{h}", columns)
            self.assertNotIn(f"label_end{h}", columns)


if __name__ == "__main__":
    unittest.main()
