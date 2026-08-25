import unittest
import warnings

import numpy as np
import pandas as pd

from crypto.features import check_causal, feature_cols, feature_groups, make_features


class RegimeFeatureTests(unittest.TestCase):
    def test_regime_features_are_present_and_causal(self):
        n = 500
        df = pd.DataFrame({
            "asset": "BTC", "date": pd.date_range("2024-01-01", periods=n, tz="UTC"),
            "open": np.arange(n) + 100.0, "high": np.arange(n) + 102.0,
            "low": np.arange(n) + 99.0, "close": np.arange(n) + 101.0,
            "volume": np.arange(n) + 1000.0,
        })
        feat = make_features(df)
        for column in ["vol_regime", "drawdown_63d", "volume_z21"]:
            self.assertIn(column, feat.columns)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            check_causal(df, ["vol_regime", "drawdown_63d", "volume_z21"])

    def test_feature_groups_cover_each_feature_once(self):
        columns = ["ret_lag1", "vol_21d", "range", "btc_ret", "fund_7d"]
        groups = feature_groups(columns)
        flattened = [column for members in groups.values() for column in members]
        self.assertCountEqual(flattened, columns)

    def test_feature_groups_cover_actual_features_without_targets(self):
        n = 100
        df = pd.DataFrame({
            "asset": "BTC", "date": pd.date_range("2024-01-01", periods=n, tz="UTC"),
            "open": np.arange(n) + 100.0, "high": np.arange(n) + 102.0,
            "low": np.arange(n) + 99.0, "close": np.arange(n) + 101.0,
            "volume": np.arange(n) + 1000.0,
        })
        columns = feature_cols(make_features(df))
        flattened = [column for members in feature_groups(columns).values() for column in members]
        self.assertCountEqual(flattened, columns)
        self.assertNotIn("y1", columns)


if __name__ == "__main__":
    unittest.main()