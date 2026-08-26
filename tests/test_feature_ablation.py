import unittest

from crypto.features import feature_cols, make_features
from experiments.feature_ablation import ablation_candidates
from tests.test_features import market_frame


class FeatureAblationTests(unittest.TestCase):
    def test_legacy_candidate_excludes_only_new_regime_features(self):
        columns = [
            "ret_lag1", "vol_21d", "vol_regime", "drawdown_63d",
            "volume_z21", "btc_ret", "fund_7d",
        ]

        candidates = ablation_candidates(columns)
        self.assertIn("legacy", candidates)
        self.assertEqual(
            candidates["legacy"],
            ["ret_lag1", "vol_21d", "btc_ret", "fund_7d"],
        )

    def test_ablation_explicitly_includes_candidate_features(self):
        columns = feature_cols(
            make_features(market_frame(100)), include_candidates=True
        )

        self.assertTrue(
            {"vol_regime", "drawdown_63d", "volume_z21"}.issubset(
                ablation_candidates(columns)["all"]
            )
        )

    def test_ablation_candidates_remove_only_named_group(self):
        columns = ["ret_lag1", "vol_21d", "volume_z21", "btc_ret", "fund_7d"]
        candidates = ablation_candidates(columns)
        self.assertEqual(
            candidates["without_volume"],
            ["ret_lag1", "vol_21d", "btc_ret", "fund_7d"],
        )


if __name__ == "__main__":
    unittest.main()
