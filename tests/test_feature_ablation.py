import unittest

from experiments.feature_ablation import ablation_candidates


class FeatureAblationTests(unittest.TestCase):
    def test_ablation_candidates_remove_only_named_group(self):
        columns = ["ret_lag1", "vol_21d", "volume_z21", "btc_ret", "fund_7d"]
        candidates = ablation_candidates(columns)
        self.assertEqual(
            candidates["without_volume"],
            ["ret_lag1", "vol_21d", "btc_ret", "fund_7d"],
        )


if __name__ == "__main__":
    unittest.main()
