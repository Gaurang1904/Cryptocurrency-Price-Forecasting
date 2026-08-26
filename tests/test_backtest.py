import unittest

import numpy as np
import pandas as pd

from crypto.backtest import H, _self_check, purge_forward_labels, run_folds, score
from crypto.model import split_calibration


class BacktestSelfCheckTests(unittest.TestCase):
    def test_self_check_validates_constant_and_trending_prices(self):
        _self_check()


class LabelPurgingTests(unittest.TestCase):
    @staticmethod
    def _frame():
        rows = []
        calendars = {
            "BTC": pd.date_range("2020-09-01", periods=150, tz="UTC"),
            "ETH": pd.date_range("2020-09-01", periods=150, tz="UTC").delete([4, 9, 17]),
        }
        for asset, dates in calendars.items():
            frame = pd.DataFrame({
                "asset": asset, "date": dates, "close": 100.0,
                "feature": 1.0,
            })
            for h in range(1, H + 1):
                frame[f"y{h}"] = 0.0
                frame[f"label_end{h}"] = frame.date.shift(-h)
            rows.append(frame)
        return pd.concat(rows, ignore_index=True)

    def test_each_horizon_uses_actual_per_asset_endpoint(self):
        frame = self._frame()
        boundary = pd.Timestamp("2021-01-01 12:00", tz="UTC")

        for h in range(1, H + 1):
            with self.subTest(h=h):
                purged = purge_forward_labels(frame, boundary, horizons=[h])
                self.assertTrue((purged[f"label_end{h}"] <= boundary).all())
                crossing = frame[
                    frame[f"label_end{h}"].notna()
                    & (frame[f"label_end{h}"] > boundary)
                    & (frame.date < boundary)
                ]
                self.assertFalse(crossing.empty)
                self.assertTrue(set(crossing.index).isdisjoint(purged.index))
                exact_boundary = pd.Timestamp("2021-01-01", tz="UTC")
                equal = frame[frame[f"label_end{h}"] == exact_boundary]
                inclusive = purge_forward_labels(frame, exact_boundary, horizons=[h])
                self.assertFalse(equal.empty)
                self.assertTrue(set(equal.index).issubset(inclusive.index))

    def test_fold_and_calibration_splits_purge_all_seven_horizons(self):
        frame = self._frame()
        train, _test, fold_start = next(run_folds(frame, ["feature"]))
        for h in range(1, H + 1):
            self.assertTrue((train[f"label_end{h}"] <= fold_start).all())

        fit, cal = split_calibration(train)
        calibration_start = cal.date.min()
        for h in range(1, H + 1):
            self.assertTrue((fit[f"label_end{h}"] <= calibration_start).all())


class IntervalScoringTests(unittest.TestCase):
    def test_pinball_percent_normalizes_each_forecast_row_before_averaging(self):
        frame = pd.DataFrame({
            "model": ["m", "m"],
            "y": [100.0, 100.0],
            "last": [100.0, 1000.0],
            "q10": [110.0, 110.0],
            "q50": [110.0, 110.0],
            "q90": [110.0, 110.0],
        })

        # Per row, mean quantile loss is (9 + 5 + 1) / 3 = 5.
        expected = np.mean([5.0 / 100.0, 5.0 / 1000.0]) * 100
        self.assertAlmostEqual(score(frame).loc["m", "pinball_%"], expected)


if __name__ == "__main__":
    unittest.main()
