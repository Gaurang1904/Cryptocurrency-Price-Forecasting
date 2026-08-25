import unittest

from crypto.backtest import _self_check


class BacktestSelfCheckTests(unittest.TestCase):
    def test_self_check_validates_constant_and_trending_prices(self):
        _self_check()


if __name__ == "__main__":
    unittest.main()
