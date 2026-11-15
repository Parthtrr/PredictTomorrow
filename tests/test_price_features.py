"""Check daily percentage changes, missing prices, and ticker boundaries."""

import unittest

from services.price_features import add_day_change_pct


class PriceFeaturesTest(unittest.TestCase):
    """Use small closing-price examples to verify the financial calculation."""

    def test_gains_losses_and_unchanged_prices(self):
        """A 100-to-105 move is 5 percent, not 0.05 or a five-rupee feature."""
        prices = []
        for close in [100, 105, 99.75, 99.75]:
            prices.append({"ticker": "TEST", "close": close})
        add_day_change_pct(prices)
        self.assertIsNone(prices[0]["day_change_pct"])
        self.assertAlmostEqual(prices[1]["day_change_pct"], 5.0)
        self.assertAlmostEqual(prices[2]["day_change_pct"], -5.0)
        self.assertAlmostEqual(prices[3]["day_change_pct"], 0.0)

    def test_missing_prices_do_not_create_a_return_across_the_gap(self):
        """A missing previous close must remain unknown instead of being filled."""
        prices = []
        for close in [100, None, 110, 121]:
            prices.append({"ticker": "TEST", "close": close})
        add_day_change_pct(prices)
        for price in prices[:3]:
            self.assertIsNone(price["day_change_pct"])
        self.assertAlmostEqual(prices[3]["day_change_pct"], 10.0)
        self.assertEqual(add_day_change_pct([]), [])

    def test_each_ticker_uses_its_own_previous_close(self):
        """A different stock's closing price must never become the denominator."""
        prices = [
            {"ticker": "A", "close": 100}, {"ticker": "B", "close": 1000},
            {"ticker": "A", "close": 110}, {"ticker": "B", "close": 900},
        ]
        add_day_change_pct(prices)
        self.assertIsNone(prices[0]["day_change_pct"])
        self.assertIsNone(prices[1]["day_change_pct"])
        self.assertAlmostEqual(prices[2]["day_change_pct"], 10.0)
        self.assertAlmostEqual(prices[3]["day_change_pct"], -10.0)


if __name__ == "__main__":
    unittest.main()
