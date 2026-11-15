"""Verify named pattern geometry, history boundaries, and causal feature values."""

import copy
import unittest

from services.candle_patterns import add_candle_patterns


def make_candle(open_price, close_price, high=None, low=None, ticker="TEST"):
    """Build an imaginary candle; omitted wicks produce a full-body candle."""
    if high is None:
        high = max(open_price, close_price)
    if low is None:
        low = min(open_price, close_price)
    return {"ticker": ticker, "open": open_price, "close": close_price, "high": high, "low": low}


class CandlePatternsTest(unittest.TestCase):
    """Use explicit candles rather than a dependency on a trading library."""

    def test_single_candle_marubozu_and_neutral_cases(self):
        """Only a sufficiently full body gets a directional one-candle score."""
        cases = [
            (make_candle(100, 110), 1),
            (make_candle(110, 100), -1),
            (make_candle(101, 110, high=110, low=100), 1),
            (make_candle(100, 100, high=101, low=99), 0),
            (make_candle(100, 101, high=110, low=90), 0),
            (make_candle(100, 100), 0),
        ]
        for candle, expected in cases:
            with self.subTest(candle=candle):
                add_candle_patterns([candle])
                self.assertEqual(candle["candle_pattern_1"], expected)
                self.assertIsNone(candle["candle_pattern_2"])
                self.assertIsNone(candle["candle_pattern_3"])

    def test_two_candle_engulfing_and_harami(self):
        """Bullish/bearish direction is encoded as plus/minus two."""
        cases = [
            ([make_candle(110, 105), make_candle(104, 112)], 2),
            ([make_candle(105, 110), make_candle(112, 104)], -2),
            ([make_candle(110, 100), make_candle(102, 105)], 2),
            ([make_candle(100, 110), make_candle(108, 105)], -2),
            ([make_candle(100, 105), make_candle(104, 110)], 0),
            ([make_candle(100, 100), make_candle(100, 110)], 0),
        ]
        for candles, expected in cases:
            with self.subTest(expected=expected, candles=candles):
                add_candle_patterns(candles)
                self.assertEqual(candles[-1]["candle_pattern_2"], expected)

    def test_three_candle_stars_soldiers_and_crows(self):
        """Recognize the four supported three-candle shapes on their last bar."""
        cases = [
            ([make_candle(110, 100, 111, 99), make_candle(96, 97, 98, 95), make_candle(98, 107, 108, 97)], 3),
            ([make_candle(100, 110, 111, 99), make_candle(114, 113, 115, 112), make_candle(112, 103, 113, 102)], -3),
            ([make_candle(100, 110, 111, 99), make_candle(106, 116, 117, 105), make_candle(112, 122, 123, 111)], 3),
            ([make_candle(122, 112, 123, 111), make_candle(116, 106, 117, 105), make_candle(110, 100, 111, 99)], -3),
        ]
        for candles, expected in cases:
            with self.subTest(expected=expected, candles=candles):
                add_candle_patterns(candles)
                self.assertIsNone(candles[0]["candle_pattern_3"])
                self.assertIsNone(candles[1]["candle_pattern_3"])
                self.assertEqual(candles[-1]["candle_pattern_3"], expected)

    def test_same_color_candles_alone_are_not_a_named_pattern(self):
        """Three green candles opening outside previous bodies are not soldiers."""
        candles = [make_candle(100, 110), make_candle(112, 122), make_candle(124, 134)]
        add_candle_patterns(candles)
        self.assertEqual(candles[-1]["candle_pattern_3"], 0)

    def test_missing_ohlc_resets_the_history(self):
        """Do not invent multi-candle patterns across missing source records."""
        missing = {"ticker": "TEST", "open": None, "high": None, "low": None, "close": None}
        candles = [make_candle(100, 110), missing, make_candle(110, 120), make_candle(115, 125)]
        add_candle_patterns(candles)
        for length in (1, 2, 3):
            self.assertIsNone(missing["candle_pattern_" + str(length)])
        self.assertIsNone(candles[2]["candle_pattern_2"])
        self.assertIsNone(candles[3]["candle_pattern_3"])

    def test_tickers_do_not_share_pattern_history(self):
        """Interleaved symbols must still use their own previous candles."""
        candles = [
            make_candle(110, 105, ticker="A"), make_candle(100, 105, ticker="B"),
            make_candle(104, 112, ticker="A"), make_candle(106, 99, ticker="B"),
        ]
        add_candle_patterns(candles)
        self.assertEqual(candles[2]["candle_pattern_2"], 2)
        self.assertEqual(candles[3]["candle_pattern_2"], -2)
        self.assertIsNone(candles[2]["candle_pattern_3"])

    def test_future_candles_cannot_change_existing_scores(self):
        """Feature construction must never use future prices for earlier rows."""
        history = [make_candle(110, 105), make_candle(104, 112), make_candle(108, 115)]
        expected = add_candle_patterns(copy.deepcopy(history))
        extended = copy.deepcopy(history) + [make_candle(1000, 10)]
        actual = add_candle_patterns(extended)
        self.assertEqual(actual[:len(expected)], expected)
        self.assertEqual(add_candle_patterns([]), [])


if __name__ == "__main__":
    unittest.main()
