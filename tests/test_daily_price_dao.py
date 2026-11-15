"""Check real storage behavior using disposable indices, never stock data."""

import os
import unittest
from datetime import date, datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from config import load_config
from dao.daily_price_dao import DailyPriceDao
from dao.elasticsearch_client import create_client


def make_price(day="2026-01-02", ticker="TEST.NS"):
    """Return an imaginary daily candle for database tests."""
    return {
        "ticker": ticker,
        "date": day,
        "open": 100,
        "high": 110,
        "low": 90,
        "close": 105,
        "volume": 1000,
    }


class ConfigTest(unittest.TestCase):
    """Check that the date window stays current without editing the config."""

    def test_today_is_resolved_in_the_configured_timezone(self):
        """Today's date should be evaluated each time configuration is loaded."""
        config = load_config()
        expected = datetime.now(ZoneInfo(config["data"]["timezone"])).date().isoformat()
        self.assertEqual(config["data"]["start_date"], "2010-01-01")
        self.assertEqual(config["data"]["end_date"], expected)


@unittest.skipUnless(os.environ.get("RUN_ES_TESTS") == "1", "Set RUN_ES_TESTS=1 for local Elasticsearch tests.")
class DailyPriceDaoTest(unittest.TestCase):
    """Verify saves and paginated reads against the local Elasticsearch server."""

    def setUp(self):
        """Use a new, clearly named test index for each check."""
        self.client = create_client(load_config()["elasticsearch"])
        self.addCleanup(self.client.close)
        self.index_name = "predict_tomorrow_dao_test_" + uuid4().hex
        self.dao = DailyPriceDao(self.client, self.index_name, batch_size=2)
        self.dao.create_index()
        self.addCleanup(self.client.indices.delete, index=self.index_name)

    def test_create_index_can_be_repeated(self):
        """Running setup again must preserve existing records."""
        self.dao.save_prices([make_price()], refresh=True)
        self.assertFalse(self.dao.create_index())
        self.assertEqual(self.client.count(index=self.index_name)["count"], 1)

    def test_repeated_save_updates_the_same_ticker_and_date(self):
        """Ticker case and whitespace must not cause duplicate documents."""
        self.dao.save_prices([make_price(ticker=" test.ns ")], refresh=True)
        price = make_price()
        price["close"] = 108
        self.dao.save_prices([price], refresh=True)
        records = self.dao.get_prices("test.ns", "2026-01-02", "2026-01-02")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["close"], 108)
        self.assertEqual(self.client.count(index=self.index_name)["count"], 1)

    def test_date_range_reads_every_page_in_order(self):
        """Read multiple pages, filter the ticker, and include both date bounds."""
        prices = []
        for offset in range(8):
            day = (date(2026, 1, 1) + timedelta(days=offset)).isoformat()
            prices.append(make_price(day))
        prices.reverse()
        prices.append(make_price(ticker="OTHER.NS"))
        self.assertEqual(self.dao.save_prices(prices, refresh=True), 9)

        records = self.dao.get_prices("TEST.NS", "2026-01-02", "2026-01-07")
        dates = []
        for record in records:
            dates.append(record["date"])
        self.assertEqual(dates, ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07"])

    def test_invalid_prices_are_rejected(self):
        """Reject bad numbers, impossible candles, invalid dates, and empty tickers."""
        invalid_values = [
            ("close", float("nan")),
            ("open", -1),
            ("high", 99),
            ("low", 106),
            ("volume", 1.5),
            ("volume", -1),
            ("volume", True),
            ("date", "2026-02-30"),
            ("ticker", " "),
        ]
        for field, value in invalid_values:
            with self.subTest(field=field, value=value):
                price = make_price()
                price[field] = value
                with self.assertRaises(ValueError):
                    self.dao.save_prices([price])
        self.assertEqual(self.client.count(index=self.index_name)["count"], 0)

    def test_missing_source_prices_are_stored_as_null(self):
        """Keep incomplete source rows without inventing replacement prices."""
        price = make_price()
        for field in ("open", "high", "low", "close"):
            price[field] = None
        price["volume"] = 0
        self.dao.save_prices([price], refresh=True)
        records = self.dao.get_prices("TEST.NS", "2026-01-02", "2026-01-02")
        self.assertEqual(records, [price])

    def test_missing_ticker_and_empty_batch(self):
        """Empty inputs and an unknown ticker should return empty results."""
        self.assertEqual(self.dao.save_prices([]), 0)
        self.assertEqual(self.dao.get_prices("MISSING.NS", "2010-01-01", "2026-01-01"), [])

    def test_reversed_date_range_is_rejected(self):
        """Do not silently accept a start date after the end date."""
        with self.assertRaises(ValueError):
            self.dao.get_prices("TEST.NS", "2026-01-03", "2026-01-02")


if __name__ == "__main__":
    unittest.main()
