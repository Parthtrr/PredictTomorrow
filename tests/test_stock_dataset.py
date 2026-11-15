"""Check chronological splitting, missing data, and real Elasticsearch reads."""

import os
import unittest
from datetime import date, timedelta
from unittest.mock import Mock
from uuid import uuid4

import pandas as pd

from config import load_config
from dao.daily_price_dao import DailyPriceDao
from dao.elasticsearch_client import create_client
from datasets.stock_dataset import StockDataset


def make_records(count=20):
    """Create imaginary daily records for deterministic dataset tests."""
    records = []
    for offset in range(count):
        close = 100 + offset
        records.append({
            "ticker": "TEST.NS",
            "date": (date(2026, 1, 1) + timedelta(days=offset)).isoformat(),
            "open": close, "high": close + 2, "low": close - 2,
            "close": close, "volume": 1000, "day_change_pct": 1.0,
        })
    return records


def make_settings():
    """Use the default 70/15/15 split with incomplete rows excluded."""
    return {
        "train_ratio": 0.70, "validation_ratio": 0.15, "test_ratio": 0.15,
        "drop_missing_rows": True,
    }


class StockDatasetTest(unittest.TestCase):
    """Verify dataset rules without needing a running database."""

    def test_chronological_splits_have_no_overlap_or_lost_rows(self):
        """Twenty unordered rows should become ordered splits of 14, 3, and 3."""
        records = make_records()
        records.reverse()
        dao = Mock()
        dao.get_prices.return_value = records
        dataset = StockDataset(dao, "TEST.NS", make_settings())
        train, validation, test = dataset.load("2026-01-01", "2026-01-20")

        dao.get_prices.assert_called_once_with("TEST.NS", "2026-01-01", "2026-01-20")
        self.assertEqual((len(train), len(validation), len(test)), (14, 3, 3))
        combined = pd.concat([train, validation, test], ignore_index=True)
        self.assertTrue(combined["date"].is_monotonic_increasing)
        self.assertEqual(combined["date"].nunique(), 20)
        self.assertLess(train["date"].max(), validation["date"].min())
        self.assertLess(validation["date"].max(), test["date"].min())
        self.assertEqual(list(train.columns), list(records[0]))

    def test_missing_values_are_excluded_but_zero_volume_is_kept(self):
        """A null percentage or close is unusable; zero volume is a valid value."""
        records = make_records()
        records[0]["day_change_pct"] = None
        records[1]["close"] = None
        records[2]["volume"] = 0
        dao = Mock()
        dao.get_prices.return_value = records
        frames = StockDataset(dao, "TEST.NS", make_settings()).load("2026-01-01", "2026-01-20")
        combined = pd.concat(frames, ignore_index=True)
        self.assertEqual(len(combined), 18)
        self.assertFalse(combined.isna().any().any())
        self.assertEqual(combined["volume"].min(), 0)
        self.assertIsNone(records[0]["day_change_pct"])

    def test_missing_values_can_be_preserved(self):
        """Turning off filtering should preserve raw records for inspection."""
        records = make_records()
        records[0]["day_change_pct"] = None
        settings = make_settings()
        settings["drop_missing_rows"] = False
        dao = Mock()
        dao.get_prices.return_value = records
        frames = StockDataset(dao, "TEST.NS", settings).load("2026-01-01", "2026-01-20")
        combined = pd.concat(frames, ignore_index=True)
        self.assertEqual(len(combined), 20)
        self.assertEqual(combined["day_change_pct"].isna().sum(), 1)

    def test_split_ratios_are_configurable(self):
        """A 60/20/20 split should return 12, 4, and 4 rows."""
        settings = make_settings()
        settings.update({"train_ratio": 0.60, "validation_ratio": 0.20, "test_ratio": 0.20})
        dao = Mock()
        dao.get_prices.return_value = make_records()
        train, validation, test = StockDataset(dao, "TEST.NS", settings).load("2026-01-01", "2026-01-20")
        self.assertEqual((len(train), len(validation), len(test)), (12, 4, 4))

    def test_invalid_settings_are_rejected(self):
        """Bad ratios must fail before any Elasticsearch read occurs."""
        invalid_settings = [
            ("train_ratio", 0.50), ("train_ratio", -0.1), ("train_ratio", 1),
            ("train_ratio", float("nan")), ("train_ratio", "0.7"),
            ("train_ratio", True), ("drop_missing_rows", "false"),
        ]
        for field, value in invalid_settings:
            with self.subTest(field=field, value=value):
                settings = make_settings()
                settings[field] = value
                dao = Mock()
                with self.assertRaises(ValueError):
                    StockDataset(dao, "TEST.NS", settings)
                dao.get_prices.assert_not_called()

    def test_empty_and_too_small_ranges_raise_clear_errors(self):
        """Do not silently return an empty validation or test split."""
        for records in [[], make_records(3)]:
            with self.subTest(row_count=len(records)):
                dao = Mock()
                dao.get_prices.return_value = records
                with self.assertRaises(ValueError):
                    StockDataset(dao, "TEST.NS", make_settings()).load("2026-01-01", "2026-01-20")


@unittest.skipUnless(os.environ.get("RUN_ES_TESTS") == "1", "Set RUN_ES_TESTS=1 for local Elasticsearch tests.")
class StockDatasetIntegrationTest(unittest.TestCase):
    """Check the complete datasets -> DAO -> Elasticsearch read path."""

    def test_requested_dates_and_ticker_are_respected(self):
        """Only the seven requested dates should enter the three DataFrames."""
        client = create_client(load_config()["elasticsearch"])
        self.addCleanup(client.close)
        index_name = "predict_tomorrow_dataset_test_" + uuid4().hex
        dao = DailyPriceDao(client, index_name, batch_size=2)
        dao.create_index()
        self.addCleanup(client.indices.delete, index=index_name)
        records = make_records(10)
        other = make_records(1)[0]
        other["ticker"] = "OTHER.NS"
        records.append(other)
        dao.save_prices(records, refresh=True)

        dataset = StockDataset(dao, "TEST.NS", make_settings())
        train, validation, test = dataset.load("2026-01-03", "2026-01-09")
        self.assertEqual((len(train), len(validation), len(test)), (4, 1, 2))
        combined = pd.concat([train, validation, test], ignore_index=True)
        self.assertEqual(combined["date"].min(), pd.Timestamp("2026-01-03"))
        self.assertEqual(combined["date"].max(), pd.Timestamp("2026-01-09"))
        self.assertEqual(combined["ticker"].unique().tolist(), ["TEST.NS"])
        self.assertEqual(client.count(index=index_name)["count"], 11)


if __name__ == "__main__":
    unittest.main()
