"""Check date boundaries and Yahoo-to-Elasticsearch data conversion offline."""

import unittest
from datetime import datetime
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pandas as pd

from dao.yfinance_dao import YFinanceDao
from services.price_download_service import PriceDownloadService


def make_history(volume=0):
    """Return one imaginary Nifty candle, including Yahoo's possible zero volume."""
    return pd.DataFrame(
        {"Open": [100.0], "High": [110.0], "Low": [90.0],
         "Close": [105.0], "Volume": [volume]},
        index=pd.to_datetime(["2026-01-02"]),
    )


def make_settings():
    """Use a short date range for deterministic service tests."""
    return {
        "source": "yfinance", "ticker": "^NSEI",
        "start_date": "2010-01-01", "end_date": "2026-01-02",
        "timezone": "Asia/Kolkata", "market_close_time": "15:30",
    }


class YFinanceDaoTest(unittest.TestCase):
    """Check provider requests without making any Yahoo Finance calls."""

    @patch("dao.yfinance_dao.yf.set_tz_cache_location")
    @patch("dao.yfinance_dao.yf.download")
    def test_end_date_is_inclusive_and_ohlcv_is_unadjusted(self, download, cache):
        """An inclusive January 2 range must send January 3 to Yahoo."""
        download.return_value = make_history()
        history = YFinanceDao().get_daily_prices("^NSEI", "2010-01-01", "2026-01-02")
        self.assertEqual(download.call_args.kwargs["end"], "2026-01-03")
        self.assertEqual(download.call_args.kwargs["start"], "2010-01-01")
        self.assertFalse(download.call_args.kwargs["auto_adjust"])
        self.assertFalse(download.call_args.kwargs["repair"])
        self.assertEqual(history.iloc[0]["Close"], 105.0)

    @patch("dao.yfinance_dao.yf.set_tz_cache_location")
    @patch("dao.yfinance_dao.yf.download")
    def test_empty_download_fails_instead_of_reporting_success(self, download, cache):
        """A provider failure must not look like a successful empty import."""
        download.return_value = pd.DataFrame()
        with self.assertRaises(RuntimeError):
            YFinanceDao().get_daily_prices("^NSEI", "2010-01-01", "2026-01-02")


class PriceDownloadServiceTest(unittest.TestCase):
    """Check the business rules independently of either real data source."""

    def test_cutoff_excludes_an_unfinished_today_and_preserves_zero_volume(self):
        """Only after the configured close may today's daily candle be imported."""
        for hour, expected_end in [(14, "2026-01-01"), (16, "2026-01-02")]:
            with self.subTest(hour=hour):
                downloader = Mock()
                downloader.get_daily_prices.return_value = make_history()
                store = Mock()
                store.save_prices.return_value = 1
                service = PriceDownloadService(downloader, store)
                now = datetime(2026, 1, 2, hour, tzinfo=ZoneInfo("Asia/Kolkata"))
                if hour == 14:
                    # Mimic a provider returning only the requested earlier session.
                    downloader.get_daily_prices.return_value.index = pd.to_datetime(["2026-01-01"])
                result = service.download_history(make_settings(), now=now)
                downloader.get_daily_prices.assert_called_once_with("^NSEI", "2010-01-01", expected_end)
                record = store.save_prices.call_args.args[0][0]
                self.assertEqual(record["date"], expected_end)
                self.assertEqual(record["volume"], 0)
                self.assertIsInstance(record["volume"], int)
                self.assertEqual(result["saved_count"], 1)

    def test_daily_change_is_calculated_before_saving(self):
        """The importer should save 5 percent when the close rises from 100 to 105."""
        history = pd.concat([make_history(), make_history()])
        history.index = pd.to_datetime(["2026-01-01", "2026-01-02"])
        history.loc["2026-01-01", "Close"] = 100.0
        downloader = Mock()
        downloader.get_daily_prices.return_value = history
        store = Mock()
        service = PriceDownloadService(downloader, store)
        now = datetime(2026, 1, 2, 16, tzinfo=ZoneInfo("Asia/Kolkata"))
        service.download_history(make_settings(), now=now)
        records = store.save_prices.call_args.args[0]
        self.assertIsNone(records[0]["day_change_pct"])
        self.assertAlmostEqual(records[1]["day_change_pct"], 5.0)
        self.assertEqual(records[0]["candle_pattern_1"], 0)
        self.assertEqual(records[1]["candle_pattern_2"], 0)
        self.assertIsNone(records[1]["candle_pattern_3"])

    def test_missing_prices_become_null_and_are_counted(self):
        """Preserve Yahoo's empty price fields and report incomplete records."""
        history = make_history()
        for field in ("Open", "High", "Low", "Close"):
            history[field] = float("nan")
        downloader = Mock()
        downloader.get_daily_prices.return_value = history
        store = Mock()
        store.save_prices.return_value = 1
        service = PriceDownloadService(downloader, store)
        now = datetime(2026, 1, 2, 16, tzinfo=ZoneInfo("Asia/Kolkata"))
        result = service.download_history(make_settings(), now=now)
        record = store.save_prices.call_args.args[0][0]
        for field in ("open", "high", "low", "close"):
            self.assertIsNone(record[field])
        for length in (1, 2, 3):
            self.assertIsNone(record["candle_pattern_" + str(length)])
        self.assertEqual(result["incomplete_count"], 1)

    def test_invalid_volume_is_not_rounded_or_saved(self):
        """Missing and fractional volume must fail before the DAO saves records."""
        for volume in [float("nan"), 1.5, -1]:
            with self.subTest(volume=volume):
                downloader = Mock()
                downloader.get_daily_prices.return_value = make_history(volume)
                store = Mock()
                service = PriceDownloadService(downloader, store)
                now = datetime(2026, 1, 2, 16, tzinfo=ZoneInfo("Asia/Kolkata"))
                with self.assertRaises(ValueError):
                    service.download_history(make_settings(), now=now)
                store.save_prices.assert_not_called()


if __name__ == "__main__":
    unittest.main()
