"""Download daily history and store it through the Elasticsearch DAO."""

import math
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from services.price_features import add_day_change_pct
from services.candle_patterns import add_candle_patterns


class PriceDownloadService:
    """Connect the Yahoo Finance reader to the daily-price store."""

    def __init__(self, download_dao, price_dao):
        """Receive the two DAOs instead of creating database connections here."""
        self.download_dao = download_dao
        self.price_dao = price_dao

    def download_history(self, settings, now=None):
        """Download the configured window and return a small save summary."""
        if settings["source"] != "yfinance":
            raise ValueError("Only the yfinance source is supported.")

        # The optional clock lets tests check behavior before and after market close.
        timezone = ZoneInfo(settings["timezone"])
        if now is None:
            now = datetime.now(timezone)
        else:
            now = now.astimezone(timezone)

        last_completed_date = now.date()
        if now.time() < time.fromisoformat(settings["market_close_time"]):
            last_completed_date -= timedelta(days=1)

        start_date = date.fromisoformat(settings["start_date"])
        end_date = min(date.fromisoformat(settings["end_date"]), last_completed_date)
        if start_date > end_date:
            raise ValueError("The configured window has no completed trading dates yet.")

        ticker = settings["ticker"]
        history = self.download_dao.get_daily_prices(
            ticker, start_date.isoformat(), end_date.isoformat()
        )
        prices = []
        incomplete_count = 0
        for timestamp, row in history.iterrows():
            trading_date = timestamp.date()
            if trading_date < start_date or trading_date > end_date:
                continue

            # Convert pandas values into plain Python values for Elasticsearch.
            # Preserve zero volume; reject missing or fractional volume, never invent it.
            volume = float(row["Volume"])
            if not math.isfinite(volume) or volume < 0 or not volume.is_integer():
                raise ValueError("Invalid volume on " + trading_date.isoformat())
            price = {
                "ticker": ticker,
                "date": trading_date.isoformat(),
                "volume": int(volume),
            }
            has_missing_price = False
            for field in ("Open", "High", "Low", "Close"):
                # JSON cannot store NaN; null records the missing source value.
                if pd.isna(row[field]):
                    price[field.lower()] = None
                    has_missing_price = True
                else:
                    price[field.lower()] = float(row[field])
            if has_missing_price:
                incomplete_count += 1
            prices.append(price)

        if not prices:
            raise RuntimeError("No daily prices were returned inside the configured window.")

        # Calculate daily change after the rows are in chronological order.
        add_day_change_pct(prices)
        # Store candle-pattern features now; the pretrained model still uses OHLCV.
        add_candle_patterns(prices)

        # Stable document IDs make a full-history rerun safe from duplicates.
        saved_count = self.price_dao.save_prices(prices, refresh=True)
        return {
            "ticker": ticker,
            "saved_count": saved_count,
            "incomplete_count": incomplete_count,
            "first_date": prices[0]["date"],
            "last_date": prices[-1]["date"],
        }
