"""Read daily historical prices from Yahoo Finance."""

from datetime import date, timedelta
from pathlib import Path

import yfinance as yf


class YFinanceDao:
    """Keep Yahoo Finance access separate from the Elasticsearch DAO."""

    def get_daily_prices(self, ticker, start_date, end_date):
        """Download daily prices, including both requested dates."""
        # Keep yfinance's local cache inside this project.
        cache_path = Path(__file__).resolve().parents[1] / ".cache/yfinance"
        yf.set_tz_cache_location(str(cache_path))

        # Yahoo excludes the end date, while our application's range includes it.
        exclusive_end = date.fromisoformat(end_date) + timedelta(days=1)
        prices = yf.download(
            tickers=ticker,
            start=start_date,
            end=exclusive_end.isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            repair=False,
            keepna=True,
            ignore_tz=True,
            threads=False,
            progress=False,
            multi_level_index=False,
            timeout=30,
        )
        if prices is None or prices.empty:
            raise RuntimeError("Yahoo Finance returned no daily prices for " + ticker)

        # Preserve Yahoo's values; only put the sessions into chronological order.
        return prices.sort_index()
