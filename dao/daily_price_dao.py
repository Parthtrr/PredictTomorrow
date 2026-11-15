"""Store one OHLCV document for each ticker and trading date."""

import json
import math
from datetime import date
from pathlib import Path

from elasticsearch.helpers import bulk, scan


def normalize_ticker(ticker):
    """Keep ticker names consistent when saving and searching."""
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string.")
    return ticker.strip().upper()


class DailyPriceDao:
    """Handle Elasticsearch operations without downloading or training logic."""

    def __init__(self, client, index_name, batch_size=500):
        """Receive a shared connection and the configured index name."""
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero.")
        self.client = client
        self.index_name = index_name
        self.batch_size = batch_size

    def create_index(self):
        """Create the OHLCV index once; leave an existing index unchanged."""
        if self.client.indices.exists(index=self.index_name):
            return False

        mapping_path = Path(__file__).resolve().parents[1] / "mappings/nifty_daily_data.json"
        with mapping_path.open(encoding="utf-8") as mapping_file:
            definition = json.load(mapping_file)

        self.client.indices.create(
            index=self.index_name,
            settings=definition["settings"],
            mappings=definition["mappings"],
        )
        return True

    def save_prices(self, prices, refresh=False):
        """Save batches and return their successful document count; failures raise."""
        # Wait for search visibility only when the caller needs an immediate read.
        refresh_option = False
        if refresh:
            refresh_option = "wait_for"

        saved_count, _ = bulk(
            self.client,
            self._create_actions(prices),
            chunk_size=self.batch_size,
            stats_only=True,
            max_retries=3,
            refresh=refresh_option,
        )
        return saved_count

    def _create_actions(self, prices):
        """Prepare one document at a time so large downloads can be streamed."""
        for price in prices:
            document = self._prepare_document(price)
            # Reusing this ID replaces a day's prices instead of creating duplicates.
            document_id = document["ticker"] + "|" + document["date"]
            yield {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": document_id,
                "_source": document,
            }

    def _prepare_document(self, price):
        """Reject invalid numbers while preserving missing source prices as null."""
        document = {
            "ticker": normalize_ticker(price["ticker"]),
            "date": date.fromisoformat(price["date"]).isoformat(),
        }
        has_missing_price = False
        for field in ("open", "high", "low", "close"):
            value = price[field]
            # A raw source row may have no price; JSON null preserves that absence.
            if value is None:
                document[field] = None
                has_missing_price = True
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(field + " must be a number.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(field + " must be finite and greater than zero.")
            document[field] = float(value)

        # Volume must remain a whole number; never silently round downloaded data.
        volume = price["volume"]
        if isinstance(volume, bool) or not isinstance(volume, int) or volume < 0:
            raise ValueError("volume must be a non-negative integer.")
        document["volume"] = volume

        # Compare candle prices only when all four values were supplied.
        if not has_missing_price:
            if document["high"] < max(document["open"], document["close"], document["low"]):
                raise ValueError("high must be at least open, close, and low.")
            if document["low"] > min(document["open"], document["close"]):
                raise ValueError("low must be at most open and close.")
        return document

    def get_prices(self, ticker, start_date, end_date):
        """Read every matching daily record, oldest first, including both dates."""
        start_date = date.fromisoformat(start_date).isoformat()
        end_date = date.fromisoformat(end_date).isoformat()
        if start_date > end_date:
            raise ValueError("start_date must be on or before end_date.")

        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"ticker": normalize_ticker(ticker)}},
                        {"range": {"date": {"gte": start_date, "lte": end_date}}},
                    ]
                }
            },
            "sort": [{"date": "asc"}],
        }
        # Scroll through every page instead of silently stopping at 10,000 records.
        records = scan(
            self.client,
            index=self.index_name,
            query=query,
            size=self.batch_size,
            preserve_order=True,
        )
        prices = []
        for record in records:
            prices.append(record["_source"])
        return prices
