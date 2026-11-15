"""Backfill candle-pattern fields using history already stored in Elasticsearch."""

from config import load_config
from dao.daily_price_dao import DailyPriceDao
from dao.elasticsearch_client import create_client
from services.candle_patterns import add_candle_patterns


def main():
    """Build the configured ticker's pattern features without downloading prices."""
    config = load_config()
    settings = config["elasticsearch"]
    data = config["data"]
    client = create_client(settings)
    try:
        dao = DailyPriceDao(client, settings["index_name"], settings["batch_size"])
        prices = dao.get_prices(data["ticker"], data["start_date"], data["end_date"])
        if not prices:
            raise ValueError("No stored prices found; download daily history first.")

        # Add fields to the live mapping and save with the existing ticker/date IDs.
        add_candle_patterns(prices)
        dao.create_index()
        saved_count = dao.save_prices(prices, refresh=True)
        print("Updated", saved_count, "records for", data["ticker"])
        for length in (1, 2, 3):
            field = "candle_pattern_" + str(length)
            counts = {}
            for price in prices:
                score = price[field]
                counts[score] = counts.get(score, 0) + 1
            print(field, counts)
    finally:
        client.close()


if __name__ == "__main__":
    main()
