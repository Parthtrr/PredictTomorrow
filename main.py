"""Download the configured daily history and store it in Elasticsearch."""

from config import load_config
from dao.daily_price_dao import DailyPriceDao
from dao.elasticsearch_client import create_client
from dao.yfinance_dao import YFinanceDao
from services.price_download_service import PriceDownloadService


def main():
    """Create the index, download Yahoo Finance prices, and save the history."""
    config = load_config()
    settings = config["elasticsearch"]
    client = create_client(settings)
    try:
        dao = DailyPriceDao(client, settings["index_name"], settings["batch_size"])
        created = dao.create_index()
        if created:
            print("Created index:", settings["index_name"])
        else:
            print("Index already exists:", settings["index_name"])

        print("Configured history:", config["data"]["start_date"], "to", config["data"]["end_date"])

        # The service coordinates the two DAOs; main only wires them together.
        service = PriceDownloadService(YFinanceDao(), dao)
        result = service.download_history(config["data"])
        print("Saved", result["saved_count"], "daily records for", result["ticker"])
        print("Stored history:", result["first_date"], "to", result["last_date"])
        print("Records with missing prices:", result["incomplete_count"])
    finally:
        # Close the connection even when an Elasticsearch request fails.
        client.close()


if __name__ == "__main__":
    main()
