# PredictTomorrow

Download daily `^NSEI` history from Yahoo Finance using `yfinance` and store OHLCV
in local Elasticsearch index `nifty_daily_data`. Forecasting will be added later.

## Run

From `/Users/parthdubey/PersonalDev/PredictTomorrow`:

```bash
# Install dependencies in the existing virtual environment.
.venv/bin/python -m pip install -r requirements.txt

# Create the index if needed, download daily history, and save it.
.venv/bin/python main.py
```

Elasticsearch must run at `http://localhost:9200`. The current environment uses Python 3.13.

## Layers

- `main.py`: connects the components and prints the import result.
- `config.json` and `config.py`: shared settings and configuration loading.
- `services/price_download_service.py`: date-window rules and record preparation.
- `dao/yfinance_dao.py`: Yahoo Finance downloads.
- `dao/elasticsearch_client.py`: Elasticsearch connection.
- `dao/daily_price_dao.py`: index creation, bulk saves, and date-range reads.
- `mappings/nifty_daily_data.json`: explicit Elasticsearch field definitions.

The dependency direction is `main -> service -> DAOs -> data sources`.
The service receives the DAOs and does not create database connections.

## Data settings

`config.json` selects `yfinance`, ticker `^NSEI`, and the inclusive date range
`2010-01-01` through `today`. Today is evaluated on each run in `Asia/Kolkata`.
Before the configured regular-session close of `15:30`, the importer excludes
today's unfinished candle. This cutoff is configurable for a special session.
Weekends and exchange holidays have no generated rows; only Yahoo's available
daily records are stored.

The Yahoo request uses `auto_adjust=False` and `repair=False` to preserve the
provider's OHLCV values. Yahoo's exclusive end date is advanced by one day so
the application's range remains inclusive. Missing OHLC prices are stored as
JSON null, and the import prints the number of incomplete records. These raw
records stay in storage; model preparation must select complete candles later.
Invalid numeric prices or missing/fractional volume raise an error. Zero volume
is preserved when supplied. The program never invents prices or volume.
Yahoo's timezone/cookie cache stays in ignored `.cache/`.

## Elasticsearch records

Each document has `ticker` (keyword), `date` (date), `open`, `high`, `low`,
`close` (double), and `volume` (long). The record ID combines ticker and date,
for example `^NSEI|2010-01-04`. Rerunning the importer replaces existing dates
and inserts new dates without creating duplicates. Tickers are uppercased.

The mapping rejects unlisted fields. Future fields must be added explicitly;
editing the mapping file alone does not change an existing index. Index setup
preserves existing mappings and records. The local index has one shard and no replicas.

`save_prices()` streams records in configured batches and returns the number
saved. Failed batches raise an exception; already completed batches are not
rolled back. Stable IDs make retries safe. `get_prices(ticker, start_date,
end_date)` includes both boundaries, returns oldest first, and reads every page.
The service selects the download window; the storage DAO does not impose it.

## Read records for future model use

```python
from config import load_config
from dao.daily_price_dao import DailyPriceDao
from dao.elasticsearch_client import create_client

# Read the same configuration used by the downloader.
config = load_config()
settings = config["elasticsearch"]
client = create_client(settings)
try:
    dao = DailyPriceDao(client, settings["index_name"], settings["batch_size"])
    # Fetch a chronological list of daily records for the configured ticker.
    prices = dao.get_prices(config["data"]["ticker"], config["data"]["start_date"], config["data"]["end_date"])
finally:
    client.close()
```

## Checks

```bash
# Run offline download/conversion tests and configuration checks.
.venv/bin/python -m unittest discover -s tests -v

# Also test real Elasticsearch storage using disposable test indices.
RUN_ES_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
```

Live DAO tests use imaginary records only in `predict_tomorrow_dao_test_*`
indices and delete them after each check. They do not write to `nifty_daily_data`.

Provider reference: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
