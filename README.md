# PredictTomorrow

Download daily `^NSEI` history from Yahoo Finance using `yfinance` and store OHLCV
in local Elasticsearch index `nifty_daily_data`. Evaluate pretrained Kronos on
stored history using the `model_train` layer, without fitting the model.

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
- `services/price_features.py`: daily percentage change from the previous close.
- `services/candle_patterns.py`: one-, two-, and three-candle geometry scores.
- `build_candle_patterns.py`: backfill scores using prices already in Elasticsearch.
- `datasets/stock_dataset.py`: date-range reads and chronological DataFrame splits.
- `model_train/`: pretrained Kronos loading, historical evaluation, and metrics.
- `evaluate_model.py`: evaluation entry point and CSV/JSON report writing.
- `dao/yfinance_dao.py`: Yahoo Finance downloads.
- `dao/elasticsearch_client.py`: Elasticsearch connection.
- `dao/daily_price_dao.py`: index creation, bulk saves, and date-range reads.
- `mappings/nifty_daily_data.json`: explicit Elasticsearch field definitions.

The dependency direction is `main -> service -> DAOs -> data sources`.
The service receives the DAOs and does not create database connections.
The dataset read path is `caller -> datasets -> DAO -> Elasticsearch`.

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
`close` (double), `volume` (long), `day_change_pct` (double), and three
`candle_pattern_1`, `candle_pattern_2`, `candle_pattern_3` fields (byte).
Daily change is `(close - previous_close) / previous_close * 100`, so a move
from 100 to 105 is stored as `5.0`. The first row and rows missing the current
or immediately previous close store null. A missing row is not forward-filled.
The service calculates this field for each download, without changing OHLCV.
The record ID combines ticker and date,
for example `^NSEI|2010-01-04`. Rerunning the importer replaces existing dates
and inserts new dates without creating duplicates. Tickers are uppercased.

The mapping rejects unlisted fields. Future fields must be added explicitly.
Index setup applies the mapping file to an existing index using the mapping API,
which adds fields without deleting existing records. Existing field types cannot
be changed this way. Adding a mapping does not calculate old records; their new
values must be backfilled separately. The local index has one shard and no replicas.

`save_prices()` streams records in configured batches and returns the number
saved. Failed batches raise an exception; already completed batches are not
rolled back. Stable IDs make retries safe. `get_prices(ticker, start_date,
end_date)` includes both boundaries, returns oldest first, and reads every page.
The service selects the download window; the storage DAO does not impose it.

## Candle-pattern features

The daily index stores these shape-based features for later transfer learning:

| Field | Bullish | Bearish | Supported shapes |
|---|---:|---:|---|
| `candle_pattern_1` | 1 | -1 | Marubozu |
| `candle_pattern_2` | 2 | -2 | Engulfing, Harami |
| `candle_pattern_3` | 3 | -3 | Morning/Evening Star, Three White Soldiers/Black Crows |

Zero means a complete window matched none of the supported shapes. Null means
missing OHLC or insufficient history. A doji/flat candle is neutral on its own.
Scores are attached to the final candle, using that ticker's current and earlier
records only. Missing OHLC resets the window, so patterns never bridge unknown
prices. These are a documented starter set, not an exhaustive pattern library.

The geometric thresholds are explicit: Marubozu body >=90% of full range;
Engulfing bodies have opposite colors with the second strictly larger and
containing the first; Harami has an opposite-color body strictly inside the first
and <=50% its size. A strong candle has body >=60% of range and closes within
20% of its directional extreme. Stars need strong first/last candles, a middle
body <=30% of the first body, body gaps on both sides, and a final close beyond
the first body's midpoint. Soldiers/crows need three strong candles, opens
inside preceding real bodies, and strictly rising/falling closes. These scores
describe candle geometry; they do not add a preceding-trend or future-confirmation
filter. Magnitude encodes pattern length, not confidence or predictive accuracy.

```bash
# Add the mapping fields and backfill the configured ticker from stored history.
.venv/bin/python build_candle_patterns.py
```

Future downloads calculate the same fields before saving. Dataset DataFrames
include them automatically; the current missing-row filter still considers only
OHLCV and day_change_pct. Pretrained Kronos inputs and saved evaluation results
are unchanged. Using the new fields in a model is a later transfer-learning step.

Pattern references: [Zerodha Varsity](https://zerodha.com/varsity/chapter/getting-started-candlesticks/),
[Three White Soldiers/Crows](https://www.fidelity.com/bin-public/060_www_fidelity_com/documents/Top10CandlesWork_620095.pdf).

## Load train, validation, and test DataFrames

`StockDataset.load(start_date, end_date)` fetches one ticker's stored records
within the inclusive date range and returns three separate pandas DataFrames.
The ticker comes from `data.ticker`; split settings are in `datasets` in the config.

Defaults are 70% training, 15% validation, and 15% testing. Training uses the
oldest rows and testing uses the newest. Rows are never shuffled. Training and
validation counts are rounded down; testing receives the remainder. All splits
must have at least one usable row, otherwise a clear ValueError is raised.

By default, rows missing OHLCV or `day_change_pct` are excluded from the returned
frames. Set `drop_missing_rows` to false to retain them. Stored Elasticsearch
records are unchanged. Each frame keeps every stored column, has a datetime
`date` column, and starts its row index at zero. This step does not scale values,
fill gaps, generate labels, or train a model.

```python
from config import load_config
from dao.daily_price_dao import DailyPriceDao
from dao.elasticsearch_client import create_client
from datasets.stock_dataset import StockDataset

# Read the same configuration used by the downloader.
config = load_config()
settings = config["elasticsearch"]
client = create_client(settings)
try:
    dao = DailyPriceDao(client, settings["index_name"], settings["batch_size"])
    dataset = StockDataset(dao, config["data"]["ticker"], config["datasets"])
    # Choose the inclusive date range and receive three independent DataFrames.
    train_data, validation_data, test_data = dataset.load("2010-01-01", config["data"]["end_date"])
    print(len(train_data), len(validation_data), len(test_data))
finally:
    client.close()
```

## Evaluate pretrained Kronos

Install the requirements and download the official source once:

```bash
# Install dependencies and keep the official model source in an ignored folder.
.venv/bin/python -m pip install -r requirements.txt
git clone https://github.com/shiyu-coder/Kronos.git vendor/Kronos
git -C vendor/Kronos checkout 67b630e67f6a18c9e9be918d9b4337c960db1e9a

# Evaluate the configured 2010-01-01 through 2011-12-31 period.
.venv/bin/python evaluate_model.py

# Override the two model dates when evaluating another period.
.venv/bin/python evaluate_model.py --model-start-date 2010-01-01 --model-end-date 2011-12-31
```

The source is already downloaded in the current workspace; skip cloning if
`vendor/Kronos` exists. Model and tokenizer weights download on the first run
into ignored `.cache/huggingface`. Exact source/weight revisions, seed, device,
lookback, and dates are in the `model_train` config section.

`KronosEvaluator.evaluate(model_start_date, model_end_date)` calls the datasets
layer, combines all three splits, and evaluates chronologically. The first
60 usable rows provide initial context and are not scored. Every later row is
predicted from the previous 60 rows only; its actual prices are withheld until
scoring. Each step uses newly observed historical prices rather than feeding
earlier predictions back into context. No model weights are updated.

The unmodified pretrained interface receives OHLCV and dates. Its own predictor
derives the optional turnover input and normalizes each context window.
`day_change_pct` is retained in storage but is not fed into this architecture.
Targets are the next usable observations after dataset filtering; gaps are not
filled, and consecutive observations may not be consecutive calendar days.

- Closing-price MAE: mean of `abs(actual_close - predicted_close)`, in original
  price units (index points for `^NSEI`).
- Closing-price MAPE: mean of `abs(actual_close - predicted_close) / actual_close * 100`.
- Directional error: percentage whose predicted UP/DOWN/FLAT direction differs
  from the actual direction, relative to the same previous observed close.
- MAE uses price units; MAPE and directional error use percentages. Smaller
  values mean lower errors.

For example, previous close 100, actual 110, predicted 105 has MAE 5 price units,
MAPE 4.545%, and directional error 0% for that prediction: both directions are UP.
Reports include scored dates and warmup counts. Predictions are saved as CSV
and aggregate metrics as JSON under ignored `outputs/`.

Pretraining overlap with 2010-2011 is unknown. This is a historical pretrained
evaluation, not a verified out-of-sample test of the original pretraining data.

## Checks

```bash
# Run offline download/conversion tests and configuration checks.
.venv/bin/python -m unittest discover -s tests -v

# Also test real Elasticsearch storage using disposable test indices.
RUN_ES_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
```

Live tests use imaginary records only in `predict_tomorrow_dao_test_*` and
`predict_tomorrow_dataset_test_*` indices, and delete them after each check.
They do not write to `nifty_daily_data`.

Provider reference: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
