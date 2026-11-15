"""Read a date range from Elasticsearch and split it into three DataFrames."""

import math

import pandas as pd


class StockDataset:
    """Prepare one ticker's train, validation, and test data through the DAO."""

    def __init__(self, price_dao, ticker, settings):
        """Receive the storage DAO, ticker, and dataset settings from config.json."""
        self.price_dao = price_dao
        self.ticker = ticker
        self.train_ratio = settings["train_ratio"]
        self.validation_ratio = settings["validation_ratio"]
        self.test_ratio = settings["test_ratio"]
        self.drop_missing_rows = settings["drop_missing_rows"]

        # All three splits must have a positive share and together use all rows.
        ratios = [self.train_ratio, self.validation_ratio, self.test_ratio]
        for ratio in ratios:
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise ValueError("Dataset ratios must be numbers between 0 and 1.")
            if not math.isfinite(ratio) or not 0 < ratio < 1:
                raise ValueError("Dataset ratios must be numbers between 0 and 1.")
        if not math.isclose(sum(ratios), 1.0):
            raise ValueError("Dataset ratios must add up to 1.")
        if not isinstance(self.drop_missing_rows, bool):
            raise ValueError("drop_missing_rows must be true or false.")

    def load(self, start_date, end_date):
        """Return (train_data, validation_data, test_data) for an inclusive range."""
        records = self.price_dao.get_prices(self.ticker, start_date, end_date)
        if not records:
            raise ValueError("No stored prices found in the requested date range.")

        data = pd.DataFrame(records)
        data["date"] = pd.to_datetime(data["date"])
        # Keep the oldest rows for training and the newest rows for testing.
        data = data.sort_values("date").reset_index(drop=True)

        if self.drop_missing_rows:
            # Filter the returned frames only; never fill gaps or modify Elasticsearch.
            numeric_columns = ["open", "high", "low", "close", "volume", "day_change_pct"]
            data = data.dropna(subset=numeric_columns).reset_index(drop=True)

        train_count = int(len(data) * self.train_ratio)
        validation_count = int(len(data) * self.validation_ratio)
        # Give any rounding remainder to the test split so no usable rows are lost.
        test_count = len(data) - train_count - validation_count
        if min(train_count, validation_count, test_count) < 1:
            raise ValueError("Not enough usable rows to create three non-empty datasets.")

        validation_end = train_count + validation_count
        # Independent copies let callers work on one split without changing another.
        train_data = data.iloc[:train_count].copy().reset_index(drop=True)
        validation_data = data.iloc[train_count:validation_end].copy().reset_index(drop=True)
        test_data = data.iloc[validation_end:].copy().reset_index(drop=True)
        return train_data, validation_data, test_data
