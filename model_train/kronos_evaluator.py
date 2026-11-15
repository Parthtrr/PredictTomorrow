"""Combine every dataset split and evaluate one-step forecasts chronologically."""

import pandas as pd

from model_train.metrics import calculate_metrics


class KronosEvaluator:
    """Evaluate a pretrained model using only earlier rows for each prediction."""

    def __init__(self, dataset, model, settings):
        """Receive the dataset layer and model instead of creating dependencies."""
        self.dataset = dataset
        self.model = model
        self.lookback = settings["lookback"]
        if isinstance(self.lookback, bool) or not isinstance(self.lookback, int) or not 1 <= self.lookback <= 512:
            raise ValueError("lookback must be an integer from 1 to 512.")

    def evaluate(self, model_start_date, model_end_date):
        """Return (summary, predictions) without fitting or changing model weights."""
        train, validation, test = self.dataset.load(model_start_date, model_end_date)
        # Every split contributes to this historical evaluation, as requested.
        data = pd.concat([train, validation, test], ignore_index=True)
        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values("date").reset_index(drop=True)
        if data["date"].duplicated().any():
            raise ValueError("Evaluation data must not contain overlapping dates.")
        if data[["open", "high", "low", "close", "volume"]].isna().any().any():
            raise ValueError("Evaluation needs complete OHLCV rows; enable dataset missing-row filtering.")
        if len(data) <= self.lookback:
            raise ValueError("Evaluation needs more rows than the configured lookback.")

        predictions = []
        total = len(data) - self.lookback
        print("Combined rows:", len(data), "| Context rows:", self.lookback, "| Predictions:", total, flush=True)
        for position in range(self.lookback, len(data)):
            # Stop before the target row: its actual prices never enter the model.
            history = data.iloc[position - self.lookback:position].copy()
            target = data.iloc[position]
            predicted_close = self.model.predict_close(history, target["date"])
            previous = history.iloc[-1]
            predictions.append({
                "previous_date": previous["date"],
                "date": target["date"],
                "previous_close": float(previous["close"]),
                "actual_close": float(target["close"]),
                "predicted_close": predicted_close,
            })
            if len(predictions) % 25 == 0 or len(predictions) == total:
                print("Evaluated", len(predictions), "of", total, flush=True)

        predictions = pd.DataFrame(predictions)
        summary = calculate_metrics(predictions)
        summary.update({
            "model_start_date": model_start_date,
            "model_end_date": model_end_date,
            "combined_rows": len(data),
            "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
            "warmup_rows": self.lookback,
            "evaluated_start_date": predictions["date"].min().date().isoformat(),
            "evaluated_end_date": predictions["date"].max().date().isoformat(),
        })
        return summary, predictions
