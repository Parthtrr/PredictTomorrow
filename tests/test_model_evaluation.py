"""Check metric definitions and prevent target prices from leaking into context."""

import unittest
from unittest.mock import Mock

import pandas as pd

from model_train.kronos_evaluator import KronosEvaluator
from model_train.metrics import calculate_metrics


class PersistenceModel:
    """A test double that predicts the last observed close and records its inputs."""

    def __init__(self):
        """Keep the windows so tests can inspect them afterward."""
        self.calls = []

    def predict_close(self, history, target_date):
        """Return yesterday's observed close without seeing the target price."""
        self.calls.append((history.copy(), target_date))
        return float(history.iloc[-1]["close"])


def make_dataset():
    """Split five imaginary daily candles into three non-empty DataFrames."""
    rows = []
    for day, close in enumerate([100, 105, 110, 120, 130], start=1):
        rows.append({
            "date": pd.Timestamp(2010, 1, day), "ticker": "TEST",
            "open": close, "high": close + 2, "low": close - 2,
            "close": close, "volume": 1000, "day_change_pct": 1,
        })
    frame = pd.DataFrame(rows)
    dataset = Mock()
    dataset.load.return_value = (frame.iloc[:3], frame.iloc[3:4], frame.iloc[4:])
    return dataset


class MetricsTest(unittest.TestCase):
    """Verify that price error and direction error are independent measures."""

    def test_correct_direction_can_still_have_price_error(self):
        """Previous 100, actual 110, predicted 105 has a correct UP prediction."""
        predictions = pd.DataFrame([{"previous_close": 100, "actual_close": 110, "predicted_close": 105}])
        metrics = calculate_metrics(predictions)
        self.assertEqual(metrics["mae"], 5)
        self.assertAlmostEqual(metrics["mape_pct"], 5 / 110 * 100)
        self.assertEqual(metrics["directional_error_pct"], 0)

    def test_wrong_direction_and_flat_prices_are_counted(self):
        """One wrong prediction out of two is 50 percent directional error."""
        predictions = pd.DataFrame([
            {"previous_close": 100, "actual_close": 90, "predicted_close": 105},
            {"previous_close": 100, "actual_close": 100, "predicted_close": 100},
        ])
        metrics = calculate_metrics(predictions)
        self.assertEqual(metrics["mae"], 7.5)
        self.assertAlmostEqual(metrics["mape_pct"], (15 / 90 * 100) / 2)
        self.assertEqual(metrics["directional_error_pct"], 50)
        self.assertEqual(metrics["directional_error_count"], 1)

    def test_mae_uses_price_units_while_mape_uses_percentages(self):
        """Equal percentage errors at different prices have different point errors."""
        predictions = pd.DataFrame([
            {"previous_close": 90, "actual_close": 100, "predicted_close": 110},
            {"previous_close": 900, "actual_close": 1000, "predicted_close": 1100},
        ])
        metrics = calculate_metrics(predictions)
        self.assertEqual(metrics["mae"], 55)
        self.assertEqual(metrics["mape_pct"], 10)

    def test_empty_or_invalid_prices_are_rejected(self):
        """Do not silently report metrics for missing data or divide by zero."""
        with self.assertRaises(ValueError):
            calculate_metrics(pd.DataFrame())
        for actual, predicted in [(0, 100), (100, float("nan"))]:
            with self.subTest(actual=actual, predicted=predicted):
                frame = pd.DataFrame([{"previous_close": 100, "actual_close": actual, "predicted_close": predicted}])
                with self.assertRaises(ValueError):
                    calculate_metrics(frame)


class KronosEvaluatorTest(unittest.TestCase):
    """Check the evaluation flow without loading model weights or making network calls."""

    def test_all_splits_are_combined_and_only_past_rows_are_used(self):
        """Score a training row, a validation row, and a test row after warmup."""
        dataset = make_dataset()
        model = PersistenceModel()
        evaluator = KronosEvaluator(dataset, model, {"lookback": 2})
        summary, predictions = evaluator.evaluate("2010-01-01", "2011-12-31")
        dataset.load.assert_called_once_with("2010-01-01", "2011-12-31")
        self.assertEqual(summary["split_rows"], {"train": 3, "validation": 1, "test": 1})
        self.assertEqual(summary["prediction_count"], 3)
        self.assertEqual(predictions["predicted_close"].tolist(), [105, 110, 120])
        self.assertEqual(summary["directional_error_pct"], 100)
        for history, target_date in model.calls:
            self.assertEqual(len(history), 2)
            self.assertLess(history["date"].max(), target_date)

    def test_short_history_or_invalid_lookback_is_rejected(self):
        """The model must have enough past rows before any target is scored."""
        for lookback in [0, 513, True]:
            with self.subTest(lookback=lookback):
                with self.assertRaises(ValueError):
                    KronosEvaluator(make_dataset(), PersistenceModel(), {"lookback": lookback})
        evaluator = KronosEvaluator(make_dataset(), PersistenceModel(), {"lookback": 5})
        with self.assertRaises(ValueError):
            evaluator.evaluate("2010-01-01", "2011-12-31")

    def test_overlapping_split_dates_are_rejected(self):
        """Duplicate target dates must not be scored twice."""
        dataset = make_dataset()
        train, validation, test = dataset.load.return_value
        dataset.load.return_value = (train, train.iloc[-1:], test)
        evaluator = KronosEvaluator(dataset, PersistenceModel(), {"lookback": 2})
        with self.assertRaises(ValueError):
            evaluator.evaluate("2010-01-01", "2011-12-31")


if __name__ == "__main__":
    unittest.main()
