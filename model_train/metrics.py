"""Closing-price MAE, MAPE, and directional error, calculated separately."""

import math


def get_direction(close, previous_close):
    """Return the direction relative to the same previous observed close."""
    if close > previous_close:
        return "UP"
    if close < previous_close:
        return "DOWN"
    return "FLAT"


def calculate_metrics(predictions):
    """Return MAE in price units and MAPE/directional error in percentages."""
    if predictions.empty:
        raise ValueError("No predictions are available for evaluation.")

    total_absolute_error = 0.0
    total_percentage_error = 0.0
    direction_errors = 0
    for row in predictions.itertuples(index=False):
        actual = float(row.actual_close)
        predicted = float(row.predicted_close)
        previous = float(row.previous_close)
        if not math.isfinite(actual) or not math.isfinite(predicted) or not math.isfinite(previous):
            raise ValueError("Evaluation prices must be finite numbers.")
        if actual <= 0 or previous <= 0:
            raise ValueError("Actual and previous closing prices must be positive.")

        # MAE keeps the original price units; MAPE expresses the error as a percent.
        absolute_error = abs(actual - predicted)
        total_absolute_error += absolute_error
        total_percentage_error += absolute_error / actual * 100
        actual_direction = get_direction(actual, previous)
        predicted_direction = get_direction(predicted, previous)
        if actual_direction != predicted_direction:
            direction_errors += 1

    count = len(predictions)
    return {
        "prediction_count": count,
        "mae": total_absolute_error / count,
        "mape_pct": total_percentage_error / count,
        "directional_error_pct": direction_errors / count * 100,
        "directional_error_count": direction_errors,
    }
