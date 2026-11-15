"""Encode a small, explicit set of candle shapes for future model features."""

import math


def _prepare_candle(price):
    """Describe a complete candle without changing the original OHLC prices."""
    candle = {}
    for field in ("open", "high", "low", "close"):
        if price[field] is None:
            return None
        value = float(price[field])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Candle prices must be finite and positive.")
        candle[field] = value
    if candle["low"] > min(candle["open"], candle["close"]) or candle["high"] < max(candle["open"], candle["close"]):
        raise ValueError("Candle high/low must contain its open and close.")

    candle["body"] = abs(candle["close"] - candle["open"])
    candle["range"] = candle["high"] - candle["low"]
    candle["body_low"] = min(candle["open"], candle["close"])
    candle["body_high"] = max(candle["open"], candle["close"])
    candle["direction"] = 0
    if candle["close"] > candle["open"]:
        candle["direction"] = 1
    elif candle["close"] < candle["open"]:
        candle["direction"] = -1
    return candle


def _score_one(candle):
    """Marubozu: the real body covers at least 90% of the candle's range."""
    if candle["range"] > 0 and candle["body"] >= 0.90 * candle["range"]:
        return candle["direction"]
    return 0


def _score_two(first, second):
    """Recognize opposite-color Engulfing or small-body Harami patterns."""
    if first["direction"] == 0 or second["direction"] != -first["direction"]:
        return 0

    # Engulfing uses real bodies, not the full high-low ranges.
    engulfing = (
        second["body_low"] <= first["body_low"]
        and second["body_high"] >= first["body_high"]
        and second["body"] > first["body"]
    )
    # Harami requires the smaller body strictly inside the first body's bounds.
    harami = (
        second["body_low"] > first["body_low"]
        and second["body_high"] < first["body_high"]
        and second["body"] <= 0.50 * first["body"]
    )
    if engulfing or harami:
        return 2 * second["direction"]
    return 0


def _strong_candle(candle, direction):
    """A long body is >=60% of the range and closes within 20% of its extreme."""
    if candle["direction"] != direction or candle["body"] < 0.60 * candle["range"]:
        return False
    if direction == 1:
        return candle["high"] - candle["close"] <= 0.20 * candle["range"]
    return candle["close"] - candle["low"] <= 0.20 * candle["range"]


def _score_three(first, middle, last):
    """Recognize Morning/Evening Star and Three White Soldiers/Black Crows."""
    midpoint = (first["open"] + first["close"]) / 2
    small_middle = middle["body"] <= 0.30 * first["body"]

    # Stars have a small middle body separated by body gaps on both sides.
    if (
        _strong_candle(first, -1) and _strong_candle(last, 1) and small_middle
        and middle["body_high"] < first["close"]
        and last["open"] > middle["body_high"] and last["close"] > midpoint
    ):
        return 3
    if (
        _strong_candle(first, 1) and _strong_candle(last, -1) and small_middle
        and middle["body_low"] > first["close"]
        and last["open"] < middle["body_low"] and last["close"] < midpoint
    ):
        return -3

    # Soldiers/crows need strong candles and opens inside the preceding real body.
    opens_inside = (
        first["body_low"] <= middle["open"] <= first["body_high"]
        and middle["body_low"] <= last["open"] <= middle["body_high"]
    )
    if opens_inside:
        if (
            _strong_candle(first, 1) and _strong_candle(middle, 1) and _strong_candle(last, 1)
            and first["close"] < middle["close"] < last["close"]
        ):
            return 3
        if (
            _strong_candle(first, -1) and _strong_candle(middle, -1) and _strong_candle(last, -1)
            and first["close"] > middle["close"] > last["close"]
        ):
            return -3
    return 0


def add_candle_patterns(prices):
    """Add scores in date order per ticker; missing OHLC resets pattern history."""
    histories = {}
    for price in prices:
        ticker = price["ticker"].strip().upper()
        history = histories.setdefault(ticker, [])
        for length in (1, 2, 3):
            price["candle_pattern_" + str(length)] = None

        candle = _prepare_candle(price)
        if candle is None:
            # Never join candles across a row whose actual prices are unknown.
            history.clear()
            continue
        history.append(candle)
        if len(history) > 3:
            history.pop(0)

        price["candle_pattern_1"] = _score_one(candle)
        if len(history) >= 2:
            price["candle_pattern_2"] = _score_two(history[-2], history[-1])
        if len(history) == 3:
            price["candle_pattern_3"] = _score_three(history[0], history[1], history[2])
    return prices
