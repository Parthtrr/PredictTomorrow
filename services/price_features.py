"""Calculate derived fields without changing the source OHLCV values."""


def add_day_change_pct(prices):
    """Add close-to-close change to rows ordered oldest first for each ticker."""
    previous_closes = {}
    for price in prices:
        ticker = price["ticker"]
        previous_close = previous_closes.get(ticker)
        close = price["close"]

        # The first row and rows missing either close have no known daily change.
        price["day_change_pct"] = None
        if close is not None and previous_close is not None and previous_close > 0:
            price["day_change_pct"] = (close - previous_close) / previous_close * 100

        # Reset on a missing close so a later row cannot silently bridge the gap.
        previous_closes[ticker] = close
    return prices
