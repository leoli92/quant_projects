"""
analysis/cross_sectional.py
----------------------------
Rebalance-date scheduling and forward-return labeling for cross-sectional
factor validation.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.fetch_data import log_prices

_PERIODS_PER_YEAR = {"M": 12, "W": 52}


def get_rebalance_dates(trading_days: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """
    Last trading day within each calendar period.

    Parameters
    ----------
    trading_days : DatetimeIndex of actual trading days (e.g. prices.index)
    freq         : "M" for month-end trading day, "W" for week-end trading day

    Returns
    -------
    DatetimeIndex, sorted, one date per period.
    """
    s = pd.Series(trading_days, index=trading_days)
    last_per_period = s.groupby(trading_days.to_period(freq)).max()
    return pd.DatetimeIndex(sorted(last_per_period.values))


def periods_per_year(freq: str) -> int:
    return _PERIODS_PER_YEAR[freq]


def forward_returns(prices: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Log return from rebalance_dates[i] (close) to rebalance_dates[i+1] (close)
    — the return earned by a position established at date i and held to i+1.

    This is the ONE place `.shift(-1)` is legitimate in this codebase: it
    labels each rebalance date with a return that happens strictly AFTER it,
    for IC/backtest validation only. It is never fed into Factor.compute().

    Returns
    -------
    pd.DataFrame, index=rebalance_dates, columns=tickers. Last row is NaN
    (no next rebalance date to compute a forward return to).
    """
    lp = log_prices(prices).loc[rebalance_dates]
    return lp.shift(-1) - lp


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data.fetch_data import fetch_prices

    prices = fetch_prices()

    monthly = get_rebalance_dates(prices.index, "M")
    weekly = get_rebalance_dates(prices.index, "W")
    print(f"Monthly rebalance dates: {len(monthly)} (expect ~{12 * 12})")
    print(monthly[:5], "...", monthly[-3:])
    print(f"\nWeekly rebalance dates: {len(weekly)} (expect ~{52 * 12})")
    print(weekly[:5], "...", weekly[-3:])

    fwd = forward_returns(prices, monthly)
    print(f"\nForward returns shape: {fwd.shape}")
    print(fwd.iloc[-3:, :4])
