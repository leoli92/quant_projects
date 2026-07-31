"""
factors/momentum.py
--------------------
12-1 month cross-sectional momentum: cumulative log return from t-252 to
t-21 trading days, skipping the most recent month to avoid the short-term
reversal effect. Signal direction: positive (higher past return -> higher
expected future return).
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from data.fetch_data import log_prices
from factors.base import Factor


class MomentumFactor(Factor):
    name = "momentum_12_1"
    lookback_days = config.MOMENTUM_LOOKBACK_DAYS + config.MOMENTUM_SKIP_DAYS
    rebalance_freq = "M"

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        lp = log_prices(prices)
        # Equivalent to returns.shift(21).rolling(252).sum(): a 252-trading-day
        # cumulative log return, measured over the window ending 21 days ago
        # (i.e. skipping the most recent month). Causal: only uses data
        # through t - 21.
        far  = config.MOMENTUM_SKIP_DAYS + config.MOMENTUM_LOOKBACK_DAYS
        near = config.MOMENTUM_SKIP_DAYS
        return lp.shift(near) - lp.shift(far)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data.fetch_data import fetch_prices

    prices = fetch_prices()
    factor = MomentumFactor()
    raw = factor.compute(prices)

    nan_rate = factor.nan_rate(raw)
    print("NaN rate, first 5 dates:")
    print(nan_rate.head())
    print(f"\nNaN rate at trading day {factor.lookback_days} (should be near 0):")
    print(nan_rate.iloc[factor.lookback_days])
    print(f"\nNaN rate at trading day {factor.lookback_days - 1} (should be ~1.0):")
    print(nan_rate.iloc[factor.lookback_days - 1])

    print("\nDescribe non-NaN raw values (log-return units over ~11mo window):")
    print(raw.stack().describe())
