"""
factors/volatility.py
----------------------
Realized volatility (low-volatility anomaly): negative of trailing 60-day
annualized realized volatility of daily log returns. Signal direction:
negative in the raw-return sense (low realized vol -> higher expected
future return), which is why the raw value is negated here — this keeps
low-vol names at the TOP of the cross-sectional ranking, consistent with
the backtest engine's "long top quintile / short bottom quintile"
convention used by every other factor in this project. This sign choice is
not obvious from the bare formula, hence called out explicitly here.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from data.fetch_data import log_returns
from factors.base import Factor


class VolatilityFactor(Factor):
    name = "realized_vol_60d"
    lookback_days = config.VOLATILITY_LOOKBACK_DAYS
    rebalance_freq = "M"

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        ret = log_returns(prices)
        realized_vol = ret.rolling(config.VOLATILITY_LOOKBACK_DAYS).std() * np.sqrt(252)
        return -realized_vol


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data.fetch_data import fetch_prices

    prices = fetch_prices()
    factor = VolatilityFactor()
    raw = factor.compute(prices)

    nan_rate = factor.nan_rate(raw)
    print(f"NaN rate at day {factor.lookback_days - 1} (should be near 1.0):",
          nan_rate.iloc[factor.lookback_days - 1])
    print(f"NaN rate at day {factor.lookback_days + 1} (should be ~0):",
          nan_rate.iloc[factor.lookback_days + 1])

    print("\nDescribe non-NaN raw values (negative annualized vol):")
    print(raw.stack().describe())
