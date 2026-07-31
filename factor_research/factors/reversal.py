"""
factors/reversal.py
--------------------
1-month short-term reversal: negative of the trailing 1-month cumulative
log return. Signal direction: negative (recent losers tend to bounce,
recent winners tend to pull back over the next short horizon).
Rebalanced weekly, since the effect decays fast.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from data.fetch_data import log_prices
from factors.base import Factor


class ReversalFactor(Factor):
    name = "reversal_1m"
    lookback_days = config.REVERSAL_LOOKBACK_DAYS
    rebalance_freq = "W"

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        lp = log_prices(prices)
        trailing_1m_return = lp - lp.shift(config.REVERSAL_LOOKBACK_DAYS)
        return -trailing_1m_return


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data.fetch_data import fetch_prices

    prices = fetch_prices()
    factor = ReversalFactor()
    raw = factor.compute(prices)

    nan_rate = factor.nan_rate(raw)
    print(f"NaN rate at day {factor.lookback_days - 1} (should be 1.0):",
          nan_rate.iloc[factor.lookback_days - 1])
    print(f"NaN rate at day {factor.lookback_days} (should be ~0):",
          nan_rate.iloc[factor.lookback_days])

    print("\nDescribe non-NaN raw values:")
    print(raw.stack().describe())
