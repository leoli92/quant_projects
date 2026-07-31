"""
analysis/ic.py
---------------
Information Coefficient (IC): cross-sectional Spearman rank correlation
between factor values and forward returns, computed independently at each
rebalance date.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from analysis.cross_sectional import periods_per_year


def compute_ic_series(
    factor_zscore: pd.DataFrame,
    forward_ret: pd.DataFrame,
    min_names: int = None,
) -> pd.Series:
    """
    Spearman rank IC between factor z-score and forward return, per date.

    Parameters
    ----------
    factor_zscore : index=date, columns=ticker — MUST already be restricted
                    to the same dates as forward_ret (e.g. via
                    `factor_zscore.loc[rebalance_dates]`); a mismatch here
                    silently produces all-NaN IC.
    forward_ret   : index=date, columns=ticker (from cross_sectional.forward_returns)
    min_names     : minimum non-NaN cross-sectional pairs required to compute
                    IC at a given date (defaults to config.MIN_NAMES_FOR_IC)

    Returns
    -------
    pd.Series indexed by date.
    """
    min_names = min_names if min_names is not None else config.MIN_NAMES_FOR_IC
    dates = factor_zscore.index.intersection(forward_ret.index)
    out = pd.Series(index=dates, dtype=float)
    for d in dates:
        x, y = factor_zscore.loc[d], forward_ret.loc[d]
        valid = x.notna() & y.notna()
        if valid.sum() < min_names:
            out.loc[d] = np.nan
            continue
        out.loc[d] = spearmanr(x[valid], y[valid])[0]
    return out


def rolling_ic_mean(ic_series: pd.Series, freq: str) -> pd.Series:
    """Rolling mean IC over one year of rebalance periods (12 for monthly,
    52 for weekly)."""
    return ic_series.rolling(periods_per_year(freq)).mean()


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from analysis.cross_sectional import forward_returns, get_rebalance_dates
    from data.fetch_data import fetch_prices
    from factors.momentum import MomentumFactor

    prices = fetch_prices()
    factor = MomentumFactor()

    rebal = get_rebalance_dates(prices.index, factor.rebalance_freq)
    z = factor.compute_zscore(prices).loc[rebal]
    fwd = forward_returns(prices, rebal)

    ic = compute_ic_series(z, fwd)
    print(f"IC series: {ic.notna().sum()} valid / {len(ic)} dates")
    print(f"Mean IC:  {ic.mean():.4f}")
    print(f"Std IC:   {ic.std():.4f}")
    print(f"Hit rate: {(ic.dropna() > 0).mean():.4f}")

    rolling = rolling_ic_mean(ic, factor.rebalance_freq)
    print("\nRolling 12-period IC mean (tail):")
    print(rolling.tail())
