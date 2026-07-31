"""
analysis/quintiles.py
----------------------
Quintile bucket assignment and quintile-return computation for the
monotonicity check: Q1 < Q2 < ... < Q5 should hold for a useful factor.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def assign_quintiles(factor_row: pd.Series, n_quintiles: int = None) -> pd.Series:
    """
    Bucket tickers 1..n_quintiles by factor value, for ONE date.

    NaN factor values stay NaN (unassigned) rather than being dropped, so the
    output aligns with factor_row's original index.
    """
    n_quintiles = n_quintiles or config.N_QUINTILES
    valid = factor_row.dropna()
    if len(valid) < n_quintiles:
        return pd.Series(np.nan, index=factor_row.index)
    q = pd.qcut(valid.rank(method="first"), n_quintiles, labels=False) + 1
    return q.reindex(factor_row.index)


def quintile_returns(
    factor_zscore: pd.DataFrame,
    forward_ret: pd.DataFrame,
    n_quintiles: int = None,
) -> pd.DataFrame:
    """
    Mean forward return per quintile bucket, per rebalance date.

    Returns
    -------
    pd.DataFrame, index=rebalance date, columns=[1, 2, ..., n_quintiles, "long_short"]
    where long_short = mean(Q_n) - mean(Q_1).
    """
    n_quintiles = n_quintiles or config.N_QUINTILES
    dates = factor_zscore.index.intersection(forward_ret.index)
    rows = {}
    for d in dates:
        buckets = assign_quintiles(factor_zscore.loc[d], n_quintiles)
        row = {
            q: forward_ret.loc[d, buckets[buckets == q].index].mean()
            for q in range(1, n_quintiles + 1)
        }
        row["long_short"] = row[n_quintiles] - row[1]
        rows[d] = row
    return pd.DataFrame(rows).T


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

    qret = quintile_returns(z, fwd)
    print("Mean return per quintile (monotonicity check, Q1->Q5):")
    print(qret[list(range(1, config.N_QUINTILES + 1))].mean())
    print("\nMean long_short:", qret["long_short"].mean())
    print("\nTail of quintile returns:")
    print(qret.tail())
