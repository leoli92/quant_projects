"""
analysis/ir.py
---------------
Information Ratio (IR) and summary statistics for an IC time series.
"""

import pandas as pd


def information_ratio(ic_series: pd.Series) -> float:
    """Mean IC / Std IC. NaN-safe."""
    ic = ic_series.dropna()
    if len(ic) < 2 or ic.std() == 0:
        return float("nan")
    return float(ic.mean() / ic.std())


def ic_summary(ic_series: pd.Series) -> dict:
    """
    Summary statistics for an IC series.

    Returns
    -------
    dict with keys: ic_mean, ic_std, ir, hit_rate, n_periods, nan_rate
    """
    ic = ic_series.dropna()
    return {
        "ic_mean": float(ic.mean()) if len(ic) else float("nan"),
        "ic_std": float(ic.std()) if len(ic) else float("nan"),
        "ir": information_ratio(ic_series),
        "hit_rate": float((ic > 0).mean()) if len(ic) else float("nan"),
        "n_periods": int(len(ic)),
        "nan_rate": float(ic_series.isna().mean()) if len(ic_series) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from analysis.cross_sectional import forward_returns, get_rebalance_dates
    from analysis.ic import compute_ic_series
    from data.fetch_data import fetch_prices
    from factors.momentum import MomentumFactor

    prices = fetch_prices()
    factor = MomentumFactor()
    rebal = get_rebalance_dates(prices.index, factor.rebalance_freq)
    z = factor.compute_zscore(prices).loc[rebal]
    fwd = forward_returns(prices, rebal)
    ic = compute_ic_series(z, fwd)

    print("IR:", information_ratio(ic))
    print("Summary:", ic_summary(ic))
