"""
backtest/engine.py
-------------------
Long/short quintile portfolio simulation with transaction costs and a
1-day execution lag.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from analysis.cross_sectional import get_rebalance_dates
from analysis.quintiles import assign_quintiles
from backtest.metrics import compute_factor_metrics
from data.fetch_data import log_returns
from factors.base import Factor


def build_weight_panel(
    factor_zscore: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    n_quintiles: int = None,
) -> pd.DataFrame:
    """
    Dollar-neutral target weights at each rebalance date: long the top
    quintile equal-weighted (weights sum to +1.0), short the bottom
    quintile equal-weighted (weights sum to -1.0). Gross exposure = 2.0.

    Returns
    -------
    pd.DataFrame, index=rebalance_dates, columns=factor_zscore.columns.
    """
    n_quintiles = n_quintiles or config.N_QUINTILES
    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=factor_zscore.columns)
    for d in rebalance_dates:
        buckets = assign_quintiles(factor_zscore.loc[d], n_quintiles)
        top = buckets[buckets == n_quintiles].index
        bottom = buckets[buckets == 1].index
        if len(top):
            weights.loc[d, top] = 1.0 / len(top)
        if len(bottom):
            weights.loc[d, bottom] = -1.0 / len(bottom)
    return weights


def run_factor_backtest(
    prices: pd.DataFrame,
    factor: Factor,
    rebalance_dates: pd.DatetimeIndex = None,
    cost_bps: float = None,
    execution_lag: int = None,
    n_quintiles: int = None,
) -> dict:
    """
    Simulate a long-top-quintile / short-bottom-quintile portfolio.

    Steps (lookahead is foreclosed at step 5: weights decided using
    information through rebalance date t are not applied to the portfolio
    until t + execution_lag trading days later):

      1. factor_zscore = factor.compute_zscore(prices)          [causal]
      2. rebalance_dates defaults to get_rebalance_dates(prices.index, factor.rebalance_freq)
      3. weights_at_rebal = build_weight_panel(...)
      4. weights_daily = weights_at_rebal reindexed to all trading days, ffilled
      5. weights_lagged = weights_daily.shift(execution_lag)
      6. daily_ret = log_returns(prices)
      7. gross_return = (weights_lagged * daily_ret).sum(axis=1)
      8. turnover_series = weights_lagged.diff().abs().sum(axis=1)
      9. net_return = gross_return - turnover_series * cost_bps / 10_000
      10. equity_curve = INITIAL_CAPITAL * (1 + net_return).cumprod()

    Returns
    -------
    dict with keys: weights_daily, daily_returns, equity_curve,
    turnover_series, metrics
    """
    cost_bps = cost_bps if cost_bps is not None else config.TRANSACTION_COST_BPS
    execution_lag = execution_lag if execution_lag is not None else config.EXECUTION_LAG_DAYS
    n_quintiles = n_quintiles or config.N_QUINTILES

    factor_zscore = factor.compute_zscore(prices)

    if rebalance_dates is None:
        rebalance_dates = get_rebalance_dates(prices.index, factor.rebalance_freq)

    weights_at_rebal = build_weight_panel(factor_zscore, rebalance_dates, n_quintiles)
    weights_daily = weights_at_rebal.reindex(prices.index, method="ffill").fillna(0.0)
    weights_lagged = weights_daily.shift(execution_lag).fillna(0.0)

    daily_ret = log_returns(prices).fillna(0.0)
    gross_return = (weights_lagged * daily_ret).sum(axis=1)

    turnover_series = weights_lagged.diff().abs().sum(axis=1).fillna(0.0)
    net_return = gross_return - turnover_series * cost_bps / 10_000

    equity_curve = config.INITIAL_CAPITAL * (1 + net_return).cumprod()

    metrics = compute_factor_metrics(net_return, equity_curve, weights_lagged, label=factor.name)

    return {
        "weights_daily": weights_lagged,
        "daily_returns": net_return,
        "equity_curve": equity_curve,
        "turnover_series": turnover_series,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data.fetch_data import fetch_prices
    from factors.momentum import MomentumFactor

    prices = fetch_prices()
    factor = MomentumFactor()

    result = run_factor_backtest(prices, factor)
    print("Metrics:", result["metrics"])
    print("\nEquity curve tail:")
    print(result["equity_curve"].tail())
    print("\nDaily returns describe:")
    print(result["daily_returns"].describe())
