"""
backtest/validation.py
-----------------------
In-sample vs out-of-sample validation for a single factor and across all
factors: IC/IR, backtest Sharpe, and Sharpe decay (overfitting flag).
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from analysis.cross_sectional import forward_returns, get_rebalance_dates
from analysis.ic import compute_ic_series
from analysis.ir import ic_summary
from analysis.quintiles import quintile_returns
from backtest.engine import run_factor_backtest
from backtest.metrics import compute_factor_metrics
from factors.base import Factor


def validate_factor(prices: pd.DataFrame, factor: Factor) -> dict:
    """
    Run one continuous backtest spanning IN_SAMPLE_START..OOS_END (the
    DATA_START..IN_SAMPLE_START window is warmup for factor.compute() only
    and is never traded), then slice the resulting daily-returns series
    into IS/OOS sub-periods. This gives an IS Sharpe covering the full
    2015-2022 window with no truncation from the lookback requirement.

    Returns
    -------
    dict with keys: is_result, oos_result, is_metrics, oos_metrics,
    sharpe_decay, is_overfit, ic_series, ic_summary_is, ic_summary_oos,
    quintile_returns
    """
    rebalance_dates_full = get_rebalance_dates(prices.index, factor.rebalance_freq)
    traded_dates = rebalance_dates_full[
        (rebalance_dates_full >= pd.Timestamp(config.IN_SAMPLE_START))
        & (rebalance_dates_full <= pd.Timestamp(config.OOS_END))
    ]

    result = run_factor_backtest(prices, factor, rebalance_dates=traded_dates)
    daily_returns = result["daily_returns"]
    weights_daily = result["weights_daily"]

    is_returns = daily_returns.loc[config.IN_SAMPLE_START:config.IN_SAMPLE_END]
    oos_returns = daily_returns.loc[config.OOS_START:config.OOS_END]
    is_weights = weights_daily.loc[config.IN_SAMPLE_START:config.IN_SAMPLE_END]
    oos_weights = weights_daily.loc[config.OOS_START:config.OOS_END]

    # Rebase each sub-period into its own equity curve so OOS drawdown isn't
    # inherited from IS peaks.
    is_equity = config.INITIAL_CAPITAL * (1 + is_returns).cumprod()
    oos_equity = config.INITIAL_CAPITAL * (1 + oos_returns).cumprod()

    is_metrics = compute_factor_metrics(is_returns, is_equity, is_weights, label=f"{factor.name} IS")
    oos_metrics = compute_factor_metrics(oos_returns, oos_equity, oos_weights, label=f"{factor.name} OOS")

    is_result = {**result, "daily_returns": is_returns, "equity_curve": is_equity,
                 "weights_daily": is_weights, "metrics": is_metrics}
    oos_result = {**result, "daily_returns": oos_returns, "equity_curve": oos_equity,
                  "weights_daily": oos_weights, "metrics": oos_metrics}

    sharpe_decay = is_metrics["sharpe_ratio"] - oos_metrics["sharpe_ratio"]
    is_overfit = sharpe_decay > config.SHARPE_DECAY_THRESHOLD

    # IC/IR: computed on the full rebalance-date grid (independent of the
    # backtest's traded-date restriction — each date's IC only needs that
    # date plus the next rebalance date, so no chaining/lookback subtlety).
    factor_zscore = factor.compute_zscore(prices)
    fwd = forward_returns(prices, rebalance_dates_full)
    ic_series = compute_ic_series(factor_zscore.loc[rebalance_dates_full], fwd)
    qret = quintile_returns(factor_zscore.loc[rebalance_dates_full], fwd)

    ic_is = ic_series.loc[config.IN_SAMPLE_START:config.IN_SAMPLE_END]
    ic_oos = ic_series.loc[config.OOS_START:config.OOS_END]

    return {
        "factor_name": factor.name,
        "is_result": is_result,
        "oos_result": oos_result,
        "is_metrics": is_metrics,
        "oos_metrics": oos_metrics,
        "sharpe_decay": round(sharpe_decay, 4),
        "is_overfit": bool(is_overfit),
        "ic_series": ic_series,
        "ic_summary_is": ic_summary(ic_is),
        "ic_summary_oos": ic_summary(ic_oos),
        "quintile_returns": qret,
    }


def validate_all_factors(prices: pd.DataFrame, factors: list[Factor]) -> tuple[pd.DataFrame, dict]:
    """
    Validate every factor and build a summary table.

    Returns
    -------
    summary_df : one row per factor (ic_mean, ic_std, ir, is_sharpe,
                 oos_sharpe, sharpe_decay, flagged_overfit, nan_rate_mean),
                 sorted by oos_sharpe descending
    results    : dict mapping factor.name -> validate_factor() result, for
                 downstream tearsheet generation
    """
    rows = []
    results = {}
    for factor in factors:
        val = validate_factor(prices, factor)
        results[factor.name] = val

        nan_rate_mean = float(factor.nan_rate(factor.compute(prices)).loc[
            config.IN_SAMPLE_START:config.OOS_END
        ].mean())

        rows.append({
            "factor": factor.name,
            "ic_mean": round(val["ic_summary_is"]["ic_mean"], 4),
            "ic_std": round(val["ic_summary_is"]["ic_std"], 4),
            "ir": round(val["ic_summary_is"]["ir"], 4),
            "is_sharpe": val["is_metrics"]["sharpe_ratio"],
            "oos_sharpe": val["oos_metrics"]["sharpe_ratio"],
            "sharpe_decay": val["sharpe_decay"],
            "flagged_overfit": val["is_overfit"],
            "nan_rate_mean": round(nan_rate_mean, 4),
        })

    summary_df = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False)
    return summary_df, results


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data.fetch_data import fetch_prices
    from factors.momentum import MomentumFactor
    from factors.reversal import ReversalFactor
    from factors.volatility import VolatilityFactor

    prices = fetch_prices()
    factors = [MomentumFactor(), ReversalFactor(), VolatilityFactor()]

    summary_df, results = validate_all_factors(prices, factors)
    print(summary_df.to_string(index=False))
