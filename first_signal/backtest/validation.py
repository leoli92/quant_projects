"""
backtest/validation.py
-----------------------
In-sample vs out-of-sample validation.

For each selected pair:
  1. Fit hedge ratio and optimise thresholds on the in-sample period.
  2. Apply those fixed parameters to the out-of-sample period.
  3. Compare performance metrics side-by-side.
  4. Flag potential overfitting (large IS/OOS Sharpe gap).
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from analysis.spread import (
    compute_hedge_ratio_static,
    compute_spread,
    compute_zscore,
    generate_signals,
)
from backtest.engine import run_backtest, run_portfolio_backtest


def validate_pair(
    log_px_is: pd.DataFrame,
    log_px_oos: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    best_entry: float = None,
    best_exit: float = None,
    window: int = None,
) -> dict:
    """
    Run IS and OOS backtests for a single pair using fixed parameters.

    Parameters
    ----------
    log_px_is   : in-sample log-price DataFrame
    log_px_oos  : out-of-sample log-price DataFrame
    ticker_a/b  : ticker symbols
    best_entry  : optimised entry threshold (falls back to config default)
    best_exit   : optimised exit  threshold (falls back to config default)
    window      : rolling z-score window

    Returns
    -------
    dict with keys: is_result, oos_result, comparison_df
    """
    best_entry = best_entry if best_entry is not None else config.ENTRY_ZSCORE
    best_exit  = best_exit  if best_exit  is not None else config.EXIT_ZSCORE
    window     = window     if window     is not None else config.LOOKBACK_WINDOW

    # Fit hedge ratio on IS data only
    log_a_is = log_px_is[ticker_a].dropna()
    log_b_is = log_px_is[ticker_b].dropna()
    common_is = log_a_is.index.intersection(log_b_is.index)
    log_a_is, log_b_is = log_a_is.loc[common_is], log_b_is.loc[common_is]
    hedge_ratio = compute_hedge_ratio_static(log_a_is, log_b_is)

    def _build_and_backtest(log_px, label):
        log_a = log_px[ticker_a].dropna()
        log_b = log_px[ticker_b].dropna()
        common = log_a.index.intersection(log_b.index)
        log_a, log_b = log_a.loc[common], log_b.loc[common]

        spread  = compute_spread(log_a, log_b, hedge_ratio)
        zscore  = compute_zscore(spread, window=window)
        signals = generate_signals(zscore, entry=best_entry, exit_=best_exit)

        pair_df = pd.DataFrame(
            {"log_a": log_a, "log_b": log_b,
             "spread": spread, "zscore": zscore, "signal": signals}
        )
        return run_backtest(pair_df, ticker_a, ticker_b, hedge_ratio)

    is_result  = _build_and_backtest(log_px_is,  label="IS")
    oos_result = _build_and_backtest(log_px_oos, label="OOS")

    # Side-by-side comparison table
    is_m  = is_result["metrics"]
    oos_m = oos_result["metrics"]

    comparison = pd.DataFrame(
        {
            "Metric":          ["Ann. Return", "Ann. Volatility", "Sharpe Ratio",
                                "Max Drawdown", "Win Rate", "Avg Trade Return",
                                "Num Trades", "Turnover"],
            "In-Sample":       [is_m["annualized_return"], is_m["annualized_vol"],
                                is_m["sharpe_ratio"],      is_m["max_drawdown"],
                                is_m["win_rate"],          is_m["avg_trade_return"],
                                is_m["num_trades"],        is_m["turnover"]],
            "Out-of-Sample":   [oos_m["annualized_return"], oos_m["annualized_vol"],
                                oos_m["sharpe_ratio"],       oos_m["max_drawdown"],
                                oos_m["win_rate"],           oos_m["avg_trade_return"],
                                oos_m["num_trades"],         oos_m["turnover"]],
        }
    )

    return {
        "ticker_a":      ticker_a,
        "ticker_b":      ticker_b,
        "hedge_ratio":   hedge_ratio,
        "is_result":     is_result,
        "oos_result":    oos_result,
        "comparison_df": comparison,
    }


def validate_all_pairs(
    log_px_is: pd.DataFrame,
    log_px_oos: pd.DataFrame,
    optimized_pairs: pd.DataFrame,
    window: int = None,
) -> tuple:
    """
    Validate all optimised pairs and build a portfolio-level IS/OOS comparison.

    Parameters
    ----------
    log_px_is       : in-sample log-price DataFrame
    log_px_oos      : out-of-sample log-price DataFrame
    optimized_pairs : DataFrame from optimization.optimize_all_pairs()
    window          : rolling z-score window

    Returns
    -------
    pair_validations : list of dicts from validate_pair()
    summary_df       : DataFrame with IS/OOS Sharpe for each pair
    portfolio_is     : portfolio backtest result (IS)
    portfolio_oos    : portfolio backtest result (OOS)
    """
    pair_validations = []
    summary_rows = []

    for _, row in optimized_pairs.iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        entry = float(row["best_entry"])
        exit_ = float(row["best_exit"])

        # Skip if tickers not available in both periods
        if a not in log_px_is.columns or b not in log_px_is.columns:
            continue
        if a not in log_px_oos.columns or b not in log_px_oos.columns:
            continue

        try:
            val = validate_pair(
                log_px_is, log_px_oos, a, b,
                best_entry=entry, best_exit=exit_, window=window,
            )
            pair_validations.append(val)
            summary_rows.append(
                {
                    "pair":        f"{a}/{b}",
                    "is_sharpe":   val["is_result"]["metrics"]["sharpe_ratio"],
                    "oos_sharpe":  val["oos_result"]["metrics"]["sharpe_ratio"],
                    "is_return":   val["is_result"]["metrics"]["annualized_return"],
                    "oos_return":  val["oos_result"]["metrics"]["annualized_return"],
                    "is_maxdd":    val["is_result"]["metrics"]["max_drawdown"],
                    "oos_maxdd":   val["oos_result"]["metrics"]["max_drawdown"],
                    "sharpe_decay": (
                        val["is_result"]["metrics"]["sharpe_ratio"]
                        - val["oos_result"]["metrics"]["sharpe_ratio"]
                    ),
                }
            )
        except Exception as e:
            print(f"[validation] Skipping {a}/{b}: {e}")

    summary_df = pd.DataFrame(summary_rows).sort_values("oos_sharpe", ascending=False)

    # Portfolio-level IS and OOS
    is_pair_results  = [v["is_result"]  for v in pair_validations]
    oos_pair_results = [v["oos_result"] for v in pair_validations]

    portfolio_is  = run_portfolio_backtest(is_pair_results)  if is_pair_results  else {}
    portfolio_oos = run_portfolio_backtest(oos_pair_results) if oos_pair_results else {}

    return pair_validations, summary_df, portfolio_is, portfolio_oos
