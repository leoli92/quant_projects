"""
analysis/optimization.py
-------------------------
Grid search over entry/exit z-score thresholds to find the parameter
combination that maximises in-sample Sharpe ratio.

Also produces a sensitivity heatmap of Sharpe vs (entry, exit) thresholds.
"""

import os
import sys
import itertools

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from analysis.spread import compute_spread, compute_zscore, generate_signals
from backtest.engine import run_backtest


def optimize_thresholds(
    log_px: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    hedge_ratio: float,
    entry_range: list = None,
    exit_range: list = None,
    window: int = None,
) -> tuple:
    """
    Grid search over (entry, exit) threshold combinations.

    Parameters
    ----------
    log_px       : DataFrame of log prices (in-sample period)
    ticker_a/b   : ticker symbols
    hedge_ratio  : pre-computed OLS hedge ratio
    entry_range  : list of entry z-score values to test
    exit_range   : list of exit  z-score values to test
    window       : rolling z-score window

    Returns
    -------
    best_params  : dict with best entry, exit, and achieved Sharpe
    results_df   : DataFrame with all (entry, exit, sharpe) combinations
    sharpe_pivot : pivot table suitable for heatmap plotting
    """
    entry_range = entry_range if entry_range is not None else config.ENTRY_RANGE
    exit_range  = exit_range  if exit_range  is not None else config.EXIT_RANGE
    window      = window      if window      is not None else config.LOOKBACK_WINDOW

    log_a = log_px[ticker_a].dropna()
    log_b = log_px[ticker_b].dropna()
    common = log_a.index.intersection(log_b.index)
    log_a, log_b = log_a.loc[common], log_b.loc[common]

    spread = compute_spread(log_a, log_b, hedge_ratio)
    zscore = compute_zscore(spread, window=window)

    records = []
    for entry, exit_ in itertools.product(entry_range, exit_range):
        if exit_ >= entry:
            continue  # exit must be tighter than entry

        signals = generate_signals(zscore, entry=entry, exit_=exit_)

        pair_df = pd.DataFrame(
            {"log_a": log_a, "log_b": log_b, "spread": spread,
             "zscore": zscore, "signal": signals}
        )

        result = run_backtest(pair_df, ticker_a, ticker_b, hedge_ratio)
        sharpe = result["metrics"]["sharpe_ratio"]
        n_trades = result["metrics"]["num_trades"]

        records.append(
            {
                "entry":    entry,
                "exit":     exit_,
                "sharpe":   sharpe,
                "n_trades": n_trades,
                "ann_return": result["metrics"]["annualized_return"],
                "max_dd":   result["metrics"]["max_drawdown"],
            }
        )

    results_df = pd.DataFrame(records)

    if results_df.empty:
        return {}, results_df, pd.DataFrame()

    best_row = results_df.loc[results_df["sharpe"].idxmax()]
    best_params = {
        "entry":    float(best_row["entry"]),
        "exit":     float(best_row["exit"]),
        "sharpe":   float(best_row["sharpe"]),
        "n_trades": int(best_row["n_trades"]),
    }

    sharpe_pivot = results_df.pivot(index="entry", columns="exit", values="sharpe")

    print(
        f"[optimization] {ticker_a}/{ticker_b} — best entry={best_params['entry']}, "
        f"exit={best_params['exit']}, Sharpe={best_params['sharpe']:.3f}"
    )
    return best_params, results_df, sharpe_pivot


def optimize_all_pairs(
    log_px: pd.DataFrame,
    candidates: pd.DataFrame,
    entry_range: list = None,
    exit_range: list = None,
    window: int = None,
) -> pd.DataFrame:
    """
    Run threshold optimisation for every candidate pair.

    Parameters
    ----------
    log_px      : log-price DataFrame (in-sample)
    candidates  : DataFrame from cointegration.run_cointegration_tests()
    entry_range : grid of entry thresholds
    exit_range  : grid of exit  thresholds
    window      : rolling z-score window

    Returns
    -------
    DataFrame with columns: ticker_a, ticker_b, hedge_ratio,
                             best_entry, best_exit, best_sharpe
    """
    rows = []
    for _, row in candidates.iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        hr   = row["hedge_ratio"]
        try:
            best, _, _ = optimize_thresholds(
                log_px, a, b, hr,
                entry_range=entry_range,
                exit_range=exit_range,
                window=window,
            )
            rows.append(
                {
                    "ticker_a":    a,
                    "ticker_b":    b,
                    "hedge_ratio": hr,
                    "best_entry":  best.get("entry", config.ENTRY_ZSCORE),
                    "best_exit":   best.get("exit",  config.EXIT_ZSCORE),
                    "best_sharpe": best.get("sharpe", float("nan")),
                    "n_trades":    best.get("n_trades", 0),
                }
            )
        except Exception as e:
            print(f"[optimization] Skipping {a}/{b}: {e}")

    return pd.DataFrame(rows).sort_values("best_sharpe", ascending=False).reset_index(drop=True)
