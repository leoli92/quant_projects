"""
backtest/engine.py
------------------
Core backtesting engine for a single pairs-trading strategy.

Given a signal series (+1 / -1 / 0) and the two price series, the engine:
  1. Computes daily PnL from the spread position.
  2. Deducts transaction costs on every position change.
  3. Builds an equity curve.
  4. Produces a trade log (entry date, exit date, direction, return).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from backtest.metrics import compute_all_metrics


def run_backtest(
    pair_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    hedge_ratio: float,
    cost_bps: float = None,
    initial_capital: float = None,
) -> dict:
    """
    Backtest a single pair.

    Parameters
    ----------
    pair_df        : DataFrame with columns [log_a, log_b, spread, zscore, signal]
                     as produced by analysis.spread.build_pair_signals()
    ticker_a/b     : ticker labels (for the trade log)
    hedge_ratio    : β used to size the position
    cost_bps       : one-way transaction cost in basis points
    initial_capital: starting capital in USD

    Returns
    -------
    dict with keys:
        equity_curve   : pd.Series  (cumulative wealth, starts at initial_capital)
        daily_returns  : pd.Series  (daily log-returns of the strategy)
        trade_log      : pd.DataFrame
        metrics        : dict of performance statistics
    """
    cost_bps        = cost_bps        if cost_bps        is not None else config.TRANSACTION_COST_BPS
    initial_capital = initial_capital if initial_capital is not None else config.INITIAL_CAPITAL

    cost_per_trade = cost_bps / 10_000  # convert bps → fraction

    signals = pair_df["signal"]
    log_a   = pair_df["log_a"]
    log_b   = pair_df["log_b"]

    # Daily log-return of the spread position
    # When signal = +1: long A, short B  → PnL = Δlog_a − β·Δlog_b
    # When signal = -1: short A, long B  → PnL = −(Δlog_a − β·Δlog_b)
    spread_return = log_a.diff() - hedge_ratio * log_b.diff()

    # Strategy return = signal (lagged by 1 day) × spread_return
    strat_return = signals.shift(1) * spread_return

    # Deduct transaction costs on every position change
    position_change = signals.diff().abs()
    tc_series = position_change * cost_per_trade

    net_return = strat_return - tc_series
    net_return = net_return.fillna(0.0)

    # Equity curve
    equity_curve = initial_capital * (1 + net_return).cumprod()

    # Trade log
    trade_log = _build_trade_log(signals, net_return, ticker_a, ticker_b)

    trade_returns = pd.Series(
        [t["trade_return"] for t in trade_log], dtype=float
    )

    metrics = compute_all_metrics(
        daily_returns=net_return,
        equity_curve=equity_curve,
        trade_returns=trade_returns,
        signals=signals,
        label=f"{ticker_a}/{ticker_b}",
    )

    return {
        "equity_curve":  equity_curve,
        "daily_returns": net_return,
        "trade_log":     pd.DataFrame(trade_log),
        "metrics":       metrics,
    }


def _build_trade_log(
    signals: pd.Series,
    net_return: pd.Series,
    ticker_a: str,
    ticker_b: str,
) -> list:
    """
    Iterate through the signal series and record each completed trade.
    A trade starts when signal changes from 0 → ±1 and ends when it returns to 0.
    """
    trades = []
    in_trade = False
    entry_date = None
    direction = 0
    cumulative_return = 0.0

    for date, sig in signals.items():
        prev_sig = signals.shift(1).loc[date] if date != signals.index[0] else 0

        if not in_trade and sig != 0:
            in_trade = True
            entry_date = date
            direction = int(sig)
            cumulative_return = 0.0

        if in_trade:
            cumulative_return += float(net_return.loc[date])

            if sig == 0:
                trades.append(
                    {
                        "ticker_a":     ticker_a,
                        "ticker_b":     ticker_b,
                        "direction":    "long_spread" if direction == 1 else "short_spread",
                        "entry_date":   entry_date,
                        "exit_date":    date,
                        "trade_return": round(cumulative_return, 6),
                        "holding_days": (date - entry_date).days,
                    }
                )
                in_trade = False
                entry_date = None
                direction = 0
                cumulative_return = 0.0

    # Close any open trade at end of period
    if in_trade and entry_date is not None:
        trades.append(
            {
                "ticker_a":     ticker_a,
                "ticker_b":     ticker_b,
                "direction":    "long_spread" if direction == 1 else "short_spread",
                "entry_date":   entry_date,
                "exit_date":    signals.index[-1],
                "trade_return": round(cumulative_return, 6),
                "holding_days": (signals.index[-1] - entry_date).days,
            }
        )

    return trades


def run_portfolio_backtest(
    pair_results: list,
    initial_capital: float = None,
) -> dict:
    """
    Combine individual pair equity curves into a portfolio.

    Each pair is equally weighted. The portfolio daily return is the
    equal-weighted average of individual pair daily returns.

    Parameters
    ----------
    pair_results : list of dicts returned by run_backtest()
    initial_capital : total portfolio capital

    Returns
    -------
    dict with equity_curve, daily_returns, metrics
    """
    initial_capital = initial_capital if initial_capital is not None else config.INITIAL_CAPITAL

    if not pair_results:
        return {}

    returns_list = [r["daily_returns"] for r in pair_results]
    combined = pd.concat(returns_list, axis=1).fillna(0.0)
    portfolio_returns = combined.mean(axis=1)

    equity_curve = initial_capital * (1 + portfolio_returns).cumprod()

    all_trades = pd.concat(
        [r["trade_log"] for r in pair_results if not r["trade_log"].empty],
        ignore_index=True,
    )
    trade_returns = all_trades["trade_return"] if not all_trades.empty else pd.Series(dtype=float)

    # Aggregate signals (average absolute position)
    signals_list = [r["daily_returns"].apply(np.sign) for r in pair_results]
    agg_signals = pd.concat(signals_list, axis=1).fillna(0).mean(axis=1)

    metrics = compute_all_metrics(
        daily_returns=portfolio_returns,
        equity_curve=equity_curve,
        trade_returns=trade_returns,
        signals=agg_signals,
        label="Portfolio",
    )

    return {
        "equity_curve":  equity_curve,
        "daily_returns": portfolio_returns,
        "trade_log":     all_trades,
        "metrics":       metrics,
    }
