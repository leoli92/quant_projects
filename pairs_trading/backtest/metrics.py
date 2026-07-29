"""
backtest/metrics.py
--------------------
Performance metrics for a backtest equity curve.

All functions accept a pd.Series of daily PnL or returns.
"""

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def annualized_return(daily_returns: pd.Series) -> float:
    """Compound annualized return from a series of daily log-returns."""
    total = daily_returns.sum()
    n_days = len(daily_returns)
    if n_days == 0:
        return 0.0
    return float(np.exp(total * TRADING_DAYS_PER_YEAR / n_days) - 1)


def annualized_volatility(daily_returns: pd.Series) -> float:
    """Annualized volatility (std of daily returns × √252)."""
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualized Sharpe ratio.
    risk_free_rate is the annual risk-free rate (e.g. 0.04 for 4 %).
    """
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    vol = excess.std()
    if vol == 0:
        return 0.0
    return float(excess.mean() / vol * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum drawdown as a fraction (e.g. -0.15 means -15 %).
    equity_curve should be a cumulative wealth index (starts at 1 or capital).
    """
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return float(drawdown.min())


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return the full drawdown time series."""
    roll_max = equity_curve.cummax()
    return (equity_curve - roll_max) / roll_max


def win_rate(trade_returns: pd.Series) -> float:
    """Fraction of trades with positive return."""
    if len(trade_returns) == 0:
        return float("nan")
    return float((trade_returns > 0).sum() / len(trade_returns))


def average_trade_return(trade_returns: pd.Series) -> float:
    """Mean return per trade."""
    if len(trade_returns) == 0:
        return float("nan")
    return float(trade_returns.mean())


def turnover(signals: pd.Series) -> float:
    """
    Average daily turnover = mean of |Δposition| / 2.
    Signals are {-1, 0, +1}.
    """
    changes = signals.diff().abs()
    return float(changes.mean() / 2)


def compute_all_metrics(
    daily_returns: pd.Series,
    equity_curve: pd.Series,
    trade_returns: pd.Series,
    signals: pd.Series,
    label: str = "",
) -> dict:
    """
    Compute and return all performance metrics as a dictionary.
    """
    metrics = {
        "label":              label,
        "annualized_return":  round(annualized_return(daily_returns), 4),
        "annualized_vol":     round(annualized_volatility(daily_returns), 4),
        "sharpe_ratio":       round(sharpe_ratio(daily_returns), 4),
        "max_drawdown":       round(max_drawdown(equity_curve), 4),
        "win_rate":           round(win_rate(trade_returns), 4),
        "avg_trade_return":   round(average_trade_return(trade_returns), 6),
        "num_trades":         len(trade_returns),
        "turnover":           round(turnover(signals), 6),
    }
    return metrics
