"""
backtest/metrics.py
--------------------
Performance metrics for a factor backtest equity curve.

annualized_return / annualized_volatility / sharpe_ratio / max_drawdown /
drawdown_series are reused unchanged from pairs_trading/backtest/metrics.py
(generic — operate on any daily-returns/equity-curve Series). turnover() and
compute_factor_metrics() are specific to this project's continuous
long/short quintile weights (not the {-1,0,1} discrete signals used by the
pairs-trading engine).
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
    """Annualized volatility (std of daily returns x sqrt(252))."""
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio. risk_free_rate is annual (e.g. 0.04 for 4%)."""
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    vol = excess.std()
    if vol == 0:
        return 0.0
    return float(excess.mean() / vol * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a fraction (e.g. -0.15 means -15%)."""
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return float(drawdown.min())


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Full drawdown time series."""
    roll_max = equity_curve.cummax()
    return (equity_curve - roll_max) / roll_max


def turnover(weights_daily: pd.DataFrame) -> float:
    """
    Mean daily L1 turnover / 2, for continuous long/short weights (not the
    discrete {-1,0,1} signals used elsewhere) — dividing by 2 avoids
    double-counting a simultaneous buy+sell of equal size as turnover of 2.
    """
    changes = weights_daily.diff().abs().sum(axis=1)
    return float(changes.mean() / 2)


def compute_factor_metrics(
    daily_returns: pd.Series,
    equity_curve: pd.Series,
    weights_daily: pd.DataFrame,
    label: str = "",
) -> dict:
    """
    Compute and return performance metrics for a factor long/short portfolio.

    No win_rate/avg_trade_return/num_trades: a continuously-rebalanced
    quintile portfolio has no discrete trade log to draw those from.
    """
    return {
        "label": label,
        "annualized_return": round(annualized_return(daily_returns), 4),
        "annualized_vol": round(annualized_volatility(daily_returns), 4),
        "sharpe_ratio": round(sharpe_ratio(daily_returns), 4),
        "max_drawdown": round(max_drawdown(equity_curve), 4),
        "turnover": round(turnover(weights_daily), 6),
    }
