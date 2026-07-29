"""
analysis/spread.py
------------------
Construct the spread between a cointegrated pair and generate trading signals.

Steps:
  1. Estimate hedge ratio β via rolling OLS (or static OLS on in-sample data).
  2. Compute spread = log(P_A) − β · log(P_B).
  3. Normalise to a rolling z-score.
  4. Generate long/short entry and exit signals based on z-score thresholds.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def compute_hedge_ratio_static(
    log_a: pd.Series,
    log_b: pd.Series,
) -> float:
    """
    Estimate a static (full-period) OLS hedge ratio.
    Regresses log_a on log_b: log_a = α + β·log_b + ε
    Returns β.
    """
    x = log_b.values.reshape(-1, 1)
    y = log_a.values
    model = LinearRegression().fit(x, y)
    return float(model.coef_[0])


def compute_spread(
    log_a: pd.Series,
    log_b: pd.Series,
    hedge_ratio: float,
) -> pd.Series:
    """
    Compute the spread series: spread = log_a − β · log_b
    """
    spread = log_a - hedge_ratio * log_b
    spread.name = f"spread_{log_a.name}_{log_b.name}"
    return spread


def compute_zscore(
    spread: pd.Series,
    window: int = None,
) -> pd.Series:
    """
    Compute rolling z-score of the spread.

    z = (spread − rolling_mean) / rolling_std

    Parameters
    ----------
    spread : spread time series
    window : lookback window in trading days (default config.LOOKBACK_WINDOW)
    """
    window = window or config.LOOKBACK_WINDOW
    roll_mean = spread.rolling(window).mean()
    roll_std  = spread.rolling(window).std()
    zscore = (spread - roll_mean) / roll_std
    zscore.name = "zscore"
    return zscore


def generate_signals(
    zscore: pd.Series,
    entry: float = None,
    exit_: float = None,
    stop_loss: float = None,
) -> pd.Series:
    """
    Generate position signals from z-score.

    Convention
    ----------
    +1  : long spread  (long A, short B) — spread is too low, expect mean reversion up
    -1  : short spread (short A, long B) — spread is too high, expect mean reversion down
     0  : flat (no position)

    Entry rules
    -----------
    z < -entry  →  go long  spread (+1)
    z >  entry  →  go short spread (-1)

    Exit rules
    ----------
    |z| < exit_  →  close position (0)
    |z| > stop_loss → emergency close (0)

    Parameters
    ----------
    zscore    : z-score series
    entry     : entry threshold (default config.ENTRY_ZSCORE)
    exit_     : exit  threshold (default config.EXIT_ZSCORE)
    stop_loss : stop-loss threshold (default config.STOP_LOSS_ZSCORE)

    Returns
    -------
    pd.Series of {-1, 0, +1}
    """
    entry     = entry     if entry     is not None else config.ENTRY_ZSCORE
    exit_     = exit_     if exit_     is not None else config.EXIT_ZSCORE
    stop_loss = stop_loss if stop_loss is not None else config.STOP_LOSS_ZSCORE

    position = pd.Series(0, index=zscore.index, dtype=int)
    current  = 0

    for i, z in enumerate(zscore):
        if np.isnan(z):
            position.iloc[i] = 0
            continue

        if current == 0:
            if z < -entry:
                current = 1
            elif z > entry:
                current = -1
        elif current == 1:
            if abs(z) < exit_ or z > stop_loss:
                current = 0
        elif current == -1:
            if abs(z) < exit_ or z < -stop_loss:
                current = 0

        position.iloc[i] = current

    position.name = "signal"
    return position


def build_pair_signals(
    log_px: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    hedge_ratio: float = None,
    window: int = None,
    entry: float = None,
    exit_: float = None,
    stop_loss: float = None,
) -> pd.DataFrame:
    """
    Full pipeline for one pair: hedge ratio → spread → z-score → signals.

    Returns a DataFrame with columns:
        log_a, log_b, spread, zscore, signal
    """
    log_a = log_px[ticker_a].dropna()
    log_b = log_px[ticker_b].dropna()
    common = log_a.index.intersection(log_b.index)
    log_a, log_b = log_a.loc[common], log_b.loc[common]

    if hedge_ratio is None:
        hedge_ratio = compute_hedge_ratio_static(log_a, log_b)

    spread  = compute_spread(log_a, log_b, hedge_ratio)
    zscore  = compute_zscore(spread, window=window)
    signals = generate_signals(zscore, entry=entry, exit_=exit_, stop_loss=stop_loss)

    result = pd.DataFrame(
        {
            "log_a":  log_a,
            "log_b":  log_b,
            "spread": spread,
            "zscore": zscore,
            "signal": signals,
        }
    )
    return result, hedge_ratio


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data.fetch_data import fetch_prices, log_prices

    px = fetch_prices(start=config.IN_SAMPLE_START, end=config.IN_SAMPLE_END)
    lp = log_prices(px)

    df, hr = build_pair_signals(lp, "XOM", "CVX")
    print(f"Hedge ratio XOM/CVX: {hr:.4f}")
    print(df.tail(10))
    print(f"\nSignal counts:\n{df['signal'].value_counts()}")
