"""
analysis/cointegration.py
--------------------------
Cointegration testing for all pairs in the ticker universe.

Steps:
  1. Run Engle-Granger cointegration test on every pair (log prices).
  2. Run ADF test on the resulting spread to confirm stationarity.
  3. Return a p-value matrix and a ranked table of candidate pairs.
"""

import os
import sys
import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def run_cointegration_tests(
    log_px: pd.DataFrame,
    pvalue_threshold: float = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test all pairs for cointegration using the Engle-Granger two-step method.

    Parameters
    ----------
    log_px            : DataFrame of log prices (rows = dates, cols = tickers)
    pvalue_threshold  : significance level (default config.COINT_PVALUE_THRESHOLD)

    Returns
    -------
    pvalue_matrix : symmetric DataFrame of EG p-values (NaN on diagonal)
    candidates    : DataFrame of pairs with p-value < threshold, sorted by p-value
                    columns: ticker_a, ticker_b, eg_pvalue, adf_pvalue, adf_stat
    """
    pvalue_threshold = pvalue_threshold or config.COINT_PVALUE_THRESHOLD
    tickers = list(log_px.columns)
    n = len(tickers)

    # Build symmetric p-value matrix
    pval_matrix = pd.DataFrame(np.nan, index=tickers, columns=tickers)

    records = []
    pairs = list(itertools.combinations(tickers, 2))
    print(f"[cointegration] Testing {len(pairs)} pairs …")

    for a, b in pairs:
        series_a = log_px[a].dropna()
        series_b = log_px[b].dropna()
        # Align on common dates
        common = series_a.index.intersection(series_b.index)
        if len(common) < 100:
            continue
        sa, sb = series_a.loc[common], series_b.loc[common]

        try:
            eg_stat, eg_pval, _ = coint(sa, sb)
        except Exception:
            continue

        pval_matrix.loc[a, b] = eg_pval
        pval_matrix.loc[b, a] = eg_pval

        if eg_pval < pvalue_threshold:
            # ADF test on the OLS residual (spread)
            hedge = _ols_hedge_ratio(sa, sb)
            spread = sa - hedge * sb
            try:
                adf_stat, adf_pval, *_ = adfuller(spread, autolag="AIC")
            except Exception:
                adf_stat, adf_pval = np.nan, np.nan

            records.append(
                {
                    "ticker_a": a,
                    "ticker_b": b,
                    "eg_pvalue": round(eg_pval, 6),
                    "adf_stat": round(adf_stat, 4) if not np.isnan(adf_stat) else np.nan,
                    "adf_pvalue": round(adf_pval, 6) if not np.isnan(adf_pval) else np.nan,
                    "hedge_ratio": round(hedge, 6),
                }
            )

    candidates = pd.DataFrame(records)
    if not candidates.empty:
        candidates = candidates.sort_values("eg_pvalue").reset_index(drop=True)

    print(
        f"[cointegration] Found {len(candidates)} candidate pairs "
        f"(EG p-value < {pvalue_threshold})"
    )
    return pval_matrix, candidates


def _ols_hedge_ratio(series_a: pd.Series, series_b: pd.Series) -> float:
    """
    Estimate hedge ratio β via OLS: series_a = α + β * series_b + ε
    Returns β (the slope).
    """
    x = series_b.values
    y = series_a.values
    # Add constant
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(beta[1])


def adf_summary(spread: pd.Series, label: str = "") -> dict:
    """
    Run ADF test on a spread series and return a summary dict.
    """
    stat, pval, lags, nobs, crit, _ = adfuller(spread.dropna(), autolag="AIC")
    result = {
        "label": label,
        "adf_stat": round(stat, 4),
        "pvalue": round(pval, 6),
        "lags": lags,
        "nobs": nobs,
        "crit_1pct": round(crit["1%"], 4),
        "crit_5pct": round(crit["5%"], 4),
        "crit_10pct": round(crit["10%"], 4),
        "stationary_5pct": pval < 0.05,
    }
    return result


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data.fetch_data import fetch_prices, log_prices

    px = fetch_prices(
        start=config.IN_SAMPLE_START,
        end=config.IN_SAMPLE_END,
    )
    lp = log_prices(px)
    pval_mat, cands = run_cointegration_tests(lp)
    print("\nTop cointegrated pairs:")
    print(cands.head(10).to_string(index=False))
