"""
factors/base.py
----------------
Abstract base class for cross-sectional equity factors.

Subclasses implement `compute()` — the ONLY place factor-specific logic
lives — and inherit cross-sectional ranking/z-scoring/NaN-rate reporting
for free. This is also the seam for a future fundamentals-based factor
(e.g. a WRDS-backed value/earnings-quality factor): a new subclass just
needs a causal `compute()` implementation to plug into the rest of the
pipeline (analysis/, backtest/, visualization/) unchanged.
"""

from abc import ABC, abstractmethod

import pandas as pd


class Factor(ABC):
    """
    Parameters
    ----------
    name           : short identifier used in filenames / summary tables
    lookback_days  : trading days of trailing history required before the
                      first valid (non-NaN) factor value
    rebalance_freq : "M" (monthly) or "W" (weekly) — passed to
                      analysis.cross_sectional.get_rebalance_dates()
    """

    name: str = "base"
    lookback_days: int = 0
    rebalance_freq: str = "M"

    @abstractmethod
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Raw factor values, index=date, columns=ticker.

        MUST be causal: every value at row t may only depend on data at or
        before t (via .shift(n>=0) / .rolling()). Never use .shift(-n) or
        center=True here — that is lookahead bias. NaN where insufficient
        trailing history exists.
        """
        raise NotImplementedError

    @staticmethod
    def zscore(raw: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional (row-wise) z-score. NaN values are excluded from
        the per-date mean/std and remain NaN in the output."""
        mu = raw.mean(axis=1)
        sd = raw.std(axis=1)
        return raw.sub(mu, axis=0).div(sd, axis=0)

    @staticmethod
    def rank(raw: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional percentile rank per date, in [0, 1]."""
        return raw.rank(axis=1, pct=True)

    @staticmethod
    def nan_rate(raw: pd.DataFrame) -> pd.Series:
        """Fraction of the universe with a NaN factor value, per date."""
        return raw.isna().mean(axis=1)

    def compute_zscore(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Convenience wrapper: compute() then cross-sectional zscore()."""
        return self.zscore(self.compute(prices))


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    class _ToyFactor(Factor):
        name = "toy"

        def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
            return prices  # identity, just for testing the shared helpers

    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    toy_prices = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [2.0, 4.0, 6.0],
            "C": [3.0, np.nan, 9.0],
            "D": [4.0, 8.0, 12.0],
            "E": [5.0, 10.0, 15.0],
        },
        index=dates,
    )

    factor = _ToyFactor()
    raw = factor.compute(toy_prices)
    z = factor.zscore(raw)
    r = factor.rank(raw)
    nr = factor.nan_rate(raw)

    print("Z-score row mean (should be ~0):")
    print(z.mean(axis=1))
    print("\nZ-score row std (should be ~1):")
    print(z.std(axis=1))
    print("\nRank (should be in [0, 1]):")
    print(r)
    print("\nNaN rate per date:")
    print(nr)
