"""
main.py
-------
Full pipeline runner: fetch prices, compute factors, validate (IC/IR,
IS/OOS backtest), and generate tearsheets for all factors.
"""

import config
from backtest.validation import validate_all_factors
from data.fetch_data import fetch_prices
from factors.momentum import MomentumFactor
from factors.reversal import ReversalFactor
from factors.volatility import VolatilityFactor
from visualization.tearsheet import build_tearsheet


def _banner(step: int, msg: str):
    print(f"\n{'=' * 70}\nSTEP {step} — {msg}\n{'=' * 70}")


def main():
    _banner(1, "Fetch prices (with CSV cache)")
    prices = fetch_prices()
    print(f"Price panel: {prices.shape[0]} trading days x {prices.shape[1]} tickers")

    _banner(2, "Instantiate factors")
    factors = [MomentumFactor(), ReversalFactor(), VolatilityFactor()]
    for f in factors:
        print(f"  - {f.name} (rebalance={f.rebalance_freq}, lookback={f.lookback_days}d)")

    _banner(3, "NaN-rate sanity check (in-sample window)")
    for f in factors:
        nan_rate = f.nan_rate(f.compute(prices)).loc[config.IN_SAMPLE_START:config.IN_SAMPLE_END]
        print(f"  {f.name}: mean NaN rate = {nan_rate.mean():.4f}")

    _banner(4, "Validate factors (IC/IR, IS/OOS backtest, Sharpe decay)")
    summary_df, results = validate_all_factors(prices, factors)
    print(summary_df.to_string(index=False))

    _banner(5, "Build tearsheets")
    for f in factors:
        build_tearsheet(f, results[f.name])

    _banner(6, "Save cross-factor summary")
    summary_path = f"{config.OUTPUT_DIR}/factor_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved -> {summary_path}")

    print("\nPipeline complete. See outputs/ for charts and CSVs.")


if __name__ == "__main__":
    main()
