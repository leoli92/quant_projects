"""
main.py
-------
Pairs Trading Strategy — Full Pipeline Orchestrator

Steps
-----
1.  Download & cache price data (yfinance).
2.  Run Engle-Granger cointegration tests on all pairs (in-sample).
3.  Optimise entry/exit z-score thresholds per pair (in-sample).
4.  Validate IS vs OOS performance.
5.  Generate all charts and save outputs.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

import config
from data.fetch_data import fetch_prices, log_prices
from analysis.cointegration import run_cointegration_tests
from analysis.spread import build_pair_signals
from analysis.optimization import optimize_all_pairs, optimize_thresholds
from backtest.engine import run_backtest, run_portfolio_backtest
from backtest.validation import validate_all_pairs
from visualization.plots import (
    plot_coint_heatmap,
    plot_spread,
    plot_zscore,
    plot_equity_curve,
    plot_sensitivity_heatmap,
    plot_is_oos_comparison,
    plot_portfolio_returns,
    save_all_outputs,
)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Fetch data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1 — Fetching price data")
    print("=" * 60)

    prices_all = fetch_prices(
        tickers=config.TICKERS,
        start=config.IN_SAMPLE_START,
        end=config.OOS_END,
    )

    # Split into IS and OOS
    prices_is  = prices_all.loc[config.IN_SAMPLE_START : config.IN_SAMPLE_END]
    prices_oos = prices_all.loc[config.OOS_START       : config.OOS_END]

    log_px_all = log_prices(prices_all)
    log_px_is  = log_prices(prices_is)
    log_px_oos = log_prices(prices_oos)

    if prices_is.empty or prices_oos.empty:
        print("ERROR: Price data is empty after splitting into IS/OOS periods.")
        print(f"  prices_all shape: {prices_all.shape}")
        print(f"  IS  slice [{config.IN_SAMPLE_START} : {config.IN_SAMPLE_END}]: {len(prices_is)} rows")
        print(f"  OOS slice [{config.OOS_START} : {config.OOS_END}]: {len(prices_oos)} rows")
        print("  Check that the cache file contains data for the configured date range.")
        return

    print(f"  IS  period: {prices_is.index[0].date()} → {prices_is.index[-1].date()} "
          f"({len(prices_is)} days)")
    print(f"  OOS period: {prices_oos.index[0].date()} → {prices_oos.index[-1].date()} "
          f"({len(prices_oos)} days)")

    # ------------------------------------------------------------------
    # 2. Cointegration testing (IS only)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 — Cointegration testing (in-sample)")
    print("=" * 60)

    pval_matrix, candidates = run_cointegration_tests(log_px_is)

    if candidates.empty:
        print("No cointegrated pairs found. Try relaxing COINT_PVALUE_THRESHOLD in config.py.")
        return

    print(f"\nTop 10 candidate pairs:")
    print(candidates.head(10).to_string(index=False))

    # Save candidates table
    cand_path = os.path.join(config.OUTPUT_DIR, "candidate_pairs.csv")
    candidates.to_csv(cand_path, index=False)
    print(f"\nSaved candidate pairs → {cand_path}")

    # Plot cointegration heatmap
    plot_coint_heatmap(pval_matrix)

    # ------------------------------------------------------------------
    # 3. Parameter optimisation (IS only)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 — Parameter optimisation (in-sample)")
    print("=" * 60)

    # Limit to top N pairs to keep runtime manageable
    MAX_PAIRS = 10
    top_candidates = candidates.head(MAX_PAIRS).copy()

    optimized_pairs = optimize_all_pairs(log_px_is, top_candidates)

    opt_path = os.path.join(config.OUTPUT_DIR, "optimized_pairs.csv")
    optimized_pairs.to_csv(opt_path, index=False)
    print(f"\nOptimised pairs saved → {opt_path}")
    print(optimized_pairs.to_string(index=False))

    # Collect sensitivity pivots for the top 3 pairs
    sensitivity_data = {}
    for _, row in optimized_pairs.head(3).iterrows():
        a, b, hr = row["ticker_a"], row["ticker_b"], row["hedge_ratio"]
        try:
            _, _, pivot = optimize_thresholds(log_px_is, a, b, hr)
            sensitivity_data[(a, b)] = pivot
            plot_sensitivity_heatmap(pivot, a, b)
        except Exception as e:
            print(f"  Sensitivity plot skipped for {a}/{b}: {e}")

    # ------------------------------------------------------------------
    # 4. Build spread/z-score charts for top pairs (IS period)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 — Spread & z-score charts (in-sample)")
    print("=" * 60)

    for _, row in optimized_pairs.head(5).iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        hr   = row["hedge_ratio"]
        entry = float(row["best_entry"])
        exit_ = float(row["best_exit"])

        try:
            pair_df, _ = build_pair_signals(
                log_px_is, a, b,
                hedge_ratio=hr,
                entry=entry,
                exit_=exit_,
            )
            plot_spread(pair_df, a, b)
            plot_zscore(pair_df, a, b, entry=entry, exit_=exit_)
        except Exception as e:
            print(f"  Chart skipped for {a}/{b}: {e}")

    # ------------------------------------------------------------------
    # 5. IS backtest for top pairs
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5 — In-sample backtest")
    print("=" * 60)

    is_pair_results = []
    for _, row in optimized_pairs.iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        hr   = row["hedge_ratio"]
        entry = float(row["best_entry"])
        exit_ = float(row["best_exit"])

        try:
            pair_df, _ = build_pair_signals(
                log_px_is, a, b,
                hedge_ratio=hr,
                entry=entry,
                exit_=exit_,
            )
            result = run_backtest(pair_df, a, b, hr)
            is_pair_results.append(result)

            m = result["metrics"]
            print(f"  {a}/{b}: Sharpe={m['sharpe_ratio']:.2f}, "
                  f"Ann.Ret={m['annualized_return']:.1%}, "
                  f"MaxDD={m['max_drawdown']:.1%}, "
                  f"Trades={m['num_trades']}")

            # Save trade log
            if not result["trade_log"].empty:
                tl_path = os.path.join(config.OUTPUT_DIR, f"trade_log_{a}_{b}_IS.csv")
                result["trade_log"].to_csv(tl_path, index=False)

            plot_equity_curve(
                result["equity_curve"],
                label=f"{a}/{b} IS",
                filename=f"equity_{a}_{b}_IS.png",
            )
        except Exception as e:
            print(f"  IS backtest skipped for {a}/{b}: {e}")

    # IS portfolio
    if is_pair_results:
        port_is = run_portfolio_backtest(is_pair_results)
        m = port_is["metrics"]
        print(f"\n  Portfolio IS: Sharpe={m['sharpe_ratio']:.2f}, "
              f"Ann.Ret={m['annualized_return']:.1%}, "
              f"MaxDD={m['max_drawdown']:.1%}")
        plot_equity_curve(port_is["equity_curve"], label="Portfolio IS",
                          filename="equity_portfolio_IS.png")
        plot_portfolio_returns(is_pair_results, port_is, label_suffix=" (IS)")

    # ------------------------------------------------------------------
    # 6. IS vs OOS validation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 — In-sample vs Out-of-sample validation")
    print("=" * 60)

    pair_validations, summary_df, portfolio_is, portfolio_oos = validate_all_pairs(
        log_px_is, log_px_oos, optimized_pairs
    )

    print("\nIS vs OOS Summary:")
    print(summary_df.to_string(index=False))

    # OOS equity curves
    for val in pair_validations:
        a, b = val["ticker_a"], val["ticker_b"]
        plot_equity_curve(
            val["oos_result"]["equity_curve"],
            label=f"{a}/{b} OOS",
            filename=f"equity_{a}_{b}_OOS.png",
        )

    if portfolio_oos:
        m = portfolio_oos["metrics"]
        print(f"\n  Portfolio OOS: Sharpe={m['sharpe_ratio']:.2f}, "
              f"Ann.Ret={m['annualized_return']:.1%}, "
              f"MaxDD={m['max_drawdown']:.1%}")
        plot_equity_curve(portfolio_oos["equity_curve"], label="Portfolio OOS",
                          filename="equity_portfolio_OOS.png")
        oos_pair_results = [v["oos_result"] for v in pair_validations]
        plot_portfolio_returns(oos_pair_results, portfolio_oos, label_suffix=" (OOS)")

    # IS vs OOS comparison chart
    plot_is_oos_comparison(summary_df)

    # Save summary CSV
    summary_path = os.path.join(config.OUTPUT_DIR, "is_oos_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved IS/OOS summary → {summary_path}")

    # ------------------------------------------------------------------
    # 7. Save all trade logs (OOS)
    # ------------------------------------------------------------------
    for val in pair_validations:
        a, b = val["ticker_a"], val["ticker_b"]
        tl = val["oos_result"]["trade_log"]
        if not tl.empty:
            tl_path = os.path.join(config.OUTPUT_DIR, f"trade_log_{a}_{b}_OOS.csv")
            tl.to_csv(tl_path, index=False)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"DONE — All outputs saved to ./{config.OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
