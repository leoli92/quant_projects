"""
visualization/plots.py
-----------------------
All charts for the Pairs Trading project.

Functions
---------
plot_coint_heatmap        – p-value heatmap of all pairs
plot_spread               – spread time series for a pair
plot_zscore               – z-score with entry/exit threshold bands
plot_equity_curve         – equity curve (single pair or portfolio)
plot_drawdown             – drawdown time series
plot_sensitivity_heatmap  – Sharpe vs (entry, exit) grid
plot_is_oos_comparison    – bar chart comparing IS vs OOS metrics
plot_portfolio_returns    – cumulative returns for all pairs + portfolio
save_all_outputs          – convenience wrapper that saves everything
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from backtest.metrics import drawdown_series

sns.set_theme(style="darkgrid", palette="muted")
OUTPUT_DIR = config.OUTPUT_DIR


def _savefig(fig, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")


# ---------------------------------------------------------------------------
# 1. Cointegration p-value heatmap
# ---------------------------------------------------------------------------
def plot_coint_heatmap(pval_matrix: pd.DataFrame, save: bool = True):
    """Heatmap of Engle-Granger p-values for all pairs."""
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = pval_matrix.isna()
    sns.heatmap(
        pval_matrix.astype(float),
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "EG p-value"},
    )
    ax.set_title("Engle-Granger Cointegration p-values\n(green = more cointegrated)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save:
        _savefig(fig, "coint_heatmap.png")
    return fig


# ---------------------------------------------------------------------------
# 2. Spread chart
# ---------------------------------------------------------------------------
def plot_spread(
    pair_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    save: bool = True,
):
    """Plot the spread (log_a − β·log_b) over time."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(pair_df.index, pair_df["spread"], color="steelblue", linewidth=0.8, label="Spread")
    ax.axhline(pair_df["spread"].mean(), color="red", linestyle="--", linewidth=1, label="Mean")
    ax.set_title(f"Spread: {ticker_a} − β·{ticker_b}", fontsize=13)
    ax.set_ylabel("Spread (log units)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save:
        _savefig(fig, f"spread_{ticker_a}_{ticker_b}.png")
    return fig


# ---------------------------------------------------------------------------
# 3. Z-score chart with entry/exit bands
# ---------------------------------------------------------------------------
def plot_zscore(
    pair_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    entry: float = None,
    exit_: float = None,
    save: bool = True,
):
    """Plot rolling z-score with entry/exit threshold lines and signal markers."""
    entry = entry if entry is not None else config.ENTRY_ZSCORE
    exit_ = exit_ if exit_ is not None else config.EXIT_ZSCORE

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(pair_df.index, pair_df["zscore"], color="navy", linewidth=0.8, label="Z-score")

    # Threshold bands
    for sign, color in [(1, "green"), (-1, "red")]:
        ax.axhline(sign * entry, color=color, linestyle="--", linewidth=1,
                   label=f"Entry ±{entry}")
        ax.axhline(sign * exit_, color=color, linestyle=":", linewidth=0.8,
                   label=f"Exit ±{exit_}")
    ax.axhline(0, color="black", linewidth=0.5)

    # Shade active positions
    if "signal" in pair_df.columns:
        long_mask  = pair_df["signal"] == 1
        short_mask = pair_df["signal"] == -1
        ax.fill_between(pair_df.index, pair_df["zscore"].min(), pair_df["zscore"].max(),
                        where=long_mask,  alpha=0.08, color="green", label="Long spread")
        ax.fill_between(pair_df.index, pair_df["zscore"].min(), pair_df["zscore"].max(),
                        where=short_mask, alpha=0.08, color="red",   label="Short spread")

    ax.set_title(f"Z-score: {ticker_a}/{ticker_b}", fontsize=13)
    ax.set_ylabel("Z-score")
    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save:
        _savefig(fig, f"zscore_{ticker_a}_{ticker_b}.png")
    return fig


# ---------------------------------------------------------------------------
# 4. Equity curve
# ---------------------------------------------------------------------------
def plot_equity_curve(
    equity_curve: pd.Series,
    label: str = "Strategy",
    save: bool = True,
    filename: str = None,
):
    """Plot cumulative equity curve."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    # Equity
    axes[0].plot(equity_curve.index, equity_curve.values, color="steelblue",
                 linewidth=1, label=label)
    axes[0].set_title(f"Equity Curve — {label}", fontsize=13)
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].legend()

    # Drawdown
    dd = drawdown_series(equity_curve)
    axes[1].fill_between(dd.index, dd.values, 0, color="red", alpha=0.4)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_ylim(dd.min() * 1.1, 0.05)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save:
        fname = filename or f"equity_{label.replace('/', '_')}.png"
        _savefig(fig, fname)
    return fig


# ---------------------------------------------------------------------------
# 5. Sensitivity heatmap (Sharpe vs entry/exit thresholds)
# ---------------------------------------------------------------------------
def plot_sensitivity_heatmap(
    sharpe_pivot: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    save: bool = True,
):
    """Heatmap of Sharpe ratio across (entry, exit) threshold grid."""
    if sharpe_pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        sharpe_pivot.astype(float),
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Sharpe Ratio"},
    )
    ax.set_title(f"Sensitivity Analysis — {ticker_a}/{ticker_b}\nSharpe vs Entry/Exit Thresholds",
                 fontsize=12)
    ax.set_xlabel("Exit Z-score")
    ax.set_ylabel("Entry Z-score")
    plt.tight_layout()
    if save:
        _savefig(fig, f"sensitivity_{ticker_a}_{ticker_b}.png")
    return fig


# ---------------------------------------------------------------------------
# 6. IS vs OOS comparison bar chart
# ---------------------------------------------------------------------------
def plot_is_oos_comparison(summary_df: pd.DataFrame, save: bool = True):
    """Grouped bar chart of IS vs OOS Sharpe ratio for each pair."""
    if summary_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(summary_df) * 1.2), 5))
    x = range(len(summary_df))
    width = 0.35

    bars_is  = ax.bar([i - width / 2 for i in x], summary_df["is_sharpe"],
                      width, label="In-Sample",      color="steelblue", alpha=0.85)
    bars_oos = ax.bar([i + width / 2 for i in x], summary_df["oos_sharpe"],
                      width, label="Out-of-Sample",  color="coral",     alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary_df["pair"].tolist(), rotation=30, ha="right")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("In-Sample vs Out-of-Sample Sharpe Ratio by Pair", fontsize=13)
    ax.legend()
    plt.tight_layout()
    if save:
        _savefig(fig, "is_oos_comparison.png")
    return fig


# ---------------------------------------------------------------------------
# 7. Portfolio cumulative returns (all pairs + aggregate)
# ---------------------------------------------------------------------------
def plot_portfolio_returns(
    pair_results: list,
    portfolio_result: dict,
    save: bool = True,
    label_suffix: str = "",
):
    """
    Plot cumulative returns for each pair and the equal-weighted portfolio.

    pair_results    : list of dicts from run_backtest()
    portfolio_result: dict from run_portfolio_backtest()
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for r in pair_results:
        m = r["metrics"]
        pair_label = m.get("label", "pair")
        cum_ret = (1 + r["daily_returns"]).cumprod() - 1
        ax.plot(cum_ret.index, cum_ret.values, linewidth=0.7, alpha=0.6, label=pair_label)

    if portfolio_result:
        port_cum = (1 + portfolio_result["daily_returns"]).cumprod() - 1
        ax.plot(port_cum.index, port_cum.values, linewidth=2.5,
                color="black", label=f"Portfolio{label_suffix}")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"Cumulative Returns — All Pairs & Portfolio{label_suffix}", fontsize=13)
    ax.set_ylabel("Cumulative Return")
    ax.legend(fontsize=7, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save:
        fname = f"portfolio_returns{label_suffix.replace(' ', '_')}.png"
        _savefig(fig, fname)
    return fig


# ---------------------------------------------------------------------------
# 8. Convenience: save all outputs for a validation run
# ---------------------------------------------------------------------------
def save_all_outputs(
    pval_matrix: pd.DataFrame,
    pair_validations: list,
    summary_df: pd.DataFrame,
    portfolio_is: dict,
    portfolio_oos: dict,
    sensitivity_data: dict = None,
):
    """
    Generate and save every chart and table.

    Parameters
    ----------
    pval_matrix       : from cointegration.run_cointegration_tests()
    pair_validations  : list of dicts from validation.validate_all_pairs()
    summary_df        : IS/OOS summary DataFrame
    portfolio_is/oos  : portfolio backtest dicts
    sensitivity_data  : dict mapping (a, b) → sharpe_pivot DataFrame
    """
    # 1. Cointegration heatmap
    plot_coint_heatmap(pval_matrix)

    # 2. Per-pair charts
    for val in pair_validations:
        a, b = val["ticker_a"], val["ticker_b"]
        is_df  = val["is_result"]
        oos_df = val["oos_result"]

        # Reconstruct pair_df from IS result for spread/zscore plots
        # (we need to re-run spread construction — stored in engine output)
        # We'll use the equity curve index as a proxy; actual spread data
        # must be passed if needed. Here we skip if not available.

        # IS equity curve
        plot_equity_curve(
            is_df["equity_curve"],
            label=f"{a}/{b} IS",
            filename=f"equity_{a}_{b}_IS.png",
        )
        # OOS equity curve
        plot_equity_curve(
            oos_df["equity_curve"],
            label=f"{a}/{b} OOS",
            filename=f"equity_{a}_{b}_OOS.png",
        )

    # 3. Sensitivity heatmaps
    if sensitivity_data:
        for (a, b), pivot in sensitivity_data.items():
            plot_sensitivity_heatmap(pivot, a, b)

    # 4. IS vs OOS comparison
    plot_is_oos_comparison(summary_df)

    # 5. Portfolio cumulative returns
    is_pair_results  = [v["is_result"]  for v in pair_validations]
    oos_pair_results = [v["oos_result"] for v in pair_validations]

    if is_pair_results and portfolio_is:
        plot_portfolio_returns(is_pair_results,  portfolio_is,  label_suffix=" (IS)")
    if oos_pair_results and portfolio_oos:
        plot_portfolio_returns(oos_pair_results, portfolio_oos, label_suffix=" (OOS)")

    # 6. Save summary table as CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "is_oos_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[plots] Saved → {summary_path}")

    print("[plots] All outputs saved.")
