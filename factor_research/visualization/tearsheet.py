"""
visualization/tearsheet.py
----------------------------
Per-factor tearsheet: IC time series, quintile return bars, equity curve +
drawdown, IS/OOS IR comparison, and a summary CSV.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from analysis.ic import rolling_ic_mean
from backtest.metrics import drawdown_series

sns.set_theme(style="darkgrid", palette="muted")
OUTPUT_DIR = config.OUTPUT_DIR


def _savefig(fig, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[tearsheet] Saved -> {path}")


# ---------------------------------------------------------------------------
# 1. IC time series
# ---------------------------------------------------------------------------
def plot_ic_series(ic_series: pd.Series, factor_name: str, freq: str, save: bool = True):
    """IC per rebalance date with a rolling 12-period mean overlay."""
    rolling = rolling_ic_mean(ic_series, freq)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(ic_series.index, ic_series.values, width=10 if freq == "M" else 4,
           color="steelblue", alpha=0.5, label="IC (per rebalance)")
    ax.plot(rolling.index, rolling.values, color="firebrick", linewidth=1.6,
            label="Rolling 12-period mean")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Information Coefficient — {factor_name}", fontsize=13)
    ax.set_ylabel("Spearman rank IC")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save:
        _savefig(fig, f"{factor_name}_ic_series.png")
    return fig


# ---------------------------------------------------------------------------
# 2. Quintile return bar chart
# ---------------------------------------------------------------------------
def plot_quintile_bars(quintile_ret: pd.DataFrame, factor_name: str, save: bool = True):
    """Mean forward return per quintile bucket + long/short bar. Should be
    roughly monotonically ordered Q1 -> Q5 for a useful factor."""
    n_q = config.N_QUINTILES
    means = quintile_ret[list(range(1, n_q + 1))].mean()
    long_short = quintile_ret["long_short"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("muted", n_q)
    ax.bar([str(q) for q in range(1, n_q + 1)], means.values, color=colors)
    ax.bar(["Long-Short"], [long_short], color="firebrick", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Mean Forward Return by Quintile — {factor_name}", fontsize=13)
    ax.set_ylabel("Mean forward return (log)")
    ax.set_xlabel("Quintile (1=lowest factor value, 5=highest)")
    plt.tight_layout()
    if save:
        _savefig(fig, f"{factor_name}_quintile_returns.png")
    return fig


# ---------------------------------------------------------------------------
# 3. Equity curve + drawdown
# ---------------------------------------------------------------------------
def plot_equity_curve(equity_curve: pd.Series, label: str, save: bool = True, filename: str = None):
    """Two-panel cumulative equity curve + drawdown."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(equity_curve.index, equity_curve.values, color="steelblue",
                 linewidth=1, label=label)
    axes[0].set_title(f"Equity Curve — {label}", fontsize=13)
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].legend()

    dd = drawdown_series(equity_curve)
    axes[1].fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.4)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_ylim(dd.min() * 1.1, 0.05)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save:
        fname = filename or f"equity_{label.replace('/', '_').replace(' ', '_')}.png"
        _savefig(fig, fname)
    return fig


# ---------------------------------------------------------------------------
# 4. IS vs OOS IR decay bar chart
# ---------------------------------------------------------------------------
def plot_ir_decay(is_summary: dict, oos_summary: dict, factor_name: str, save: bool = True):
    """Grouped bar: IS IR vs OOS IR (mirrors an IS/OOS Sharpe comparison)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["In-Sample", "Out-of-Sample"]
    values = [is_summary["ir"], oos_summary["ir"]]
    colors = ["steelblue", "coral"]

    ax.bar(labels, values, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Information Ratio — IS vs OOS — {factor_name}", fontsize=13)
    ax.set_ylabel("Information Ratio")
    plt.tight_layout()
    if save:
        _savefig(fig, f"{factor_name}_ir_decay.png")
    return fig


# ---------------------------------------------------------------------------
# 5. Per-factor summary CSV
# ---------------------------------------------------------------------------
def save_factor_summary_csv(validation_result: dict, path: str):
    """One-row CSV: ic_mean, ic_std, ir, is_sharpe, oos_sharpe, sharpe_decay, nan_rate."""
    is_ic = validation_result["ic_summary_is"]
    row = {
        "factor": validation_result["factor_name"],
        "ic_mean": is_ic["ic_mean"],
        "ic_std": is_ic["ic_std"],
        "ir": is_ic["ir"],
        "is_sharpe": validation_result["is_metrics"]["sharpe_ratio"],
        "oos_sharpe": validation_result["oos_metrics"]["sharpe_ratio"],
        "sharpe_decay": validation_result["sharpe_decay"],
        "nan_rate": is_ic["nan_rate"],
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    print(f"[tearsheet] Saved -> {path}")


# ---------------------------------------------------------------------------
# 6. Orchestrator
# ---------------------------------------------------------------------------
def build_tearsheet(factor, validation_result: dict):
    """Generate and save every chart + summary CSV for one factor."""
    name = factor.name

    plot_ic_series(validation_result["ic_series"], name, factor.rebalance_freq)
    plot_quintile_bars(validation_result["quintile_returns"], name)

    full_equity = pd.concat([
        validation_result["is_result"]["equity_curve"],
        validation_result["oos_result"]["equity_curve"]
        / validation_result["oos_result"]["equity_curve"].iloc[0]
        * validation_result["is_result"]["equity_curve"].iloc[-1],
    ])
    plot_equity_curve(full_equity, label=name, filename=f"{name}_equity_curve.png")

    plot_ir_decay(validation_result["ic_summary_is"], validation_result["ic_summary_oos"], name)

    save_factor_summary_csv(validation_result, os.path.join(OUTPUT_DIR, f"{name}_summary.csv"))
