# PROJECT CONTEXT — first_signal (Pairs Trading Strategy)
> AI-facing system prompt. Paste this at the start of any conversation to give an AI full context about this codebase.

---

## What This Project Is

`first_signal` is a **quantitative pairs trading system** written entirely in Python. It implements a complete, production-style pipeline for a statistical arbitrage strategy: data ingestion → cointegration testing → signal generation → parameter optimization → backtesting → in-sample/out-of-sample validation → visualization. It was built as a quant finance portfolio project by Leo Li.

The strategy exploits **mean reversion** in the spread between two cointegrated equities. When the spread deviates significantly from its historical mean (measured by a rolling z-score), the strategy enters a long/short position and exits when the spread reverts.

---

## Repository Layout

```
first_signal/
├── config.py                  # Single source of truth for all parameters
├── main.py                    # Full pipeline orchestrator (run this)
├── requirements.txt           # Python dependencies
├── data/
│   └── fetch_data.py          # yfinance download + CSV cache + macOS proxy detection
├── analysis/
│   ├── cointegration.py       # Engle-Granger test + ADF confirmation on all pairs
│   ├── spread.py              # OLS hedge ratio, spread, rolling z-score, signal generation
│   └── optimization.py        # Grid search over entry/exit thresholds (maximize IS Sharpe)
├── backtest/
│   ├── engine.py              # Simulation loop, transaction costs, equity curve, trade log
│   ├── metrics.py             # Sharpe, drawdown, win rate, turnover, annualized return/vol
│   └── validation.py          # IS vs OOS split, Sharpe decay detection, portfolio aggregation
├── visualization/
│   └── plots.py               # All charts: heatmaps, spread, z-score, equity, sensitivity, IS/OOS bar
└── outputs/                   # Auto-generated (do not edit manually)
    ├── prices.csv             # Cached price data
    ├── candidate_pairs.csv    # All cointegrated pairs ranked by EG p-value
    ├── optimized_pairs.csv    # Top pairs with best entry/exit thresholds
    ├── is_oos_summary.csv     # IS vs OOS metrics for every pair
    ├── trade_log_A_B_IS.csv   # Per-pair trade logs (in-sample)
    ├── trade_log_A_B_OOS.csv  # Per-pair trade logs (out-of-sample)
    └── *.png                  # All charts
```

---

## Configuration (`config.py`) — All Tunable Parameters

```python
# Ticker universe: 20 liquid US equities across 5 sectors
TICKERS = [
    "XOM", "CVX", "COP", "SLB",          # Energy
    "GS", "MS", "JPM", "BAC",             # Financials
    "KO", "PEP", "MCD", "YUM",            # Consumer Staples
    "MSFT", "GOOGL", "META", "AMZN",      # Technology
    "NEE", "DUK", "SO", "AEP",            # Utilities
]

IN_SAMPLE_START  = "2015-01-01"   # 8-year training window
IN_SAMPLE_END    = "2022-12-31"
OOS_START        = "2023-01-01"   # 2.5-year forward validation
OOS_END          = "2025-06-30"

COINT_PVALUE_THRESHOLD = 0.05     # Engle-Granger significance cutoff
LOOKBACK_WINDOW  = 60             # Rolling z-score window (trading days)
ENTRY_ZSCORE     = 2.0            # Default entry threshold (|z| > entry → open)
EXIT_ZSCORE      = 0.5            # Default exit threshold  (|z| < exit  → close)
STOP_LOSS_ZSCORE = 3.5            # Emergency stop-loss
TRANSACTION_COST_BPS = 10         # One-way cost per trade (10 bps = 0.10%)
INITIAL_CAPITAL  = 1_000_000      # USD

# Optimization grid
ENTRY_RANGE = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
EXIT_RANGE  = [0.0, 0.25, 0.5, 0.75]
```

---

## Pipeline Steps (`main.py`)

1. **Fetch data** — Downloads adjusted-close prices via yfinance for all tickers over the full IS+OOS date range. Caches to `outputs/prices.csv` to avoid repeated API calls. Splits into IS and OOS DataFrames.

2. **Cointegration testing** (IS only) — Runs Engle-Granger two-step test on all N*(N-1)/2 pair combinations (190 pairs for 20 tickers). For pairs with EG p-value < 0.05, also runs ADF test on the OLS residual spread. Returns a ranked `candidate_pairs.csv`.

3. **Parameter optimization** (IS only) — For the top 10 candidate pairs, runs a grid search over all valid (entry, exit) combinations where exit < entry. Maximizes in-sample Sharpe ratio. Saves `optimized_pairs.csv` and sensitivity heatmaps for the top 3 pairs.

4. **Spread & z-score charts** — Builds spread and z-score visualizations for the top 5 pairs using their optimized thresholds.

5. **In-sample backtest** — Runs the full backtest for each optimized pair on IS data. Builds an equal-weight portfolio. Saves equity curves and trade logs.

6. **IS vs OOS validation** — Applies the IS-fitted hedge ratio and optimized thresholds (frozen, no re-fitting) to the OOS period. Computes Sharpe decay = IS Sharpe − OOS Sharpe. Large decay signals overfitting.

7. **Save all outputs** — All charts and CSVs written to `outputs/`.

---

## Module Details

### `data/fetch_data.py`
- `fetch_prices(tickers, start, end, cache_file, force_download)` → `pd.DataFrame` of adjusted-close prices
- `log_prices(prices)` → natural log of price DataFrame
- `log_returns(prices)` → log-return DataFrame (first row NaN)
- Includes macOS HTTPS proxy auto-detection via `networksetup` (for VPN environments)

### `analysis/cointegration.py`
- `run_cointegration_tests(log_px, pvalue_threshold)` → `(pval_matrix, candidates_df)`
  - Uses `statsmodels.tsa.stattools.coint` (Engle-Granger)
  - Confirms with `adfuller` on OLS residual
  - Returns symmetric p-value matrix + ranked DataFrame with columns: `ticker_a, ticker_b, eg_pvalue, adf_stat, adf_pvalue, hedge_ratio`
- `_ols_hedge_ratio(series_a, series_b)` → float β (via `np.linalg.lstsq`)
- `adf_summary(spread, label)` → dict with stat, p-value, critical values

### `analysis/spread.py`
- `compute_hedge_ratio_static(log_a, log_b)` → float β (via `sklearn.LinearRegression`)
- `compute_spread(log_a, log_b, hedge_ratio)` → `spread = log_a − β·log_b`
- `compute_zscore(spread, window)` → rolling z-score: `(spread − roll_mean) / roll_std`
- `generate_signals(zscore, entry, exit_, stop_loss)` → `pd.Series` of {-1, 0, +1}
  - **Signal convention:** +1 = long spread (long A, short B); -1 = short spread; 0 = flat
  - Entry: z < −entry → +1; z > +entry → −1
  - Exit: |z| < exit_ → 0; |z| > stop_loss → 0 (emergency)
  - State machine: iterates day-by-day, maintains current position
- `build_pair_signals(log_px, ticker_a, ticker_b, hedge_ratio, window, entry, exit_, stop_loss)` → `(pair_df, hedge_ratio)`
  - Returns DataFrame with columns: `log_a, log_b, spread, zscore, signal`

### `analysis/optimization.py`
- `optimize_thresholds(log_px, ticker_a, ticker_b, hedge_ratio, entry_range, exit_range, window)` → `(best_params, results_df, sharpe_pivot)`
  - Grid search: skips combinations where exit >= entry
  - Returns best params dict, full results DataFrame, pivot table for heatmap
- `optimize_all_pairs(log_px, candidates, ...)` → DataFrame with columns: `ticker_a, ticker_b, hedge_ratio, best_entry, best_exit, best_sharpe, n_trades`

### `backtest/engine.py`
- `run_backtest(pair_df, ticker_a, ticker_b, hedge_ratio, cost_bps, initial_capital)` → dict
  - `spread_return = Δlog_a − β·Δlog_b`
  - `strat_return = signal.shift(1) × spread_return` (1-day execution lag)
  - `tc_series = |Δsignal| × cost_per_trade`
  - `net_return = strat_return − tc_series`
  - `equity_curve = initial_capital × cumprod(1 + net_return)`
  - Returns: `{equity_curve, daily_returns, trade_log, metrics}`
- `_build_trade_log(signals, net_return, ticker_a, ticker_b)` → list of trade dicts
  - Columns: `ticker_a, ticker_b, direction, entry_date, exit_date, trade_return, holding_days`
  - Open trades at period end are closed at last date
- `run_portfolio_backtest(pair_results, initial_capital)` → dict
  - Equal-weight: `portfolio_return = mean(all pair daily_returns)`

### `backtest/metrics.py`
All functions accept `pd.Series`:
- `annualized_return(daily_returns)` → compound annualized return (log-return based)
- `annualized_volatility(daily_returns)` → `std × √252`
- `sharpe_ratio(daily_returns, risk_free_rate=0.0)` → annualized Sharpe
- `max_drawdown(equity_curve)` → peak-to-trough fraction (negative float)
- `drawdown_series(equity_curve)` → full drawdown time series
- `win_rate(trade_returns)` → fraction of trades with positive return
- `average_trade_return(trade_returns)` → mean return per trade
- `turnover(signals)` → mean |Δposition| / 2
- `compute_all_metrics(daily_returns, equity_curve, trade_returns, signals, label)` → dict

### `backtest/validation.py`
- `validate_pair(log_px_is, log_px_oos, ticker_a, ticker_b, best_entry, best_exit, window)` → dict
  - Fits hedge ratio on IS only; applies frozen params to both IS and OOS
  - Returns: `{ticker_a, ticker_b, hedge_ratio, is_result, oos_result, comparison_df}`
- `validate_all_pairs(log_px_is, log_px_oos, optimized_pairs, window)` → `(pair_validations, summary_df, portfolio_is, portfolio_oos)`
  - `summary_df` columns: `pair, is_sharpe, oos_sharpe, is_return, oos_return, is_maxdd, oos_maxdd, sharpe_decay`
  - Sorted by `oos_sharpe` descending

### `visualization/plots.py`
All functions save to `outputs/` by default (matplotlib Agg backend, 150 dpi):
- `plot_coint_heatmap(pval_matrix)` → `coint_heatmap.png`
- `plot_spread(pair_df, ticker_a, ticker_b)` → `spread_A_B.png`
- `plot_zscore(pair_df, ticker_a, ticker_b, entry, exit_)` → `zscore_A_B.png` (shaded position regions)
- `plot_equity_curve(equity_curve, label, filename)` → 2-panel: equity + drawdown
- `plot_sensitivity_heatmap(sharpe_pivot, ticker_a, ticker_b)` → `sensitivity_A_B.png`
- `plot_is_oos_comparison(summary_df)` → `is_oos_comparison.png` (grouped bar chart)
- `plot_portfolio_returns(pair_results, portfolio_result, label_suffix)` → `portfolio_returns_(IS/OOS).png`
- `save_all_outputs(...)` → convenience wrapper

---

## Current Results (from `outputs/`)

### Optimized Pairs (ranked by IS Sharpe)
| Pair | Hedge Ratio | Best Entry | Best Exit | IS Sharpe | # Trades |
|---|---|---|---|---|---|
| MSFT/NEE | 1.436 | 2.5 | 0.75 | 0.91 | 24 |
| KO/DUK | 0.943 | 2.0 | 0.25 | 0.90 | 36 |
| YUM/DUK | 1.431 | 1.75 | 0.50 | 0.82 | 49 |
| KO/SO | 0.773 | 1.75 | 0.75 | 0.75 | 53 |
| PEP/SO | 0.969 | 1.25 | 0.25 | 0.75 | 56 |
| MCD/AEP | 1.416 | 1.0 | 0.50 | 0.72 | 83 |
| MCD/DUK | 1.653 | 2.25 | 0.75 | 0.68 | 35 |
| MSFT/DUK | 3.090 | 2.5 | 0.25 | 0.62 | 22 |
| KO/PEP | 0.781 | 1.0 | 0.25 | 0.43 | 52 |
| PEP/DUK | 1.174 | 2.25 | 0.25 | 0.40 | 31 |

### IS vs OOS Validation (ranked by OOS Sharpe)
| Pair | IS Sharpe | OOS Sharpe | IS Return | OOS Return | IS MaxDD | OOS MaxDD | Sharpe Decay |
|---|---|---|---|---|---|---|---|
| MCD/DUK | 0.68 | **0.84** | 14.3% | 12.3% | -31.7% | -20.2% | -0.16 ✅ |
| KO/SO | 0.75 | **0.76** | 8.5% | 6.2% | -13.0% | -8.9% | -0.01 ✅ |
| YUM/DUK | 0.82 | 0.64 | 19.0% | 11.3% | -32.2% | -25.9% | +0.18 |
| MCD/AEP | 0.72 | 0.63 | 19.8% | 14.1% | -26.8% | -26.3% | +0.09 |
| KO/DUK | 0.90 | 0.37 | 11.2% | 3.5% | -12.3% | -11.1% | +0.53 |
| MSFT/DUK | 0.62 | 0.24 | 30.1% | 7.4% | -48.5% | -24.4% | +0.38 |
| MSFT/NEE | **0.91** | 0.00 | 15.4% | 0.1% | -19.6% | -39.9% | +0.90 ⚠️ |
| PEP/DUK | 0.40 | -0.25 | 5.5% | -3.4% | -21.4% | -26.6% | +0.65 ⚠️ |
| PEP/SO | 0.75 | -0.42 | 11.8% | -6.1% | -26.2% | -37.1% | +1.16 ⚠️ |
| KO/PEP | 0.43 | -1.16 | 4.8% | -11.3% | -18.4% | -34.1% | +1.59 ⚠️ |

**Key insight:** MCD/DUK and KO/SO are the most robust pairs — they actually *improve* OOS. MSFT/NEE, PEP/SO, and KO/PEP show severe overfitting (large Sharpe decay).

---

## Key Design Decisions & Conventions

1. **No lookahead bias:** Hedge ratio is always fitted on IS data only. OOS uses frozen IS parameters.
2. **1-day execution lag:** `signal.shift(1)` in the backtest engine — today's signal executes tomorrow.
3. **Log prices throughout:** All spread and z-score calculations use `log(price)`, not raw price.
4. **Static hedge ratio:** OLS β computed once on IS period (not rolling). This is intentional for simplicity and to avoid overfitting.
5. **Transaction costs:** Applied as `|Δsignal| × cost_bps/10000` — charged on every position change (open and close).
6. **Stop-loss:** Closes position if |z| > 3.5, regardless of direction.
7. **Equal-weight portfolio:** Portfolio return = simple mean of all pair daily returns.
8. **CSV price cache:** `outputs/prices.csv` is loaded on subsequent runs to avoid API calls. Use `force_download=True` to refresh.
9. **macOS proxy:** `fetch_data.py` auto-detects the system HTTPS proxy via `networksetup -getsecurewebproxy Wi-Fi` and sets environment variables before importing yfinance.
10. **Matplotlib Agg backend:** All plots use the non-interactive `Agg` backend so the script runs headlessly without a display.

---

## Dependencies (`requirements.txt`)

```
pandas>=2.0
numpy>=1.24
yfinance>=0.2
matplotlib>=3.7
seaborn>=0.12
statsmodels>=0.14
scipy>=1.10
scikit-learn>=1.3
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (downloads data, tests pairs, optimizes, backtests, saves outputs)
python main.py
```

All outputs are written to `outputs/`. The pipeline prints step-by-step progress to stdout.

---

## Common Extension Points

- **Add tickers:** Edit `TICKERS` in `config.py`. Re-run `main.py` with `force_download=True`.
- **Change date range:** Edit `IN_SAMPLE_START/END` and `OOS_START/END` in `config.py`.
- **Adjust strategy params:** Edit `ENTRY_ZSCORE`, `EXIT_ZSCORE`, `STOP_LOSS_ZSCORE`, `LOOKBACK_WINDOW`, `TRANSACTION_COST_BPS`.
- **Expand optimization grid:** Edit `ENTRY_RANGE` and `EXIT_RANGE` in `config.py`.
- **Rolling hedge ratio:** Replace `compute_hedge_ratio_static` in `spread.py` with a rolling OLS implementation.
- **Add new metrics:** Add functions to `backtest/metrics.py` and include them in `compute_all_metrics`.
- **New visualizations:** Add functions to `visualization/plots.py` following the `_savefig` pattern.
- **Walk-forward validation:** Extend `backtest/validation.py` with a rolling IS/OOS window loop.
