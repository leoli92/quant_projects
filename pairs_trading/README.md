# Pairs Trading Strategy

A quantitative pairs trading project built for a quant finance portfolio. It identifies cointegrated equity pairs, constructs mean-reversion spread signals, and backtests a long/short strategy with full in-sample / out-of-sample validation.

---

## Project Structure

```
pairs_trading/
├── config.py                  # Universe, date ranges, strategy parameters
├── data/
│   └── fetch_data.py          # yfinance download + CSV cache
├── analysis/
│   ├── cointegration.py       # Engle-Granger test, ADF test, p-value table
│   ├── spread.py              # OLS hedge ratio, spread, rolling z-score, signals
│   └── optimization.py        # Grid search over entry/exit thresholds
├── backtest/
│   ├── engine.py              # Simulation loop, transaction costs, trade log
│   ├── metrics.py             # Sharpe, drawdown, win rate, turnover
│   └── validation.py          # IS vs OOS split and comparison
├── visualization/
│   └── plots.py               # All charts (heatmaps, equity curves, etc.)
├── main.py                    # Full pipeline runner
├── requirements.txt
└── outputs/                   # Auto-generated charts, CSVs, trade logs
```

---

## Theory

| Concept | Description |
|---|---|
| **Log prices & returns** | `log(P_t)` linearises compounding; `Δlog(P)` ≈ simple return |
| **Correlation vs cointegration** | Correlation measures co-movement; cointegration means a stationary linear combination exists |
| **Engle-Granger test** | Two-step: OLS regression → ADF test on residuals |
| **ADF test** | Augmented Dickey-Fuller — tests for unit root (non-stationarity) |
| **Hedge ratio β** | OLS slope: `log(A) = α + β·log(B) + ε` |
| **Z-score** | `z = (spread − μ_roll) / σ_roll` — normalised mean-reversion signal |
| **Mean reversion** | Enter when `|z| > entry`, exit when `|z| < exit` |
| **Transaction costs** | Applied as `cost_bps / 10000` on every position change |
| **Sharpe ratio** | `(E[r] − r_f) / σ(r) × √252` |
| **Max drawdown** | Peak-to-trough decline in equity curve |
| **IS vs OOS** | Fit parameters in-sample; evaluate on unseen out-of-sample data |
| **Overfitting** | Large IS/OOS Sharpe gap signals over-optimisation |

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python main.py
```

This will:
- Download price data from Yahoo Finance (cached to `outputs/prices.csv`)
- Test all pairs for cointegration
- Optimise entry/exit thresholds per pair
- Backtest in-sample and out-of-sample
- Save all charts and CSVs to `outputs/`

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `TICKERS` | 20 liquid US stocks | Ticker universe (4 sectors × 5 stocks) |
| `IN_SAMPLE_START/END` | 2015-01-01 / 2022-12-31 | Training period |
| `OOS_START/END` | 2023-01-01 / 2025-06-30 | Validation period |
| `COINT_PVALUE_THRESHOLD` | 0.05 | EG p-value cutoff |
| `LOOKBACK_WINDOW` | 60 days | Rolling z-score window |
| `ENTRY_ZSCORE` | 2.0 | Default entry threshold |
| `EXIT_ZSCORE` | 0.5 | Default exit threshold |
| `STOP_LOSS_ZSCORE` | 3.5 | Emergency stop-loss |
| `TRANSACTION_COST_BPS` | 10 bps | One-way cost per trade |

---

## Outputs

| File | Description |
|---|---|
| `outputs/coint_heatmap.png` | EG p-value heatmap for all pairs |
| `outputs/candidate_pairs.csv` | Ranked cointegrated pairs table |
| `outputs/spread_A_B.png` | Spread time series per pair |
| `outputs/zscore_A_B.png` | Z-score chart with entry/exit bands |
| `outputs/sensitivity_A_B.png` | Sharpe sensitivity heatmap |
| `outputs/equity_A_B_IS.png` | IS equity curve + drawdown |
| `outputs/equity_A_B_OOS.png` | OOS equity curve + drawdown |
| `outputs/equity_portfolio_IS.png` | Portfolio IS equity curve |
| `outputs/equity_portfolio_OOS.png` | Portfolio OOS equity curve |
| `outputs/portfolio_returns_(IS).png` | All pairs + portfolio cumulative returns |
| `outputs/is_oos_comparison.png` | IS vs OOS Sharpe bar chart |
| `outputs/is_oos_summary.csv` | IS/OOS metrics table |
| `outputs/trade_log_A_B_IS.csv` | Trade log per pair (IS) |
| `outputs/trade_log_A_B_OOS.csv` | Trade log per pair (OOS) |

---

## Key Metrics Reported

- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Average Trade Return
- Number of Trades
- Turnover

---

## Libraries Used

`pandas` · `numpy` · `yfinance` · `statsmodels` · `scikit-learn` · `matplotlib` · `seaborn` · `scipy`
