# Factor Research

A cross-sectional equity factor research pipeline built for a quant finance portfolio. It implements academic price-based factors from scratch, validates their statistical properties (IC, IR, monotonicity), and evaluates performance with institutional-grade backtesting (transaction costs, execution lag, IS/OOS Sharpe decay).

This project reimplements — independently, on public data — the same validation methodology (IC, IR, monotonicity checks, IS/OOS Sharpe decay) used to build production signal validation infrastructure at Xiyue Investments, applied here to academic factors on US equities.

---

## Project Structure

```
factor_research/
├── config.py                  # Universe, date ranges, factor/backtest parameters
├── data/
│   └── fetch_data.py          # yfinance download + CSV cache
├── factors/
│   ├── base.py                # Abstract Factor class: zscore/rank/nan_rate
│   ├── momentum.py             # 12-1 month momentum
│   ├── reversal.py             # 1-month short-term reversal
│   └── volatility.py           # 60-day realized volatility (low-vol anomaly)
├── analysis/
│   ├── cross_sectional.py     # Rebalance-date scheduling, forward returns
│   ├── ic.py                  # Information Coefficient (Spearman rank IC)
│   ├── ir.py                  # Information Ratio and summary stats
│   └── quintiles.py           # Quintile bucket assignment and returns
├── backtest/
│   ├── engine.py               # Long/short quintile portfolio simulation
│   ├── metrics.py              # Sharpe, drawdown, turnover
│   └── validation.py           # IS/OOS split, Sharpe decay detection
├── visualization/
│   └── tearsheet.py            # IC series, quintile bars, equity curve, IR decay
├── main.py                     # Full pipeline runner
└── outputs/                    # Auto-generated charts and CSVs
```

---

## Theory

| Concept | Description |
|---|---|
| **Cross-sectional z-score** | Per-date, row-wise standardization of factor values across the universe |
| **Information Coefficient (IC)** | Spearman rank correlation between factor value at t and forward return |
| **Information Ratio (IR)** | Mean IC / Std IC — signal quality adjusted for noise |
| **Quintile monotonicity** | Q1 < Q2 < ... < Q5 mean forward return — the sanity check for a real signal |
| **Execution lag** | Signal computed at rebalance date t, portfolio applied at t+1 — no lookahead |
| **Transaction costs** | `cost_bps / 10000` applied to daily portfolio turnover |
| **IS vs OOS** | All parameters fixed ex ante; OOS period never touched during factor design |
| **Sharpe decay** | IS Sharpe − OOS Sharpe; > 0.3 is flagged as a possible overfitting signal |

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
- Download adjusted-close prices from Yahoo Finance (cached to `outputs/prices.csv`)
- Compute momentum, reversal, and volatility factors
- Compute IC/IR and quintile returns for each factor
- Run an IS/OOS long/short backtest with transaction costs and execution lag
- Save all charts and summary CSVs to `outputs/`

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `TICKERS` | 149 liquid large-cap US stocks | Static universe, 11 sectors |
| `IN_SAMPLE_START/END` | 2015-01-01 / 2022-12-31 | In-sample period |
| `OOS_START/END` | 2023-01-01 / 2025-06-30 | Out-of-sample period |
| `DATA_START` | 2013-10-01 | Warmup buffer for the longest factor lookback |
| `MOMENTUM_LOOKBACK_DAYS` / `MOMENTUM_SKIP_DAYS` | 252 / 21 | 12-1 month momentum window |
| `REVERSAL_LOOKBACK_DAYS` | 21 | 1-month reversal window |
| `VOLATILITY_LOOKBACK_DAYS` | 60 | Realized volatility window |
| `N_QUINTILES` | 5 | Cross-sectional buckets |
| `EXECUTION_LAG_DAYS` | 1 | Signal-to-execution lag |
| `TRANSACTION_COST_BPS` | 10 | One-way cost per unit of turnover |
| `SHARPE_DECAY_THRESHOLD` | 0.3 | Overfitting flag cutoff |

---

## Outputs

Per factor (`{factor}` = `momentum_12_1`, `reversal_1m`, `realized_vol_60d`):

| File | Description |
|---|---|
| `outputs/{factor}_ic_series.png` | IC per rebalance date + rolling 12-period mean |
| `outputs/{factor}_quintile_returns.png` | Mean forward return by quintile + long/short bar |
| `outputs/{factor}_equity_curve.png` | Long/short equity curve + drawdown |
| `outputs/{factor}_ir_decay.png` | IS vs OOS Information Ratio |
| `outputs/{factor}_summary.csv` | IC mean/std, IR, IS/OOS Sharpe, Sharpe decay, NaN rate |
| `outputs/factor_summary.csv` | Cross-factor comparison table |
| `outputs/prices.csv` | Cached adjusted-close price panel |

---

## Known Limitations

- **No value or earnings-quality factor.** yfinance exposes only ~4 years of annual fundamentals with no reliable point-in-time history, which would introduce lookahead bias and high NaN rates across a 2015–2025 window. `factors/base.py`'s `Factor` abstract base class is architected so a fundamentals-based factor can be dropped in later without touching the analysis, backtest, or visualization layers — once WRDS/Compustat point-in-time data is available, a new `Factor` subclass with a causal `compute()` is the only change required.
- **Static, non-survivorship-bias-free universe.** The 149-ticker list reflects current large-cap constituents applied retroactively over 2015–2025; delisted or failed names from that window are absent, which biases results upward relative to a true point-in-time S&P 500 membership list.
- **Data-layer swap seam.** `data/fetch_data.py`'s `fetch_prices()` signature (tickers/start/end → wide price DataFrame) is the contract every other module depends on. A future `data/fetch_wrds.py` implementing the same signature (backed by CRSP/Compustat) can replace it without touching factors/, analysis/, backtest/, or visualization/.
- **No discrete trade log.** Because the portfolio is continuously rebalanced quintile weights (not discrete trade entries/exits), `backtest/metrics.py` omits win rate / average trade return / number of trades — these aren't meaningful for this portfolio construction.
- **Weak/noisy signal in this specific sample.** All three factors show small-magnitude, sign-inconsistent IC across 2015–2025 on this large-cap-only universe — consistent with academic findings that factor premia are markedly weaker in large-cap-only samples than in full-market (small-cap-inclusive) samples, and with several known momentum-crash and low-vol-anomaly-inversion periods (2020, 2022) within the sample window. This is treated as a genuine finding, not a pipeline defect — see `outputs/factor_summary.csv`.

---

## Libraries Used

`pandas` · `numpy` · `yfinance` · `scipy` · `matplotlib` · `seaborn` · `statsmodels` · `scikit-learn`
