"""
config.py
---------
Central configuration for the Pairs Trading Strategy project.
"""

# ---------------------------------------------------------------------------
# Ticker Universe
# ---------------------------------------------------------------------------
# Curated set of liquid US equities grouped by sector for natural pairing.
TICKERS = [
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Financials
    "GS", "MS", "JPM", "BAC",
    # Consumer Staples
    "KO", "PEP", "MCD", "YUM",
    # Technology
    "MSFT", "GOOGL", "META", "AMZN",
    # Utilities
    "NEE", "DUK", "SO", "AEP",
]

# ---------------------------------------------------------------------------
# Date Ranges
# ---------------------------------------------------------------------------
# In-sample: used for cointegration testing, parameter optimisation, fitting
# Out-of-sample: used for forward validation only
IN_SAMPLE_START  = "2015-01-01"
IN_SAMPLE_END    = "2022-12-31"
OOS_START        = "2023-01-01"
OOS_END          = "2025-06-30"

# ---------------------------------------------------------------------------
# Cointegration Testing
# ---------------------------------------------------------------------------
COINT_PVALUE_THRESHOLD = 0.05   # pairs with p-value below this are candidates

# ---------------------------------------------------------------------------
# Spread / Signal Parameters
# ---------------------------------------------------------------------------
LOOKBACK_WINDOW  = 60           # rolling window (trading days) for z-score
ENTRY_ZSCORE     = 2.0          # open a position when |z| > entry
EXIT_ZSCORE      = 0.5          # close a position when |z| < exit
STOP_LOSS_ZSCORE = 3.5          # emergency stop-loss threshold

# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
TRANSACTION_COST_BPS = 10       # one-way cost in basis points (10 bps = 0.10 %)
INITIAL_CAPITAL      = 1_000_000  # USD

# ---------------------------------------------------------------------------
# Parameter Optimisation Grid
# ---------------------------------------------------------------------------
ENTRY_RANGE  = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]   # entry z-score values
EXIT_RANGE   = [0.0, 0.25, 0.5, 0.75]                     # exit  z-score values

# ---------------------------------------------------------------------------
# Output Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR       = "outputs"
PRICE_CACHE_FILE = "outputs/prices.csv"
