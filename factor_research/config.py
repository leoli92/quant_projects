"""
config.py
---------
Central configuration for the Factor Research pipeline.
"""

# ---------------------------------------------------------------------------
# Ticker Universe
# ---------------------------------------------------------------------------
# Static list of ~150 liquid, large-cap US equities spread across sectors.
# NOT survivorship-bias-free: reflects current large-cap constituents applied
# retroactively over 2015-2025 (delisted/failed names from that window are
# absent). See README.md "Known Limitations".
TICKERS = [
    # Technology (19)
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "ACN", "TXN",
    "QCOM", "INTC", "IBM", "AMD", "NOW", "INTU", "AMAT", "MU", "PANW",
    # Communication Services (10)
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "CHTR", "EA",
    # Consumer Discretionary (17)
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "MAR",
    "GM", "F", "ORLY", "YUM", "CMG", "ROST", "HLT",
    # Consumer Staples (12)
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MDLZ", "CL", "MO", "KMB", "GIS", "STZ",
    # Financials (20)
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "AXP", "SPGI",
    "CB", "PNC", "USB", "TFC", "COF", "MET", "AIG", "TRV", "ICE",
    # Health Care (19)
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "MDT", "CVS", "ISRG", "GILD", "CI", "SYK", "BSX", "ZTS",
    # Industrials (17)
    "CAT", "HON", "UNP", "UPS", "BA", "GE", "LMT", "RTX", "DE", "MMM",
    "NOC", "GD", "EMR", "ETN", "CSX", "NSC", "WM",
    # Energy (10)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "WMB",
    # Materials (8)
    "LIN", "APD", "SHW", "FCX", "NEM", "DD", "ECL", "NUE",
    # Utilities (10)
    "NEE", "DUK", "SO", "AEP", "EXC", "SRE", "D", "XEL", "ED", "PEG",
    # Real Estate (8)
    "PLD", "AMT", "EQIX", "PSA", "SPG", "O", "WELL", "DLR",
]

# ---------------------------------------------------------------------------
# Date Ranges
# ---------------------------------------------------------------------------
# In-sample: used for IC/IR estimation and IS backtest metrics
# Out-of-sample: used for forward validation only
IN_SAMPLE_START = "2015-01-01"
IN_SAMPLE_END   = "2022-12-31"
OOS_START       = "2023-01-01"
OOS_END         = "2025-06-30"

# Data fetch starts well before IN_SAMPLE_START so the longest factor lookback
# (momentum: 252 + 21 = 273 trading days) has a full warmup window ahead of the
# first in-sample rebalance date. This warmup window is never traded.
DATA_START = "2013-10-01"

# ---------------------------------------------------------------------------
# Factor Parameters
# ---------------------------------------------------------------------------
MOMENTUM_LOOKBACK_DAYS   = 252   # trailing window (trading days)
MOMENTUM_SKIP_DAYS       = 21    # most-recent month skipped (12-1 momentum)
REVERSAL_LOOKBACK_DAYS   = 21    # 1-month short-term reversal window
VOLATILITY_LOOKBACK_DAYS = 60    # realized volatility window

# ---------------------------------------------------------------------------
# Cross-Sectional Analysis
# ---------------------------------------------------------------------------
N_QUINTILES     = 5
EXECUTION_LAG_DAYS = 1     # signal computed at t, portfolio applied at t+lag
MIN_NAMES_FOR_IC   = 20    # minimum non-NaN cross-sectional names to compute IC

# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
TRANSACTION_COST_BPS = 10          # one-way cost in basis points
INITIAL_CAPITAL      = 1_000_000   # USD

# ---------------------------------------------------------------------------
# IS/OOS Validation
# ---------------------------------------------------------------------------
SHARPE_DECAY_THRESHOLD = 0.3   # IS Sharpe - OOS Sharpe above this is flagged

# ---------------------------------------------------------------------------
# Output Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR       = "outputs"
PRICE_CACHE_FILE = "outputs/prices.csv"
