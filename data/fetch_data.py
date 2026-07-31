"""
data/fetch_data.py
------------------
Download and cache historical adjusted-close prices via yfinance.
Provides log-price and log-return helpers.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Set system proxy before importing yfinance so it can reach Yahoo Finance
# (required when running behind a local proxy / VPN on macOS)
# ---------------------------------------------------------------------------
import subprocess as _sp

def _get_macos_proxy(service: str = "Wi-Fi") -> str | None:
    """Return 'http://host:port' for the macOS HTTPS proxy, or None."""
    try:
        out = _sp.check_output(
            ["networksetup", "-getsecurewebproxy", service],
            stderr=_sp.DEVNULL, text=True,
        )
        enabled = server = port = None
        for line in out.splitlines():
            if line.startswith("Enabled:"):
                enabled = line.split(":", 1)[1].strip()
            elif line.startswith("Server:"):
                server = line.split(":", 1)[1].strip()
            elif line.startswith("Port:"):
                port = line.split(":", 1)[1].strip()
        if enabled == "Yes" and server and port and port != "0":
            return f"http://{server}:{port}"
    except Exception:
        pass
    return None

_proxy = _get_macos_proxy()
if _proxy:
    os.environ.setdefault("http_proxy",  _proxy)
    os.environ.setdefault("https_proxy", _proxy)
    os.environ.setdefault("HTTP_PROXY",  _proxy)
    os.environ.setdefault("HTTPS_PROXY", _proxy)

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def fetch_prices(
    tickers: list[str] = None,
    start: str = None,
    end: str = None,
    cache_file: str = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Download adjusted-close prices for *tickers* between *start* and *end*.

    Parameters
    ----------
    tickers       : list of ticker symbols (defaults to config.TICKERS)
    start / end   : date strings 'YYYY-MM-DD' (defaults to full IS+OOS range)
    cache_file    : path to CSV cache (defaults to config.PRICE_CACHE_FILE)
    force_download: if True, ignore cache and re-download

    Returns
    -------
    pd.DataFrame  : columns = tickers, index = DatetimeIndex (trading days)
    """
    tickers    = tickers    or config.TICKERS
    start      = start      or config.IN_SAMPLE_START
    end        = end        or config.OOS_END
    cache_file = cache_file or config.PRICE_CACHE_FILE

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if not force_download and os.path.exists(cache_file):
        print(f"[fetch_data] Loading prices from cache: {cache_file}")
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Reject empty cache (e.g. a header-only file from a failed download)
        if prices.empty:
            print(f"[fetch_data] Cache is empty; re-downloading.")
        else:
            # Verify all requested tickers are present
            missing = [t for t in tickers if t not in prices.columns]
            if not missing:
                return prices[tickers]
            print(f"[fetch_data] Cache missing tickers {missing}; re-downloading.")

    print(f"[fetch_data] Downloading {len(tickers)} tickers from {start} to {end} …")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns MultiIndex columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw_close = raw["Close"]  # type: ignore[index]
        prices: pd.DataFrame = (
            raw_close.copy() if isinstance(raw_close, pd.DataFrame) else raw_close.to_frame()
        )
    else:
        prices = raw[["Close"]].copy()  # type: ignore[index]
        prices.columns = tickers

    # Drop rows where ALL tickers are NaN, then forward-fill remaining gaps
    prices = prices.dropna(how="all").ffill()

    # Keep only requested tickers that actually downloaded
    available = [t for t in tickers if t in prices.columns]
    prices = prices[available]

    prices.to_csv(cache_file)
    print(f"[fetch_data] Saved {prices.shape} price matrix to {cache_file}")
    return prices


def log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Return natural-log of price DataFrame."""
    return pd.DataFrame(np.log(prices), index=prices.index, columns=prices.columns)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Return log-return DataFrame (first row is NaN)."""
    return pd.DataFrame(np.log(prices), index=prices.index, columns=prices.columns).diff()


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    px = fetch_prices()
    print(px.tail())
    print("\nLog prices (tail):")
    print(log_prices(px).tail())
    print("\nLog returns (tail):")
    print(log_returns(px).tail())
