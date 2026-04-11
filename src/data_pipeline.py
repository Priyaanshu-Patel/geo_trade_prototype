"""
Data pipeline: market prices (yfinance) and geopolitical events (GDELT).
"""
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# Asset universe with sector/region metadata
ASSETS = {
    "AAPL":    {"sector": "tech",     "region": "US",     "name": "Apple"},
    "XOM":     {"sector": "energy",   "region": "US",     "name": "ExxonMobil"},
    "LMT":     {"sector": "defense",  "region": "US",     "name": "Lockheed Martin"},
    "TCS.NS":  {"sector": "tech",     "region": "IN",     "name": "TCS"},
    "ONGC.NS": {"sector": "energy",   "region": "IN",     "name": "ONGC"},
    "HAL.NS":  {"sector": "defense",  "region": "IN",     "name": "HAL"},
    "GLD":     {"sector": "gold",     "region": "global", "name": "Gold ETF"},
    "USO":     {"sector": "oil",      "region": "global", "name": "Oil ETF"},
    "UUP":     {"sector": "currency", "region": "US",     "name": "USD Index ETF"},
}

# Sector groupings for display
SECTOR_COLORS = {
    "tech":     "#4A90E2",
    "energy":   "#F5A623",
    "defense":  "#7B68EE",
    "gold":     "#FFD700",
    "oil":      "#8B4513",
    "currency": "#2ECC71",
}


def fetch_market_data(
    tickers: list,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download adjusted close prices from yfinance.
    Returns DataFrame with tickers as columns, dates as index.
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data[["Close"]]
        closes.columns = tickers

    closes = closes.ffill().bfill()
    closes = closes.dropna(axis=1, how="all")
    return closes


def fetch_gdelt_events(query: str, max_records: int = 10, days_back: int = 30) -> list:
    """
    Fetch geopolitical news from the GDELT Document API (free, no auth).
    Returns list of {title, url, date, source} dicts.
    """
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days_back)

    params = {
        "query": f"{query} sourcelang:eng",
        "mode": "artlist",
        "maxrecords": max_records,
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        "sort": "DateDesc",
        "format": "json",
    }

    try:
        resp = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "date": a.get("seendate", "")[:8],
                "source": a.get("domain", ""),
            }
            for a in articles
            if a.get("title")
        ]
    except Exception as e:
        print(f"[GDELT] fetch failed: {e}")
        return []


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns from price DataFrame."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_correlation_matrix(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling correlation over the last `window` periods."""
    tail = returns.tail(window) if len(returns) >= window else returns
    return tail.corr()
