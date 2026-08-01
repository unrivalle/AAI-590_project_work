"""
data_loader.py
--------------
Yahoo Finance data access layer.

Original repo only had:

    def load_stock_data(ticker="AAPL"):
        df = yf.download(ticker, start="2023-01-01", end="2025-01-01")
        return df

This version adds:
  * configurable date range / interval
  * multi-ticker batch loading
  * a "near real-time" quote function (yfinance has no true tick-by-tick
    real-time feed on the free tier -- what we *can* do is pull the most
    recent 1-minute bar and the live bid/ask snapshot, then poll it on a
    short interval from Streamlit)
  * defensive error handling so one bad ticker doesn't crash the app
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def load_stock_data(
    ticker: str = "AAPL",
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download historical OHLCV data for a single ticker.

    Either supply `start`/`end` OR `period` (e.g. "6mo", "1y", "5y", "max").
    `period` takes precedence if both are given.
    """
    ticker = ticker.strip().upper()
    try:
        if period:
            df = yf.download(
                ticker, period=period, interval=interval,
                auto_adjust=auto_adjust, progress=False,
            )
        else:
            start = start or (datetime.today() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
            end = end or datetime.today().strftime("%Y-%m-%d")
            df = yf.download(
                ticker, start=start, end=end, interval=interval,
                auto_adjust=auto_adjust, progress=False,
            )

        if df is None or df.empty:
            logger.warning("No data returned for %s", ticker)
            return pd.DataFrame()

        # yfinance sometimes returns a MultiIndex column frame even for a
        # single ticker -- flatten it so downstream code can rely on
        # simple column names like 'Close', 'Volume', etc.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df["Ticker"] = ticker
        return df

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load data for %s: %s", ticker, exc)
        return pd.DataFrame()


def load_multiple_tickers(
    tickers: Iterable[str],
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Load several tickers at once, returned as {ticker: DataFrame}."""
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        out[t.strip().upper()] = load_stock_data(
            t, start=start, end=end, period=period, interval=interval
        )
    return out


def get_realtime_quote(ticker: str) -> dict:
    """Best-effort 'real-time' snapshot using yfinance.

    NOTE: yfinance is backed by Yahoo's public/unofficial endpoints. Prices
    are typically delayed ~15 minutes for many exchanges and there is no
    true streaming tick feed. This function pulls the freshest data
    yfinance can give us (fast_info + latest 1-minute bar) so the app can
    poll it every N seconds to *feel* real-time.
    """
    ticker = ticker.strip().upper()
    try:
        tk = yf.Ticker(ticker)
        fast = tk.fast_info

        intraday = tk.history(period="1d", interval="1m")
        last_bar = intraday.tail(1)

        quote = {
            "ticker": ticker,
            "last_price": float(fast.get("last_price") or (last_bar["Close"].iloc[-1] if not last_bar.empty else float("nan"))),
            "previous_close": float(fast.get("previous_close", float("nan"))),
            "day_high": float(fast.get("day_high", float("nan"))),
            "day_low": float(fast.get("day_low", float("nan"))),
            "open": float(fast.get("open", float("nan"))),
            "volume": int(fast.get("last_volume", 0) or 0),
            "market_cap": fast.get("market_cap"),
            "currency": fast.get("currency"),
            "exchange": fast.get("exchange"),
            "timestamp": datetime.utcnow().isoformat(),
        }
        if quote["previous_close"] and quote["previous_close"] == quote["previous_close"]:
            quote["change"] = quote["last_price"] - quote["previous_close"]
            quote["change_pct"] = (quote["change"] / quote["previous_close"]) * 100
        else:
            quote["change"] = float("nan")
            quote["change_pct"] = float("nan")
        return quote

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to fetch realtime quote for %s: %s", ticker, exc)
        return {"ticker": ticker, "error": str(exc)}


def get_company_profile(ticker: str) -> dict:
    """Basic company info (sector, industry, description, key stats)."""
    ticker = ticker.strip().upper()
    try:
        info = yf.Ticker(ticker).info
        keep = [
            "longName", "sector", "industry", "longBusinessSummary",
            "fullTimeEmployees", "website", "trailingPE", "forwardPE",
            "dividendYield", "beta", "marketCap", "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow", "averageVolume",
        ]
        return {k: info.get(k) for k in keep}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load profile for %s: %s", ticker, exc)
        return {}
