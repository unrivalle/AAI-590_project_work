"""
feature_engineering.py
------------------------
Technical indicator engineering + correlation analysis.

The original repo's README claimed "Feature Engineering" and "Correlation
Analysis" as completed features, but that logic only existed (partially)
inside the EDA notebook and wasn't reusable from the app / pipeline. This
module makes it a proper, importable, testable component.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Add common technical-analysis features to an OHLCV DataFrame."""
    if df.empty or price_col not in df.columns:
        return df

    out = df.copy()
    price = out[price_col]

    # Returns & volatility
    out["Daily_Return"] = price.pct_change()
    out["Log_Return"] = np.log(price / price.shift(1))
    out["Volatility_20d"] = out["Daily_Return"].rolling(20).std() * np.sqrt(252)

    # Moving averages
    out["SMA_20"] = price.rolling(20).mean()
    out["SMA_50"] = price.rolling(50).mean()
    out["EMA_12"] = price.ewm(span=12, adjust=False).mean()
    out["EMA_26"] = price.ewm(span=26, adjust=False).mean()

    # MACD
    out["MACD"] = out["EMA_12"] - out["EMA_26"]
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    # RSI (Wilder's smoothing)
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    sma20 = out["SMA_20"]
    std20 = price.rolling(20).std()
    out["BB_Upper"] = sma20 + 2 * std20
    out["BB_Lower"] = sma20 - 2 * std20
    out["BB_Width"] = (out["BB_Upper"] - out["BB_Lower"]) / sma20

    # Momentum
    out["Momentum_10"] = price - price.shift(10)

    return out


def build_correlation_matrix(price_frames: dict[str, pd.DataFrame], value_col: str = "Close", date_col: str = "Date") -> pd.DataFrame:
    """Build a returns-based correlation matrix across multiple tickers."""
    series_dict = {}
    for ticker, df in price_frames.items():
        if df is None or df.empty or value_col not in df.columns:
            continue
        s = df.set_index(date_col)[value_col].pct_change().rename(ticker)
        series_dict[ticker] = s

    if not series_dict:
        return pd.DataFrame()

    merged = pd.concat(series_dict.values(), axis=1, keys=series_dict.keys())
    merged = merged.dropna(how="all")
    return merged.corr()


def merge_with_macro(price_df: pd.DataFrame, macro_df: pd.DataFrame, price_date_col: str = "Date") -> pd.DataFrame:
    """Align a price series with macro indicators on the nearest available date."""
    if price_df.empty or macro_df.empty:
        return pd.DataFrame()

    p = price_df.copy()
    p[price_date_col] = pd.to_datetime(p[price_date_col])
    m = macro_df.copy()
    m["date"] = pd.to_datetime(m["date"])

    p = p.sort_values(price_date_col)
    m = m.sort_values("date")

    merged = pd.merge_asof(
        p, m, left_on=price_date_col, right_on="date", direction="backward"
    )
    return merged
