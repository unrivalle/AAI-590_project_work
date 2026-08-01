"""
macro_fred.py
-------------
FRED (Federal Reserve Economic Data) access layer.

Like SEC EDGAR, FRED was listed as a data source in the README but had
zero implementation in the original repo. Requires a free API key from
https://fred.stlouisfed.org/docs/api/api_key.html
"""

from __future__ import annotations

import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# A curated set of macro series that are broadly relevant to equity research.
DEFAULT_SERIES = {
    "FEDFUNDS": "Effective Federal Funds Rate",
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "UNRATE": "Unemployment Rate",
    "DGS10": "10-Year Treasury Yield",
    "GDP": "Gross Domestic Product",
    "PPIACO": "Producer Price Index",
    "UMCSENT": "Consumer Sentiment (Univ. of Michigan)",
}


def get_fred_series(series_id: str, api_key: str, start: str | None = None) -> pd.DataFrame:
    """Fetch a single FRED series as a tidy DataFrame [date, value]."""
    if not api_key:
        logger.warning("No FRED API key supplied.")
        return pd.DataFrame()

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if start:
        params["observation_start"] = start

    try:
        resp = requests.get(FRED_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        if not obs:
            return pd.DataFrame()
        df = pd.DataFrame(obs)[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).rename(columns={"value": series_id})
        return df
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to fetch FRED series %s: %s", series_id, exc)
        return pd.DataFrame()


def get_macro_dashboard(api_key: str, series_ids: list[str] | None = None, start: str | None = None) -> pd.DataFrame:
    """Fetch several macro series and merge them into one wide DataFrame on date."""
    series_ids = series_ids or list(DEFAULT_SERIES.keys())
    merged: pd.DataFrame | None = None

    for sid in series_ids:
        df = get_fred_series(sid, api_key, start=start)
        if df.empty:
            continue
        merged = df if merged is None else pd.merge(merged, df, on="date", how="outer")

    if merged is None:
        return pd.DataFrame()

    return merged.sort_values("date").reset_index(drop=True)
