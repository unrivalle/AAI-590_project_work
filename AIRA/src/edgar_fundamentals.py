"""
edgar_fundamentals.py
----------------------
SEC EDGAR access layer.

This data source was listed in the project README but never actually
implemented in the original repo. SEC EDGAR requires:
  1. A ticker -> CIK (Central Index Key) lookup
  2. A descriptive User-Agent header on every request (SEC will block/
     rate-limit requests without one -- see
     https://www.sec.gov/os/webmaster-faq#developers)
  3. Respecting their public rate limit (<= 10 requests/second)

Endpoints used:
  * https://www.sec.gov/files/company_tickers.json         (ticker -> CIK map)
  * https://data.sec.gov/submissions/CIK##########.json    (filing history)
  * https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json (XBRL facts)
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SEC_USER_AGENT = "AAI-590 Research Agent (educational use; contact: student@example.edu)"
HEADERS = {"User-Agent": SEC_USER_AGENT}

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"

_last_request_ts = 0.0


def _throttle(min_interval: float = 0.11) -> None:
    """Keep us comfortably under SEC's 10 req/sec limit."""
    global _last_request_ts
    now = time.time()
    wait = min_interval - (now - _last_request_ts)
    if wait > 0:
        time.sleep(wait)
    _last_request_ts = time.time()


@lru_cache(maxsize=1)
def _load_ticker_map() -> dict:
    """Download and cache the full SEC ticker -> CIK map (~9k companies)."""
    _throttle()
    resp = requests.get(TICKER_MAP_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    raw = resp.json()
    # raw is like {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in raw.values()}


def ticker_to_cik(ticker: str) -> str | None:
    ticker = ticker.strip().upper()
    try:
        mapping = _load_ticker_map()
        return mapping.get(ticker)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to resolve CIK for %s: %s", ticker, exc)
        return None


def get_recent_filings(ticker: str, limit: int = 10) -> pd.DataFrame:
    """Return the most recent SEC filings (10-K, 10-Q, 8-K, etc.) for a ticker."""
    cik = ticker_to_cik(ticker)
    if not cik:
        return pd.DataFrame()

    try:
        _throttle()
        resp = requests.get(SUBMISSIONS_URL.format(cik10=cik), headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        df = pd.DataFrame(recent)
        if df.empty:
            return df
        df = df[["form", "filingDate", "reportDate", "primaryDocument", "accessionNumber"]]
        df["filing_url"] = df.apply(
            lambda r: (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{r['accessionNumber'].replace('-', '')}/{r['primaryDocument']}"
            ),
            axis=1,
        )
        return df.head(limit)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to fetch filings for %s: %s", ticker, exc)
        return pd.DataFrame()


# A handful of the most commonly-used XBRL fundamental tags.
KEY_CONCEPTS = {
    "Revenues": "Total Revenue",
    "NetIncomeLoss": "Net Income",
    "Assets": "Total Assets",
    "Liabilities": "Total Liabilities",
    "StockholdersEquity": "Shareholders' Equity",
    "EarningsPerShareDiluted": "Diluted EPS",
    "OperatingIncomeLoss": "Operating Income",
    "CashAndCashEquivalentsAtCarryingValue": "Cash & Equivalents",
}


def get_company_facts(ticker: str) -> dict:
    """Pull raw XBRL company facts JSON for a ticker (large payload)."""
    cik = ticker_to_cik(ticker)
    if not cik:
        return {}
    try:
        _throttle()
        resp = requests.get(COMPANY_FACTS_URL.format(cik10=cik), headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to fetch company facts for %s: %s", ticker, exc)
        return {}


def get_key_fundamentals(ticker: str) -> pd.DataFrame:
    """Extract a tidy table of key annual fundamentals (USD) for a ticker."""
    facts = get_company_facts(ticker)
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        return pd.DataFrame()

    rows = []
    for tag, label in KEY_CONCEPTS.items():
        concept = us_gaap.get(tag)
        if not concept:
            continue
        units = concept.get("units", {})
        series = units.get("USD") or units.get("USD/shares") or []
        for point in series:
            # Keep only annual (10-K) figures to avoid duplicated quarterly noise
            if point.get("form") == "10-K" and point.get("fp") == "FY":
                rows.append({
                    "Metric": label,
                    "FiscalYear": point.get("fy"),
                    "Value": point.get("val"),
                    "FiledDate": point.get("filed"),
                    "Form": point.get("form"),
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset=["Metric", "FiscalYear"], keep="last")
    df = df.sort_values(["Metric", "FiscalYear"])
    return df
