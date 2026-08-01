"""
workflow.py
-----------
The original repo's workflow.py just ran a hardcoded AAPL pipeline as a
top-level script (no functions, nothing importable, no ability to plug
in FRED/EDGAR since those retrieval functions didn't exist yet).

This version wraps the whole pipeline in a single reusable function that
the Streamlit app (or a notebook, or a CLI) can call for any ticker.
"""

from __future__ import annotations

import pandas as pd

from .agents import ResearchOrchestrator
from .data_loader import load_stock_data
from .edgar_fundamentals import get_key_fundamentals
from .feature_engineering import add_technical_indicators
from .macro_fred import get_macro_dashboard
from .news_sentiment import get_news, news_to_dataframe


def run_research_pipeline(
    ticker: str,
    company_name: str,
    period: str = "1y",
    news_api_key: str = "",
    fred_api_key: str = "",
    include_fundamentals: bool = True,
    include_macro: bool = True,
    include_news: bool = True,
) -> dict:
    """Run the full data-collection + multi-agent pipeline for one ticker."""

    stock = load_stock_data(ticker, period=period)
    stock = add_technical_indicators(stock) if not stock.empty else stock

    fundamentals = get_key_fundamentals(ticker) if include_fundamentals else pd.DataFrame()
    macro = get_macro_dashboard(fred_api_key) if (include_macro and fred_api_key) else pd.DataFrame()

    news_df = pd.DataFrame()
    if include_news and news_api_key:
        news_json = get_news(company_name or ticker, news_api_key)
        news_df = news_to_dataframe(news_json)

    orchestrator = ResearchOrchestrator()
    report, findings = orchestrator.run(ticker, stock, fundamentals, macro, news_df)

    return {
        "ticker": ticker,
        "stock": stock,
        "fundamentals": fundamentals,
        "macro": macro,
        "news": news_df,
        "report": report,
        "findings": findings,
    }


if __name__ == "__main__":
    # Simple CLI smoke test (mirrors the original script's behavior, but
    # now uses whichever ticker/company you pass in and reads keys from env).
    import os
    import sys

    tkr = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    name = sys.argv[2] if len(sys.argv) > 2 else "Apple"

    result = run_research_pipeline(
        tkr,
        name,
        news_api_key=os.getenv("NEWS_API_KEY", ""),
        fred_api_key=os.getenv("FRED_API_KEY", ""),
    )
    print(result["report"])
