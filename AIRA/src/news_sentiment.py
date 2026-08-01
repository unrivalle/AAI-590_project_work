"""
news_sentiment.py
------------------
NewsAPI retrieval + lightweight sentiment scoring.

The original repo's retrieval.py had two bugs worth calling out:
    os.getenv("API_KEY")          # return value discarded, never used
    NEWS_API = "API_KEY"          # literal string used as the key -> every
                                   # request would have failed with 401

This version:
  * actually uses the key you pass in (from the sidebar / env var / secrets)
  * adds date sorting + article de-duplication
  * scores each headline with a small finance-oriented lexicon so we don't
    depend on downloading NLTK/VADER corpora at runtime (useful in
    network-restricted environments and keeps the app dependency-light)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NEWS_API_URL = "https://newsapi.org/v2/everything"

# Small finance-flavored sentiment lexicon. Not a substitute for a trained
# model, but transparent, dependency-free, and good enough to bucket
# headlines as clearly positive / negative / neutral for a dashboard.
POSITIVE_WORDS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies",
    "gain", "gains", "growth", "profit", "profits", "record", "upgrade",
    "upgraded", "outperform", "bullish", "strong", "boost", "boosts",
    "expand", "expands", "expansion", "win", "wins", "positive", "rise",
    "rises", "rising", "jump", "jumps", "breakthrough", "success",
    "exceed", "exceeds", "optimistic", "recovery", "top", "tops",
}
NEGATIVE_WORDS = {
    "miss", "misses", "plunge", "plunges", "slump", "slumps", "fall",
    "falls", "falling", "loss", "losses", "downgrade", "downgraded",
    "underperform", "bearish", "weak", "cut", "cuts", "layoff", "layoffs",
    "lawsuit", "investigation", "recall", "decline", "declines", "drop",
    "drops", "crash", "crashes", "warning", "warns", "risk", "risks",
    "concern", "concerns", "delay", "delays", "fraud", "bankruptcy",
    "default", "shortfall", "negative", "sell-off", "selloff",
}


def score_headline(text: str) -> tuple[str, int]:
    """Return (label, score) where score = #positive - #negative words."""
    if not text:
        return "Neutral", 0
    tokens = {t.strip(".,!?:;\"'()").lower() for t in text.split()}
    pos = len(tokens & POSITIVE_WORDS)
    neg = len(tokens & NEGATIVE_WORDS)
    score = pos - neg
    if score > 0:
        label = "Positive"
    elif score < 0:
        label = "Negative"
    else:
        label = "Neutral"
    return label, score


def get_news(company: str, api_key: str, page_size: int = 20, days_back: int = 14) -> dict:
    """Query NewsAPI's `everything` endpoint for a company/keyword."""
    if not api_key:
        return {"status": "error", "message": "No NewsAPI key supplied.", "articles": []}

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params = {
        "q": company,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    try:
        resp = requests.get(NEWS_API_URL, params=params, timeout=15)
        data = resp.json()
        if resp.status_code != 200:
            logger.warning("NewsAPI error: %s", data.get("message"))
        return data
    except Exception as exc:  # noqa: BLE001
        logger.exception("NewsAPI request failed: %s", exc)
        return {"status": "error", "message": str(exc), "articles": []}


def news_to_dataframe(news_json: dict) -> pd.DataFrame:
    """Convert raw NewsAPI JSON into a scored, de-duplicated DataFrame."""
    articles = news_json.get("articles", []) or []
    rows = []
    seen_titles = set()
    for a in articles:
        title = a.get("title") or ""
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        label, score = score_headline(title + " " + (a.get("description") or ""))
        rows.append({
            "PublishedAt": a.get("publishedAt"),
            "Source": (a.get("source") or {}).get("name"),
            "Title": title,
            "Description": a.get("description"),
            "URL": a.get("url"),
            "Sentiment": label,
            "SentimentScore": score,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["PublishedAt"] = pd.to_datetime(df["PublishedAt"], errors="coerce")
    return df.sort_values("PublishedAt", ascending=False).reset_index(drop=True)
