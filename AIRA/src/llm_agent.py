"""
llm_agent.py
------------
The real "agent" layer -- Claude/Llama/etc. decides which tools to call
and writes the final analysis itself, instead of a fixed if/else pipeline
(that's what `agents.py` is, and it's kept as a free deterministic
fallback).

PROVIDERS -- pick whichever fits your budget:

  * "ollama"    FREE, runs 100% locally, no API key, no usage limits.
                Install Ollama (https://ollama.com), then:
                    ollama pull gpt-oss:20b
                    ollama serve
                Costs nothing, ever -- it's your own machine's compute.

  * "groq"      FREE tier, cloud-hosted, no credit card required.
                Sign up at https://console.groq.com -> API Keys.
                Very fast (custom inference chips), generous free rate
                limits for personal/eval use.

  * "anthropic" Paid (Claude). Kept as an option if you have credits,
                but not required -- everything below works without it.

Ollama and Groq both expose an OpenAI-compatible `/v1/chat/completions`
endpoint, so both are driven through the same `openai` SDK client pointed
at a different `base_url`. Anthropic uses its own SDK/message format, so
it has a second, parallel loop below. Tool *definitions* and tool
*functions* are shared across all three -- only the wire format differs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from .data_loader import get_realtime_quote, load_stock_data
from .edgar_fundamentals import get_key_fundamentals
from .feature_engineering import add_technical_indicators, build_correlation_matrix
from .macro_fred import DEFAULT_SERIES, get_macro_dashboard
from .news_sentiment import get_news, news_to_dataframe

MAX_TOOL_TURNS = 8

PROVIDER_DEFAULTS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "gpt-oss:20b", "needs_key": False},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-20b", "needs_key": True},
    "anthropic": {"base_url": None, "model": "claude-sonnet-4-5", "needs_key": True},
}

# A few known-good alternatives per provider, shown as a dropdown in the app.
MODEL_CHOICES = {
    "ollama": ["gpt-oss:20b", "llama3.1", "qwen2.5:14b", "mistral-nemo"],
    "groq": ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"],
    "anthropic": ["claude-sonnet-4-5", "claude-opus-4-1", "claude-haiku-4-5"],
}

SYSTEM_PROMPT = """You are an autonomous investment research agent. You have \
tools that pull real data from Yahoo Finance, SEC EDGAR, FRED, and NewsAPI. \
Decide for yourself which tools you need and in what order -- you do not \
have to call all of them, and you can call the same tool again with \
different arguments if the first result wasn't enough (e.g. a longer \
price history, or a different macro series).

When you have enough information, write a structured research report with \
these sections: Market/Technical Picture, Fundamentals, Macro Backdrop, \
News & Sentiment, and a final "Synthesis" section that explicitly notes \
any agreement or tension between the signals (e.g. strong technicals but \
negative news, or healthy fundamentals but a weakening macro backdrop).

Hard rules:
- Cite the actual numbers you pulled from the tools, don't invent figures.
- If a tool returns no data or an error, say so plainly instead of guessing.
- This is educational research, not personalized financial advice. Do not \
issue buy/sell/hold recommendations or price targets. Describe what the \
data shows and let the reader draw conclusions.
- Be concise. Aim for a report a busy analyst could read in two minutes.
"""


@dataclass
class ToolCallLog:
    tool: str
    arguments: dict
    result_preview: str


@dataclass
class AgentRunResult:
    final_report: str
    transcript: list[ToolCallLog] = field(default_factory=list)
    turns_used: int = 0
    error: str | None = None


def _df_to_summary(df: pd.DataFrame, max_rows: int = 12) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.head(max_rows).to_json(orient="records", date_format="iso"))


# ----------------------------------------------------------------------
# Shared tool implementations (provider-agnostic).
# ----------------------------------------------------------------------
def _build_tool_functions(news_api_key: str, fred_api_key: str) -> dict[str, Callable[..., Any]]:
    def get_price_technicals(ticker: str, period: str = "6mo") -> dict:
        df = load_stock_data(ticker, period=period)
        if df.empty:
            return {"error": f"No price data found for {ticker}."}
        feats = add_technical_indicators(df)
        latest = feats.iloc[-1]
        first_close = float(feats["Close"].iloc[0])
        last_close = float(latest["Close"])
        return {
            "ticker": ticker.upper(),
            "period": period,
            "bars": len(feats),
            "last_close": round(last_close, 2),
            "period_change_pct": round((last_close - first_close) / first_close * 100, 2),
            "sma_20": round(float(latest["SMA_20"]), 2) if pd.notna(latest.get("SMA_20")) else None,
            "sma_50": round(float(latest["SMA_50"]), 2) if pd.notna(latest.get("SMA_50")) else None,
            "rsi_14": round(float(latest["RSI_14"]), 1) if pd.notna(latest.get("RSI_14")) else None,
            "macd": round(float(latest["MACD"]), 3) if pd.notna(latest.get("MACD")) else None,
            "macd_signal": round(float(latest["MACD_Signal"]), 3) if pd.notna(latest.get("MACD_Signal")) else None,
            "annualized_volatility_20d_pct": round(float(latest["Volatility_20d"]) * 100, 1) if pd.notna(latest.get("Volatility_20d")) else None,
        }

    def get_realtime_snapshot(ticker: str) -> dict:
        return get_realtime_quote(ticker)

    def get_fundamentals(ticker: str) -> dict:
        df = get_key_fundamentals(ticker)
        if df.empty:
            return {"error": f"No SEC EDGAR XBRL fundamentals found for {ticker} (may not file 10-Ks with the SEC)."}
        latest_per_metric = (
            df.sort_values("FiscalYear").groupby("Metric").tail(1)[["Metric", "FiscalYear", "Value"]]
        )
        return {"ticker": ticker.upper(), "latest_annual_metrics": _df_to_summary(latest_per_metric, max_rows=20)}

    def get_macro_snapshot(series_ids: list[str] | None = None) -> dict:
        if not fred_api_key:
            return {"error": "No FRED API key configured for this session."}
        ids = series_ids or list(DEFAULT_SERIES.keys())
        macro_df = get_macro_dashboard(fred_api_key, ids, start="2018-01-01")
        if macro_df.empty:
            return {"error": "No macro data returned from FRED."}
        latest = macro_df.dropna(how="all", subset=[c for c in macro_df.columns if c != "date"]).iloc[-1]
        return {
            "as_of": str(latest["date"].date()),
            "values": {c: (round(float(latest[c]), 2) if pd.notna(latest[c]) else None) for c in macro_df.columns if c != "date"},
            "series_definitions": DEFAULT_SERIES,
        }

    def get_news_and_sentiment(company: str) -> dict:
        if not news_api_key:
            return {"error": "No NewsAPI key configured for this session."}
        raw = get_news(company, news_api_key)
        ndf = news_to_dataframe(raw)
        if ndf.empty:
            return {"error": f"No recent news found for '{company}'."}
        counts = ndf["Sentiment"].value_counts().to_dict()
        headlines = _df_to_summary(ndf[["Title", "Source", "Sentiment", "PublishedAt"]], max_rows=10)
        return {"sentiment_counts": counts, "recent_headlines": headlines}

    def get_correlation(tickers: list[str], period: str = "6mo") -> dict:
        frames = {t.upper(): load_stock_data(t, period=period) for t in tickers}
        corr = build_correlation_matrix(frames)
        if corr.empty:
            return {"error": "Not enough overlapping data to compute correlations."}
        return {"correlation_matrix": json.loads(corr.round(3).to_json())}

    return {
        "get_price_technicals": get_price_technicals,
        "get_realtime_snapshot": get_realtime_snapshot,
        "get_fundamentals": get_fundamentals,
        "get_macro_snapshot": get_macro_snapshot,
        "get_news_and_sentiment": get_news_and_sentiment,
        "get_correlation": get_correlation,
    }


# Provider-agnostic tool spec: (name, description, JSON-schema parameters).
TOOL_SPECS = [
    {
        "name": "get_price_technicals",
        "description": "Get price history + technical indicators (SMA, RSI, MACD, volatility) for one ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"},
                "period": {"type": "string", "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y"], "description": "Lookback window"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_realtime_snapshot",
        "description": "Get the latest available quote (price, day range, volume) for a ticker. Not a true real-time tick feed; may be delayed.",
        "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
    },
    {
        "name": "get_fundamentals",
        "description": "Get key annual fundamentals (revenue, net income, EPS, assets, liabilities) from SEC EDGAR XBRL filings for a ticker.",
        "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
    },
    {
        "name": "get_macro_snapshot",
        "description": "Get the latest macroeconomic indicators from FRED (Fed funds rate, CPI, unemployment, 10Y yield, GDP, etc).",
        "parameters": {
            "type": "object",
            "properties": {
                "series_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific FRED series IDs. Omit to get the default macro dashboard.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_news_and_sentiment",
        "description": "Get recent news headlines and their sentiment classification for a company.",
        "parameters": {
            "type": "object",
            "properties": {"company": {"type": "string", "description": "Company name to search news for, e.g. Apple"}},
            "required": ["company"],
        },
    },
    {
        "name": "get_correlation",
        "description": "Get a return-correlation matrix across multiple tickers.",
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}, "description": "List of ticker symbols to compare"},
                "period": {"type": "string", "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y"]},
            },
            "required": ["tickers"],
        },
    },
]


def _tools_as_openai_format() -> list[dict]:
    """OpenAI-compatible function-calling shape (used by Ollama + Groq)."""
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in TOOL_SPECS]


def _tools_as_anthropic_format() -> list[dict]:
    return [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in TOOL_SPECS]


def _dispatch_call(dispatch: dict, name: str, args: dict, transcript: list[ToolCallLog]) -> str:
    fn = dispatch.get(name)
    if fn is None:
        result = {"error": f"Unknown tool: {name}"}
    else:
        try:
            result = fn(**args)
        except Exception as exc:  # noqa: BLE001
            result = {"error": str(exc)}
    result_str = json.dumps(result, default=str)
    transcript.append(ToolCallLog(name, args, result_str[:800]))
    return result_str


# ----------------------------------------------------------------------
# OpenAI-compatible loop (Ollama, Groq, or any other compatible endpoint)
# ----------------------------------------------------------------------
def _run_openai_compatible(
    api_key: str, base_url: str, model: str, user_prompt: str,
    dispatch: dict, max_turns: int,
) -> AgentRunResult:
    try:
        from openai import OpenAI
    except ImportError:
        return AgentRunResult("", error="The `openai` package isn't installed. Run: pip install openai")

    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)
    tools = _tools_as_openai_format()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    transcript: list[ToolCallLog] = []

    for turn in range(max_turns):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, tools=tools, tool_choice="auto", max_tokens=2000,
            )
        except Exception as exc:  # noqa: BLE001
            return AgentRunResult("", transcript, turn, error=f"Model call failed: {exc}")

        msg = response.choices[0].message

        if not msg.tool_calls:
            return AgentRunResult(msg.content or "(Agent returned no text.)", transcript, turn + 1)

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result_str = _dispatch_call(dispatch, tc.function.name, args, transcript)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

    return AgentRunResult(
        "Agent hit the max tool-call turn limit before finishing. Try increasing max_turns.",
        transcript, max_turns, error="max_turns_exceeded",
    )


# ----------------------------------------------------------------------
# Anthropic loop (only used if provider="anthropic" and a key is supplied)
# ----------------------------------------------------------------------
def _run_anthropic(
    api_key: str, model: str, user_prompt: str, dispatch: dict, max_turns: int,
) -> AgentRunResult:
    try:
        import anthropic
    except ImportError:
        return AgentRunResult("", error="The `anthropic` package isn't installed. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)
    tools = _tools_as_anthropic_format()
    messages: list[dict] = [{"role": "user", "content": user_prompt}]
    transcript: list[ToolCallLog] = []

    for turn in range(max_turns):
        try:
            response = client.messages.create(
                model=model, max_tokens=2000, system=SYSTEM_PROMPT, tools=tools, messages=messages,
            )
        except Exception as exc:  # noqa: BLE001
            return AgentRunResult("", transcript, turn, error=f"Anthropic API call failed: {exc}")

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
            return AgentRunResult(final_text or "(Agent returned no text.)", transcript, turn + 1)

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            result_str = _dispatch_call(dispatch, block.name, block.input, transcript)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_str})
        messages.append({"role": "user", "content": tool_results})

    return AgentRunResult(
        "Agent hit the max tool-call turn limit before finishing. Try increasing max_turns.",
        transcript, max_turns, error="max_turns_exceeded",
    )


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def run_agentic_research(
    ticker: str,
    company_name: str,
    provider: str = "ollama",
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    news_api_key: str = "",
    fred_api_key: str = "",
    max_turns: int = MAX_TOOL_TURNS,
    extra_instructions: str = "",
) -> AgentRunResult:
    """Run the full agentic loop against a free-or-paid provider.

    provider: "ollama" (free, local), "groq" (free tier, cloud), or
              "anthropic" (paid).
    """
    defaults = PROVIDER_DEFAULTS.get(provider)
    if defaults is None:
        return AgentRunResult("", error=f"Unknown provider '{provider}'. Choose from: {list(PROVIDER_DEFAULTS)}")

    if defaults["needs_key"] and not api_key:
        return AgentRunResult("", error=f"Provider '{provider}' needs an API key.")

    resolved_model = model or defaults["model"]
    dispatch = _build_tool_functions(news_api_key, fred_api_key)
    user_prompt = (
        f"Research {ticker} ({company_name}). Pull whatever combination of price/technical, "
        f"fundamental, macro, and news data you think is useful, then write the report. "
        f"{extra_instructions}"
    ).strip()

    if provider == "anthropic":
        return _run_anthropic(api_key, resolved_model, user_prompt, dispatch, max_turns)

    resolved_base_url = base_url or defaults["base_url"]
    return _run_openai_compatible(api_key, resolved_base_url, resolved_model, user_prompt, dispatch, max_turns)
