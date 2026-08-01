"""
agents.py
---------
Multi-agent research pipeline.

The original repo listed "Multi-Agent AI Pipeline" as a Next Phase item
and only had three bare-bones stubs:

    class MarketAgent:
        def analyze(self, stock): ...        # one line, no indicators

    class NewsAgent:
        def analyze(self, news): ...          # just grabbed 5 titles

    class ReportAgent:
        def generate(self, market, news): ... # string concatenation

This module implements that "next phase": each agent takes structured
inputs (price data + indicators, fundamentals, macro data, scored news)
and produces a structured, explainable finding. An Orchestrator chains
them together into one research report.

IMPORTANT: this produces *descriptive, educational research summaries*,
not personalized financial advice or automated trading signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AgentFinding:
    agent: str
    headline: str
    details: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)


class MarketAgent:
    """Analyzes price action + technical indicators."""

    name = "Market Agent"

    def analyze(self, stock: pd.DataFrame) -> AgentFinding:
        if stock is None or stock.empty:
            return AgentFinding(self.name, "No price data available.")

        latest = stock.iloc[-1]
        close = latest.get("Close", float("nan"))
        details = [f"Latest close: {close:.2f}"]

        if "SMA_20" in stock.columns and "SMA_50" in stock.columns:
            sma20, sma50 = latest["SMA_20"], latest["SMA_50"]
            if pd.notna(sma20) and pd.notna(sma50):
                trend = "above" if sma20 > sma50 else "below"
                details.append(f"20-day SMA is {trend} the 50-day SMA ({sma20:.2f} vs {sma50:.2f}) — a golden/death-cross style trend signal.")

        if "RSI_14" in stock.columns and pd.notna(latest.get("RSI_14")):
            rsi = latest["RSI_14"]
            zone = "overbought (>70)" if rsi > 70 else "oversold (<30)" if rsi < 30 else "neutral"
            details.append(f"RSI(14) = {rsi:.1f} — {zone}")

        if "Volatility_20d" in stock.columns and pd.notna(latest.get("Volatility_20d")):
            details.append(f"Annualized 20-day volatility: {latest['Volatility_20d'] * 100:.1f}%")

        if "MACD" in stock.columns and pd.notna(latest.get("MACD")):
            macd_state = "bullish (MACD > signal)" if latest["MACD"] > latest.get("MACD_Signal", 0) else "bearish (MACD < signal)"
            details.append(f"MACD posture: {macd_state}")

        headline = f"Price: {close:.2f}, technical posture summarized from {len(stock)} bars."
        return AgentFinding(self.name, headline, details, {"latest_close": float(close)})


class FundamentalAgent:
    """Analyzes SEC EDGAR fundamentals (revenue, net income, EPS, etc.)."""

    name = "Fundamental Agent"

    def analyze(self, fundamentals: pd.DataFrame) -> AgentFinding:
        if fundamentals is None or fundamentals.empty:
            return AgentFinding(self.name, "No SEC EDGAR fundamentals available.")

        details = []
        for metric, grp in fundamentals.groupby("Metric"):
            grp = grp.sort_values("FiscalYear")
            if len(grp) >= 2:
                first, last = grp.iloc[0], grp.iloc[-1]
                if first["Value"]:
                    pct_change = (last["Value"] - first["Value"]) / abs(first["Value"]) * 100
                    details.append(
                        f"{metric}: {first['Value']:,.0f} (FY{int(first['FiscalYear'])}) -> "
                        f"{last['Value']:,.0f} (FY{int(last['FiscalYear'])}) [{pct_change:+.1f}%]"
                    )
            elif len(grp) == 1:
                row = grp.iloc[0]
                details.append(f"{metric}: {row['Value']:,.0f} (FY{int(row['FiscalYear'])})")

        headline = f"Extracted {fundamentals['Metric'].nunique()} fundamental metrics from SEC EDGAR filings."
        return AgentFinding(self.name, headline, details)


class MacroAgent:
    """Analyzes macro backdrop (FRED indicators) relative to the stock's price."""

    name = "Macro Agent"

    def analyze(self, macro_df: pd.DataFrame) -> AgentFinding:
        if macro_df is None or macro_df.empty:
            return AgentFinding(self.name, "No macroeconomic data available.")

        details = []
        latest_row = macro_df.dropna(how="all", subset=[c for c in macro_df.columns if c != "date"]).iloc[-1]
        for col in macro_df.columns:
            if col == "date":
                continue
            val = latest_row.get(col)
            if pd.notna(val):
                details.append(f"{col}: latest reading {val:.2f}")

        headline = f"Macro snapshot as of {latest_row['date'].date() if 'date' in macro_df.columns else 'latest'}."
        return AgentFinding(self.name, headline, details)


class NewsAgent:
    """Analyzes recent news headlines and aggregate sentiment."""

    name = "News Agent"

    def analyze(self, news_df: pd.DataFrame) -> AgentFinding:
        if news_df is None or news_df.empty:
            return AgentFinding(self.name, "No recent news found.")

        counts = news_df["Sentiment"].value_counts()
        pos = int(counts.get("Positive", 0))
        neg = int(counts.get("Negative", 0))
        neu = int(counts.get("Neutral", 0))

        details = [f"{row.Title} [{row.Sentiment}]" for row in news_df.head(8).itertuples()]

        net = pos - neg
        tone = "net positive" if net > 0 else "net negative" if net < 0 else "balanced/neutral"
        headline = f"{len(news_df)} headlines analyzed — {pos} positive / {neg} negative / {neu} neutral ({tone})."
        return AgentFinding(self.name, headline, details, {"positive": pos, "negative": neg, "neutral": neu})


class ReportAgent:
    """Synthesizes all agent findings into one structured research report."""

    name = "Report Agent"

    def generate(self, ticker: str, findings: list[AgentFinding]) -> str:
        lines = [
            "=" * 60,
            f"AUTONOMOUS INVESTMENT RESEARCH REPORT — {ticker}",
            "=" * 60,
            "",
            "This is an automated, educational research summary generated",
            "from public market, fundamental, macro, and news data. It is",
            "NOT personalized financial advice or a recommendation to buy",
            "or sell any security.",
            "",
        ]
        for f in findings:
            lines.append(f"## {f.agent}")
            lines.append(f"{f.headline}")
            for d in f.details:
                lines.append(f"  - {d}")
            lines.append("")

        return "\n".join(lines)


class ResearchOrchestrator:
    """Coordinates the full multi-agent pipeline end-to-end."""

    def __init__(self) -> None:
        self.market_agent = MarketAgent()
        self.fundamental_agent = FundamentalAgent()
        self.macro_agent = MacroAgent()
        self.news_agent = NewsAgent()
        self.report_agent = ReportAgent()

    def run(
        self,
        ticker: str,
        stock_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        news_df: pd.DataFrame,
    ) -> tuple[str, list[AgentFinding]]:
        findings = [
            self.market_agent.analyze(stock_df),
            self.fundamental_agent.analyze(fundamentals_df),
            self.macro_agent.analyze(macro_df),
            self.news_agent.analyze(news_df),
        ]
        report = self.report_agent.generate(ticker, findings)
        return report, findings
