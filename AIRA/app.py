"""
app.py
------
Streamlit frontend for the Autonomous Investment Research Agent.

Run with:
    streamlit run app.py

Tabs:
  1. Real-Time Overview   - live-polled quote + intraday chart
  2. EDA                  - price history, distributions, summary stats
  3. Feature Engineering  - technical indicators (SMA/EMA/RSI/MACD/Bollinger)
  4. Correlation Analysis - cross-ticker & macro correlation heatmaps
  5. Fundamentals         - SEC EDGAR company facts
  6. Macro (FRED)         - macroeconomic dashboard
  7. News & Sentiment     - NewsAPI headlines with lexicon sentiment
  8. Multi-Agent Report   - full pipeline output (Market/Fundamental/Macro/News/Report agents)
"""

from __future__ import annotations

import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import get_company_profile, get_realtime_quote, load_stock_data
from src.edgar_fundamentals import get_key_fundamentals, get_recent_filings
from src.feature_engineering import add_technical_indicators, build_correlation_matrix, merge_with_macro
from src.macro_fred import DEFAULT_SERIES, get_macro_dashboard
from src.news_sentiment import get_news, news_to_dataframe
from src.agents import ResearchOrchestrator
from src.llm_agent import run_agentic_research, MODEL_CHOICES

st.set_page_config(page_title="Autonomous Investment Research Agent", layout="wide", page_icon="\U0001F4C8")

# ----------------------------------------------------------------------
# Auto-refresh (near real-time polling). streamlit-autorefresh is an
# optional dependency -- degrade gracefully if it isn't installed.
# ----------------------------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False


# ----------------------------------------------------------------------
# Cached data-access wrappers. TTL keeps calls "near real-time" while
# respecting each API's rate limits.
# ----------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def cached_quote(ticker: str):
    return get_realtime_quote(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_history(ticker: str, period: str, interval: str):
    return load_stock_data(ticker, period=period, interval=interval)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_profile(ticker: str):
    return get_company_profile(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fundamentals(ticker: str):
    return get_key_fundamentals(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_filings(ticker: str):
    return get_recent_filings(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_macro(api_key: str, series_ids: tuple, start: str):
    return get_macro_dashboard(api_key, list(series_ids), start=start)


@st.cache_data(ttl=600, show_spinner=False)
def cached_news(company: str, api_key: str):
    raw = get_news(company, api_key)
    return news_to_dataframe(raw)


# ----------------------------------------------------------------------
# Sidebar - configuration
# ----------------------------------------------------------------------
st.sidebar.title("\U0001F4C8 Research Configuration")

ticker = st.sidebar.text_input("Primary ticker", value="AAPL").strip().upper()
company_name = st.sidebar.text_input("Company name (for news search)", value="Apple")
compare_tickers = st.sidebar.text_input(
    "Compare with (comma-separated, optional)", value="MSFT, GOOGL, AMZN"
)
period = st.sidebar.selectbox("History range", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
interval = st.sidebar.selectbox("Bar interval", ["1d", "1wk", "1mo"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("API Keys")
st.sidebar.caption("Keys are only kept in this session's memory, never written to disk.")
news_api_key = st.sidebar.text_input("NewsAPI key", type="password", value=st.secrets.get("NEWS_API_KEY", "") if hasattr(st, "secrets") else "")
fred_api_key = st.sidebar.text_input("FRED API key", type="password", value=st.secrets.get("FRED_API_KEY", "") if hasattr(st, "secrets") else "")

st.sidebar.markdown("---")
st.sidebar.subheader("LLM Agent Provider")
st.sidebar.caption("Powers the Agent mode in the Multi-Agent Report tab. Ollama is 100% free (runs on your machine); Groq has a free tier; Anthropic is paid.")
llm_provider = st.sidebar.selectbox(
    "Provider", ["ollama", "groq", "anthropic"],
    format_func=lambda p: {"ollama": "Ollama — free, local", "groq": "Groq — free tier, cloud", "anthropic": "Anthropic — paid"}[p],
)
llm_model = st.sidebar.selectbox("Model", MODEL_CHOICES[llm_provider])
st.sidebar.caption(
    "`gpt-oss-20b` (OpenAI's open-weight model) is free on Groq and runs locally via Ollama — "
    "it's trained specifically for tool-calling/agentic use, so it's a good default here."
)
llm_api_key = ""
llm_base_url = ""
if llm_provider == "ollama":
    llm_base_url = st.sidebar.text_input("Ollama base URL", value="http://localhost:11434/v1")
    st.sidebar.caption(f"Requires Ollama running locally: `ollama pull {llm_model}` then `ollama serve`.")
elif llm_provider == "groq":
    llm_api_key = st.sidebar.text_input(
        "Groq API key", type="password",
        value=st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else "",
        help="Free, no credit card: https://console.groq.com/keys",
    )
else:
    llm_api_key = st.sidebar.text_input(
        "Anthropic API key", type="password",
        value=st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else "",
        help="https://console.anthropic.com/settings/keys",
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Real-Time Settings")
auto_refresh = st.sidebar.toggle("Enable auto-refresh", value=False)
refresh_secs = st.sidebar.slider("Refresh interval (seconds)", 10, 120, 30, step=5)

if auto_refresh:
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=refresh_secs * 1000, key="realtime_refresh")
    else:
        st.sidebar.warning(
            "Install `streamlit-autorefresh` for automatic polling "
            "(`pip install streamlit-autorefresh`). Falling back to a "
            "manual refresh button below for now."
        )
        if st.sidebar.button("\U0001F504 Refresh now"):
            st.cache_data.clear()
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "\u26A0\uFE0F Educational project. Not financial advice. Yahoo Finance quotes "
    "via yfinance are typically delayed and are not a licensed real-time feed."
)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("Autonomous Investment Research Agent")
st.caption(
    "Yahoo Finance \u00b7 SEC EDGAR \u00b7 FRED \u00b7 NewsAPI \u2014 Data Collection, EDA, "
    "Feature Engineering, Correlation Analysis, and a Multi-Agent Research Pipeline."
)

tabs = st.tabs([
    "\U0001F4E1 Real-Time Overview",
    "\U0001F4CA EDA",
    "\U0001F6E0\uFE0F Feature Engineering",
    "\U0001F517 Correlation Analysis",
    "\U0001F3E6 Fundamentals (SEC EDGAR)",
    "\U0001F3DB\uFE0F Macro (FRED)",
    "\U0001F4F0 News & Sentiment",
    "\U0001F916 Multi-Agent Report",
])

# ========================================================================
# TAB 1 - REAL-TIME OVERVIEW
# ========================================================================
with tabs[0]:
    st.subheader(f"Live Snapshot — {ticker}")
    quote = cached_quote(ticker)

    if "error" in quote:
        st.error(f"Could not fetch a live quote for {ticker}: {quote['error']}")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Last Price", f"${quote['last_price']:.2f}", f"{quote['change']:+.2f} ({quote['change_pct']:+.2f}%)")
        c2.metric("Open", f"${quote['open']:.2f}")
        c3.metric("Day High", f"${quote['day_high']:.2f}")
        c4.metric("Day Low", f"${quote['day_low']:.2f}")
        c5.metric("Volume", f"{quote['volume']:,}")
        st.caption(f"Snapshot polled at {quote['timestamp']} UTC \u00b7 exchange: {quote.get('exchange')} \u00b7 currency: {quote.get('currency')}")

    intraday = cached_history(ticker, "5d", "15m")
    if not intraday.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=intraday["Datetime"] if "Datetime" in intraday.columns else intraday.get("Date"),
            open=intraday["Open"], high=intraday["High"], low=intraday["Low"], close=intraday["Close"],
        )])
        fig.update_layout(title=f"{ticker} — Recent Intraday Activity (15m bars, last 5 days)", xaxis_rangeslider_visible=False, height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Intraday data unavailable right now (market may be closed, or ticker has limited intraday history).")

    profile = cached_profile(ticker)
    if profile:
        with st.expander("Company Profile"):
            st.write(f"**{profile.get('longName', ticker)}** \u2014 {profile.get('sector', 'N/A')} / {profile.get('industry', 'N/A')}")
            st.write(profile.get("longBusinessSummary", "No description available."))
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Market Cap", f"${(profile.get('marketCap') or 0):,}")
            pc2.metric("Trailing P/E", f"{profile.get('trailingPE') or 'N/A'}")
            pc3.metric("Beta", f"{profile.get('beta') or 'N/A'}")
            pc4.metric("Avg Volume", f"{(profile.get('averageVolume') or 0):,}")

# ========================================================================
# TAB 2 - EDA
# ========================================================================
with tabs[1]:
    st.subheader(f"Exploratory Data Analysis — {ticker}")
    hist = cached_history(ticker, period, interval)

    if hist.empty:
        st.warning("No historical data returned for this ticker/range.")
    else:
        date_col = "Date" if "Date" in hist.columns else hist.columns[0]

        fig = px.line(hist, x=date_col, y="Close", title=f"{ticker} Closing Price ({period})")
        st.plotly_chart(fig, use_container_width=True)

        vol_fig = px.bar(hist, x=date_col, y="Volume", title="Trading Volume")
        st.plotly_chart(vol_fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Summary Statistics**")
            st.dataframe(hist[["Open", "High", "Low", "Close", "Volume"]].describe(), use_container_width=True)
        with c2:
            st.markdown("**Daily Return Distribution**")
            returns = hist["Close"].pct_change().dropna()
            dist_fig = px.histogram(returns, nbins=50, title="Daily Return Distribution")
            st.plotly_chart(dist_fig, use_container_width=True)

        with st.expander("Missing-value check (Data Cleaning)"):
            missing = hist.isna().sum()
            st.dataframe(missing[missing > 0].rename("Missing Count") if missing.any() else pd.DataFrame({"Missing Count": ["None \u2014 data is clean"]}))

        with st.expander("Raw data"):
            st.dataframe(hist, use_container_width=True)
            st.download_button("Download as CSV", hist.to_csv(index=False), file_name=f"{ticker}_history.csv")

# ========================================================================
# TAB 3 - FEATURE ENGINEERING
# ========================================================================
with tabs[2]:
    st.subheader(f"Technical Feature Engineering — {ticker}")
    hist = cached_history(ticker, period, interval)

    if hist.empty:
        st.warning("No data to engineer features from.")
    else:
        feats = add_technical_indicators(hist)
        date_col = "Date" if "Date" in feats.columns else feats.columns[0]

        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=feats[date_col], y=feats["Close"], name="Close"))
        price_fig.add_trace(go.Scatter(x=feats[date_col], y=feats["SMA_20"], name="SMA 20"))
        price_fig.add_trace(go.Scatter(x=feats[date_col], y=feats["SMA_50"], name="SMA 50"))
        price_fig.add_trace(go.Scatter(x=feats[date_col], y=feats["BB_Upper"], name="Bollinger Upper", line=dict(dash="dot")))
        price_fig.add_trace(go.Scatter(x=feats[date_col], y=feats["BB_Lower"], name="Bollinger Lower", line=dict(dash="dot")))
        price_fig.update_layout(title="Price with Moving Averages & Bollinger Bands", height=450)
        st.plotly_chart(price_fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            rsi_fig = px.line(feats, x=date_col, y="RSI_14", title="RSI (14)")
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(rsi_fig, use_container_width=True)
        with c2:
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=feats[date_col], y=feats["MACD"], name="MACD"))
            macd_fig.add_trace(go.Scatter(x=feats[date_col], y=feats["MACD_Signal"], name="Signal"))
            macd_fig.add_trace(go.Bar(x=feats[date_col], y=feats["MACD_Hist"], name="Histogram"))
            macd_fig.update_layout(title="MACD")
            st.plotly_chart(macd_fig, use_container_width=True)

        st.markdown("**Engineered Feature Table**")
        feature_cols = [date_col, "Close", "Daily_Return", "Volatility_20d", "SMA_20", "SMA_50",
                        "RSI_14", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower", "Momentum_10"]
        st.dataframe(feats[[c for c in feature_cols if c in feats.columns]].tail(100), use_container_width=True)
        st.download_button("Download engineered features (CSV)", feats.to_csv(index=False), file_name=f"{ticker}_features.csv")

# ========================================================================
# TAB 4 - CORRELATION ANALYSIS
# ========================================================================
with tabs[3]:
    st.subheader("Cross-Asset Correlation Analysis")
    tickers_list = [ticker] + [t.strip().upper() for t in compare_tickers.split(",") if t.strip()]
    tickers_list = list(dict.fromkeys(tickers_list))  # de-dupe, preserve order

    price_frames = {t: cached_history(t, period, interval) for t in tickers_list}
    corr = build_correlation_matrix(price_frames)

    if corr.empty:
        st.warning("Not enough overlapping data to compute correlations. Try a longer history range.")
    else:
        heat_fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1,
            title=f"Return Correlation Matrix ({', '.join(tickers_list)})",
        )
        st.plotly_chart(heat_fig, use_container_width=True)
        st.dataframe(corr, use_container_width=True)

    if fred_api_key:
        st.markdown("---")
        st.markdown("**Stock vs. Macro Indicator Correlation**")
        macro_df = cached_macro(fred_api_key, tuple(DEFAULT_SERIES.keys()), "2015-01-01")
        primary_hist = price_frames.get(ticker, pd.DataFrame())
        if not macro_df.empty and not primary_hist.empty:
            merged = merge_with_macro(primary_hist, macro_df)
            macro_cols = [c for c in DEFAULT_SERIES if c in merged.columns]
            if macro_cols:
                macro_corr = merged[["Close"] + macro_cols].pct_change().corr()
                st.plotly_chart(
                    px.imshow(macro_corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1,
                              title=f"{ticker} vs. Macro Indicators"),
                    use_container_width=True,
                )
    else:
        st.info("Add a FRED API key in the sidebar to correlate this stock against macro indicators.")

# ========================================================================
# TAB 5 - FUNDAMENTALS (SEC EDGAR)
# ========================================================================
with tabs[4]:
    st.subheader(f"SEC EDGAR Fundamentals — {ticker}")
    with st.spinner("Querying SEC EDGAR..."):
        fundamentals = cached_fundamentals(ticker)
        filings = cached_filings(ticker)

    if fundamentals.empty:
        st.warning("No XBRL fundamentals found (ticker may not file 10-Ks with the SEC, e.g. foreign private issuers, ETFs, or crypto).")
    else:
        metric_choice = st.selectbox("Metric", sorted(fundamentals["Metric"].unique()))
        sub = fundamentals[fundamentals["Metric"] == metric_choice].sort_values("FiscalYear")
        fig = px.bar(sub, x="FiscalYear", y="Value", title=f"{metric_choice} by Fiscal Year")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fundamentals, use_container_width=True)

    st.markdown("**Recent Filings**")
    if filings.empty:
        st.info("No recent filings found.")
    else:
        st.dataframe(filings, use_container_width=True, column_config={"filing_url": st.column_config.LinkColumn("Filing Link")})

# ========================================================================
# TAB 6 - MACRO (FRED)
# ========================================================================
with tabs[5]:
    st.subheader("Macroeconomic Dashboard (FRED)")
    if not fred_api_key:
        st.info("Add a FRED API key in the sidebar to load this tab. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
    else:
        macro_df = cached_macro(fred_api_key, tuple(DEFAULT_SERIES.keys()), "2015-01-01")
        if macro_df.empty:
            st.warning("No macro data returned. Double check your FRED API key.")
        else:
            selected = st.multiselect("Series to plot", list(DEFAULT_SERIES.keys()),
                                       default=["FEDFUNDS", "CPIAUCSL", "UNRATE"],
                                       format_func=lambda s: f"{s} — {DEFAULT_SERIES[s]}")
            if selected:
                fig = go.Figure()
                for s in selected:
                    if s in macro_df.columns:
                        fig.add_trace(go.Scatter(x=macro_df["date"], y=macro_df[s], name=DEFAULT_SERIES[s]))
                fig.update_layout(title="Macro Indicators Over Time", height=450)
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(macro_df, use_container_width=True)

# ========================================================================
# TAB 7 - NEWS & SENTIMENT
# ========================================================================
with tabs[6]:
    st.subheader(f"News & Sentiment — {company_name or ticker}")
    if not news_api_key:
        st.info("Add a NewsAPI key in the sidebar to load this tab. Get a free key at https://newsapi.org/register")
    else:
        news_df = cached_news(company_name or ticker, news_api_key)
        if news_df.empty:
            st.warning("No recent articles found (or the NewsAPI key/quota is invalid).")
        else:
            counts = news_df["Sentiment"].value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("Positive", int(counts.get("Positive", 0)))
            c2.metric("Neutral", int(counts.get("Neutral", 0)))
            c3.metric("Negative", int(counts.get("Negative", 0)))

            pie_fig = px.pie(names=counts.index, values=counts.values, title="Headline Sentiment Split",
                              color=counts.index,
                              color_discrete_map={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"})
            st.plotly_chart(pie_fig, use_container_width=True)

            st.dataframe(
                news_df[["PublishedAt", "Source", "Title", "Sentiment", "SentimentScore", "URL"]],
                use_container_width=True,
                column_config={"URL": st.column_config.LinkColumn("Article")},
            )

# ========================================================================
# TAB 8 - MULTI-AGENT REPORT
# ========================================================================
with tabs[7]:
    st.subheader(f"Multi-Agent Research Report — {ticker}")

    mode = st.radio(
        "Mode",
        ["\U0001F9E0 LLM Agent (Claude decides which tools to call)", "\u2699\uFE0F Rule-Based Pipeline (fixed sequence, no LLM)"],
        help=(
            "LLM Agent: Claude reasons about the ticker, autonomously chooses which of your "
            "data tools to call and in what order, and writes the analysis in its own words. "
            "Rule-Based: always runs all four agents in a fixed order and fills a template — "
            "deterministic and free, but not actually 'agentic'."
        ),
    )

    if mode.startswith("\U0001F9E0"):
        st.caption(
            "The agent sees 6 tools (price/technicals, real-time quote, SEC fundamentals, FRED macro, "
            "news sentiment, correlation) and decides for itself what to pull before writing the report."
        )
        max_turns = st.slider("Max tool-call turns", 2, 12, 8)

        provider_needs_key = llm_provider != "ollama"
        if provider_needs_key and not llm_api_key:
            st.info(f"Add a {llm_provider.capitalize()} API key in the sidebar to use Agent mode.")
        elif llm_provider == "ollama":
            st.caption(f"Using local Ollama at `{llm_base_url}` — make sure `ollama serve` is running and you've pulled `{llm_model}`.")

        if st.button("\u25B6\uFE0F Run LLM Agent", type="primary"):
            with st.spinner(f"Agent ({llm_provider}/{llm_model}) is reasoning and calling tools..."):
                result = run_agentic_research(
                    ticker=ticker,
                    company_name=company_name or ticker,
                    provider=llm_provider,
                    api_key=llm_api_key,
                    base_url=llm_base_url,
                    model=llm_model,
                    news_api_key=news_api_key,
                    fred_api_key=fred_api_key,
                    max_turns=max_turns,
                )

            if result.error:
                st.error(f"Agent run failed: {result.error}")
                if llm_provider == "ollama":
                    st.caption(f"Common fix: make sure `ollama serve` is running and the model is pulled (`ollama pull {llm_model}`).")
            else:
                st.success(f"Agent finished in {result.turns_used} turn(s), calling {len(result.transcript)} tool(s).")

                with st.expander(f"\U0001F50D Agent's tool-call trace ({len(result.transcript)} calls)", expanded=False):
                    for i, call in enumerate(result.transcript, 1):
                        st.markdown(f"**{i}. `{call.tool}`**  \nArguments: `{call.arguments}`")
                        st.code(call.result_preview, language="json")

                st.markdown("---")
                st.markdown("### \U0001F4C4 Agent Report")
                st.markdown(result.final_report)
                st.download_button("Download report (.txt)", result.final_report, file_name=f"{ticker}_agent_report.txt")

    else:
        st.caption("Runs the Market, Fundamental, Macro, and News agents in a fixed order, then fills a report template.")
        if st.button("\u25B6\uFE0F Run Rule-Based Pipeline", type="primary"):
            with st.spinner("Coordinating agents..."):
                stock = cached_history(ticker, period, interval)
                stock_feat = add_technical_indicators(stock) if not stock.empty else stock
                fundamentals = cached_fundamentals(ticker)
                macro_df = cached_macro(fred_api_key, tuple(DEFAULT_SERIES.keys()), "2015-01-01") if fred_api_key else pd.DataFrame()
                news_df = cached_news(company_name or ticker, news_api_key) if news_api_key else pd.DataFrame()

                orchestrator = ResearchOrchestrator()
                report, findings = orchestrator.run(ticker, stock_feat, fundamentals, macro_df, news_df)

            for f in findings:
                with st.expander(f"{f.agent} — {f.headline}", expanded=True):
                    for d in f.details:
                        st.write(f"- {d}")

            st.markdown("---")
            st.markdown("### \U0001F4C4 Final Report")
            st.code(report, language="text")
            st.download_button("Download report (.txt)", report, file_name=f"{ticker}_research_report.txt")
        else:
            st.info("Click the button above to run the fixed pipeline for the current ticker and settings.")

st.markdown("---")
st.caption(
    "Educational capstone project (AAI-590). Data sourced from Yahoo Finance, SEC EDGAR, FRED, and NewsAPI. "
    "Quotes may be delayed. Not investment advice."
)
