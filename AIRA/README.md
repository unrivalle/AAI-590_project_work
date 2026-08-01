# Autonomous Investment Research Agent

Educational capstone project (AAI-590) — a Streamlit research dashboard that
pulls **Yahoo Finance**, **SEC EDGAR**, **FRED**, and **NewsAPI** data and runs
it through a small multi-agent pipeline (Market / Fundamental / Macro / News /
Report agents).

> ⚠️ Educational project. Nothing here is personalized financial advice or an
> automated trading signal.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                       # optional, if you script outside Streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # optional, pre-fill sidebar keys

streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

You can also just paste your **NewsAPI** and **FRED** keys directly into the
sidebar at runtime — nothing is written to disk from there.

- Get a free NewsAPI key: https://newsapi.org/register
- Get a free FRED key: https://fred.stlouisfed.org/docs/api/api_key.html
- LLM Agent mode: pick **one** —
  - **Ollama** (free, runs locally, no key): install from https://ollama.com,
    then `ollama pull llama3.1` and `ollama serve`.
  - **Groq** (free tier, no credit card): https://console.groq.com/keys
  - **Anthropic** (paid, optional): https://console.anthropic.com/settings/keys
- SEC EDGAR needs no key, just a descriptive User-Agent (already set in
  `src/edgar_fundamentals.py`).

## Project layout

```
AIRA/
├── app.py                       # Streamlit frontend (8 tabs, see below)
├── requirements.txt
├── .env.example
├── .streamlit/secrets.toml.example
└── src/
    ├── data_loader.py            # Yahoo Finance (history + near-real-time quotes)
    ├── edgar_fundamentals.py     # SEC EDGAR (CIK lookup, filings, XBRL facts)
    ├── macro_fred.py             # FRED macro series
    ├── news_sentiment.py         # NewsAPI + lightweight sentiment scoring
    ├── feature_engineering.py    # Technical indicators + correlation analysis
    ├── agents.py                 # Rule-based Market/Fundamental/Macro/News/Report agents
    ├── llm_agent.py               # Real agent: Claude tool-calling loop over the same data
    └── workflow.py               # Orchestration entry point
```

## Two "agent" modes, and why there are two

`agents.py` is **not actually agentic** — it's a fixed pipeline of Python
classes with if/else rules ("if RSI > 70, say overbought"). No reasoning, no
autonomy over what to fetch, no ability to change course. It's there because
it's free, deterministic, testable, and needs no LLM at all.

`llm_agent.py` is the real agent. It wraps every function in `data_loader.py`,
`edgar_fundamentals.py`, `macro_fred.py`, `news_sentiment.py`, and
`feature_engineering.py` as a **tool**, and hands control to a model via a
tool-calling loop (`run_agentic_research()` in that file):

1. The model reads the ticker/company and decides which tools it needs —
   it might skip fundamentals for an ETF, or call the macro tool first if it
   suspects a rate-sensitive story, or call the price tool twice with
   different lookback windows.
2. Each tool call and its JSON result gets appended back into the
   conversation so the model can read it and decide what to do next.
3. Once it has enough information, it writes the final report itself —
   real prose, not a filled-in template — including a "Synthesis" section
   that explicitly calls out agreement or tension between signals (e.g.
   strong technicals but negative news sentiment).
4. Every tool call is logged to a transcript so the UI can show *why* the
   agent reached its conclusions (see "Agent's tool-call trace" in the app).

### You do not need to pay for this

Three providers are supported, picked from the sidebar — the tool-calling
loop is identical for all three, only the wire format differs. The default
model on both free options is **`gpt-oss-20b`**, OpenAI's own open-weight
model (Apache 2.0 license) — it was specifically trained for tool-calling
and agentic workflows, so it's a strong free default here.

| Provider | Cost | Default model | Setup |
|---|---|---|---|
| **Ollama** | Free forever, runs on your own machine | `gpt-oss:20b` (~16GB RAM needed) | Install https://ollama.com, run `ollama pull gpt-oss:20b`, then `ollama serve`. No API key. |
| **Groq** | Free tier, no credit card | `openai/gpt-oss-20b` | Sign up at https://console.groq.com/keys. 1,000 req/day free, runs on custom inference hardware (~1,000 tok/sec). |
| **Anthropic** | Paid | `claude-sonnet-4-5` | Only needed if you specifically want Claude and have credits. |

Other models are available from the sidebar dropdown too (Llama 3.3 70B,
Qwen3 32B, `gpt-oss-120b` on Groq, etc.) if you want to compare quality —
smaller/faster models follow tool-calling instructions less reliably than
larger ones, so if the agent seems to skip steps or write thin reports,
try `gpt-oss-120b` or `llama-3.3-70b-versatile` on Groq (still free).

Ollama and Groq both expose an OpenAI-compatible `/v1/chat/completions`
endpoint, so `llm_agent.py` drives both through the same `openai` SDK client
pointed at a different `base_url` — that's why adding a new OpenAI-compatible
provider (Together.ai, Fireworks, LM Studio, etc.) later is a one-line change
to `PROVIDER_DEFAULTS`, not a rewrite.

In the app's **Multi-Agent Report** tab you can toggle between LLM Agent mode
and the free Rule-Based Pipeline side by side — good for a capstone demo,
since you can show the same ticker analyzed both ways.

## App tabs

1. **Real-Time Overview** — polls a live quote + intraday candlestick chart on
   a configurable refresh interval (`streamlit-autorefresh`).
2. **EDA** — price history, volume, summary stats, return distribution,
   missing-value check.
3. **Feature Engineering** — SMA/EMA/RSI/MACD/Bollinger Bands/momentum,
   downloadable as CSV.
4. **Correlation Analysis** — cross-ticker return correlation heatmap, plus
   stock-vs-macro correlation if a FRED key is supplied.
5. **Fundamentals (SEC EDGAR)** — revenue, net income, EPS, assets, etc. from
   XBRL filings, plus a table of recent 10-K/10-Q/8-K filings with links.
6. **Macro (FRED)** — Fed funds rate, CPI, unemployment, 10Y yield, GDP, PPI,
   consumer sentiment.
7. **News & Sentiment** — recent headlines scored with a small finance lexicon
   (positive/negative/neutral), no external model download required.
8. **Multi-Agent Report** — toggle between the real LLM Agent (Claude
   decides which tools to call, writes its own analysis, shows its tool-call
   trace) and the free Rule-Based Pipeline (fixed sequence, templated
   output). Both produce a downloadable report.

## ⚠️ Important limitation: "real-time" data

None of these sources offer a true streaming tick feed on a free tier:

- **Yahoo Finance / yfinance** is an unofficial wrapper around Yahoo's public
  endpoints; quotes are commonly delayed ~15 minutes for many exchanges, and
  intraday history is limited (e.g. 1-minute bars only go back ~7 days).
- **SEC EDGAR** filings are updated as companies file them (not "live" in a
  market-data sense).
- **FRED** series update on the release schedule of the underlying government
  agency (daily at the fastest, monthly/quarterly for most series).
- **NewsAPI**'s free developer tier has a 100-requests/day cap and a ~24h
  publish-time delay on some endpoints.

What the app *does* give you is **near-real-time polling**: it re-queries
these sources on a short timer (default 30s) so the dashboard stays fresh
without you refreshing the page. If you need licensed real-time/streaming
market data, you'd need a paid feed (Polygon.io, IEX Cloud, a broker API,
etc.) — the `data_loader.py` module is structured so swapping in a different
provider only touches one file.

## What was in the original repo vs. what this adds

The GitHub repo (`unrivalle/AAI-590_project_work`) had:

| File | Original content |
|---|---|
| `data_loader.py` | 8 lines: `yf.download()` for a single hardcoded date range |
| `retrieval.py` | NewsAPI call only — **and it had a real bug**: `os.getenv("API_KEY")` was called but its return value was discarded, then `NEWS_API = "API_KEY"` used the *literal string* `"API_KEY"` as the key, so every request would 401 |
| `agents.py` | 3 bare classes (`MarketAgent`, `NewsAgent`, `ReportAgent`) doing one line each — no fundamentals, no macro, no indicators |
| `workflow.py` | A hardcoded top-level script for `"AAPL"` / `"Apple"` — not importable, no CLI args, no error handling |
| `main.py` | Not reviewed in depth, but the tree implies it's similarly minimal |
| — | **No Streamlit app existed at all** |
| — | **SEC EDGAR and FRED were listed in the README as data sources but had zero implementation** |
| — | No `requirements.txt` present in the repo despite being referenced in the README's file tree |
| — | No caching, no rate-limit handling, no `.env`/secrets pattern, no `.gitignore` |

This build fixes all of the above:

- Fixed the NewsAPI key bug (key is now actually passed through and used).
- Implemented SEC EDGAR (ticker→CIK resolution, filings list, XBRL fundamentals)
  with the required User-Agent header and a self-throttling client (SEC's
  public limit is ~10 req/sec).
- Implemented FRED (multi-series dashboard, macro-vs-price merge).
- Expanded the three agent stubs into five components (added `FundamentalAgent`,
  `MacroAgent`, and a `ResearchOrchestrator`) that consume structured data
  instead of one line of string formatting.
- Added a real feature-engineering module (SMA/EMA/RSI/MACD/Bollinger/
  volatility/momentum) and a correlation-analysis module (cross-ticker +
  macro), matching the two checklist items in the README that had no
  standalone, reusable code behind them.
- Added `requirements.txt`, `.env.example`, `.streamlit/secrets.toml.example`,
  `.gitignore`.
- Added caching (`st.cache_data` with sensible TTLs per data source) and
  defensive error handling everywhere a network call happens, so one bad
  ticker/expired key doesn't crash the whole app.
- Built the full Streamlit frontend described in the README as "Next Phase."

## Still missing / good next steps

- **Automated tests** (pytest) — I ran manual + mocked smoke tests while
  building this (see below), but there's no `tests/` suite committed yet.
- **Multi-turn agent memory across runs** — `llm_agent.py` reasons within a
  single research request; it doesn't remember prior sessions or let you
  ask a follow-up question that reuses earlier tool results.
- **Parallel/critique agents** — right now it's one Claude instance calling
  tools sequentially. A framework like LangGraph could run a
  "bull case" and "bear case" agent in parallel and have a third agent
  reconcile them.
- **A real sentiment/NLP model for the free tier** — the rule-based
  pipeline's lexicon scorer is transparent and dependency-free, but a
  FinBERT-style model would be more accurate for nuanced headlines (the
  LLM Agent mode already handles this better since Claude reads full
  headlines directly).
- **Persistent storage** — everything is in-memory/session-cached; a
  database (Postgres/SQLite) would let you track history across runs and
  build a proper backtest.
- **Portfolio-level view** — right now the pipeline is single-ticker
  (plus a compare list for correlation); a true portfolio tracker would
  need position sizing, weights, and aggregated risk metrics.
- **CI** — add a GitHub Actions workflow to run `py_compile`/pytest on push.
- **Licensed real-time data** — swap `data_loader.py` for a paid feed if the
  15-minute Yahoo delay isn't acceptable for your use case.

## How this was validated

Since this environment can't spin up a live Streamlit server against real
API keys, validation was done at the code level:
- `py_compile` + `ast.parse` on every file (catches syntax errors).
- Offline logic tests with synthetic price data through
  `feature_engineering.py` and all five rule-based agents in `agents.py`.
- The `llm_agent.py` tool-calling loop was exercised against a **mocked**
  Anthropic client (fake tool_use → tool_result → final text turns) to
  confirm the loop correctly dispatches to the real data functions, logs a
  transcript, and terminates — without needing a live API key at build time.
You should still do an end-to-end run with your real keys before presenting
this, especially to confirm your NewsAPI/FRED quotas and SEC EDGAR's rate
limit behave as expected under your network.
