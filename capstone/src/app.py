import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import build_graph, LSTM_SEQUENCE_LENGTH, market_agent
from src.data_loader import load_and_preprocess_stock_data

# --- Page Configuration ---
st.set_page_config(page_title="AI Financial Analyst", page_icon="📈", layout="wide")

st.title("📈 AI-Powered Financial Research Assistant")
st.markdown("""
This system uses a **Stacked LSTM** to forecast prices and **FinBERT** to analyze real-time news sentiment. 
The agents are orchestrated using **LangGraph**.
""")

# --- Cached Resource Loading ---
@st.cache_resource
def get_compiled_app():
    """Builds and compiles the graph once and caches it."""
    return build_graph()

# --- App Sidebar ---
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock Ticker", value="AAPL").upper()
    company_name = st.text_input("Company Name", value="Apple Inc.")
    
    # Date Range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    
    analyze_btn = st.button("Run Full Analysis", type="primary")

# --- Logic ---
if analyze_btn:
    with st.status("Initializing Agents & Fetching Data...", expanded=True) as status:
        try:
            # 1. Load Data
            st.write("Fetching market data...")
            full_scaled_df, _, _, _, _, scaler_features_obj, scaler_target_obj, feature_cols = (
                load_and_preprocess_stock_data(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    sequence_length=LSTM_SEQUENCE_LENGTH,
                )
            )

            # 2. Update Agent with current scalers
            market_agent.scaler_features = scaler_features_obj
            market_agent.scaler_target = scaler_target_obj
            market_agent.feature_columns = feature_cols

            latest_market_data = full_scaled_df[feature_cols].tail(LSTM_SEQUENCE_LENGTH)

            # 3. Run Graph
            st.write("Running Multi-Agent Workflow...")
            app = get_compiled_app()
            
            initial_state = {
                "ticker": ticker,
                "company_name": company_name,
                "market_data": latest_market_data,
            }
            
            result = app.invoke(initial_state)
            status.update(label="Analysis Complete!", state="complete", expanded=False)

            # --- DISPLAY RESULTS ---
            
            # Layout Columns
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Market Analysis & Forecast")
                st.info(result.get('market_analysis'))
                
                # Plotly Chart for the Report
                st.write("Recent Price Action")
                # We show the last 100 days of the actual data
                raw_data_path = f"data/clean_stock_data_{ticker}.csv"
                if os.path.exists(raw_data_path):
                    df_plot = pd.read_csv(raw_data_path, index_col='Date', parse_dates=True)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_plot.index[-90:], y=df_plot['Close'][-90:], name="Actual Price"))
                    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("News Sentiment")
                st.write(result.get('news_sentiment'))

            st.divider()

            # Final Report Section
            st.subheader("Final Investment Research Report")
            st.markdown(result.get('final_report'))

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

else:
    st.info("Enter a ticker in the sidebar and click 'Run Full Analysis' to begin.")

# --- Footer (Capstone Requirements) ---
st.sidebar.markdown("---")
st.sidebar.caption("AAI-590 Capstone Project")
st.sidebar.caption("Neural Network: Stacked LSTM")
st.sidebar.caption("NLP: FinBERT Transformer")