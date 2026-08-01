import os
import sys
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import MarketAgent, NewsAgent, ReportAgent
from src.data_loader import load_and_preprocess_stock_data

LSTM_INPUT_SIZE = 12
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_OUTPUT_SIZE = 1
LSTM_SEQUENCE_LENGTH = 60
LSTM_MODEL_PATH = "models/lstm_model.pth"

dummy_feature_columns = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return',
    'MA20', 'MA50', 'Volatility', 'RSI', 'MACD', 'Signal_Line',
]

market_agent = MarketAgent(
    model_path=LSTM_MODEL_PATH,
    input_size=LSTM_INPUT_SIZE,
    hidden_size=LSTM_HIDDEN_SIZE,
    num_layers=LSTM_NUM_LAYERS,
    output_size=LSTM_OUTPUT_SIZE,
    sequence_length=LSTM_SEQUENCE_LENGTH,
    scaler_features=None,
    scaler_target=None,
    feature_columns=dummy_feature_columns,
)
news_agent = NewsAgent()
report_agent = ReportAgent()


class AgentState(TypedDict, total=False):
    """Represents the state of our agent's workflow."""

    ticker: str
    company_name: str
    market_data: pd.DataFrame
    market_analysis: str
    news_sentiment: str
    final_report: str


def market_analysis_node(state: AgentState):
    """Node to perform market analysis and price prediction."""
    print("---MARKET ANALYSIS NODE---")
    market_analysis_result = market_agent.analyze(state['market_data'])
    return {"market_analysis": market_analysis_result}


def news_analysis_node(state: AgentState):
    """Node to perform news sentiment analysis."""
    print("---NEWS ANALYSIS NODE---")
    news_analysis_result = news_agent.analyze(state['company_name'])
    return {"news_sentiment": news_analysis_result}


def report_generation_node(state: AgentState):
    """Node to generate the final report."""
    print("---REPORT GENERATION NODE---")
    final_report_content = report_agent.generate(
        state['market_analysis'],
        state['news_sentiment'],
    )
    return {"final_report": final_report_content}


def build_graph():
    """Builds the LangGraph state machine."""
    workflow = StateGraph(AgentState)

    workflow.add_node("market_analysis", market_analysis_node)
    workflow.add_node("news_analysis", news_analysis_node)
    workflow.add_node("report_generation", report_generation_node)

    workflow.set_entry_point("market_analysis")
    workflow.add_edge("market_analysis", "news_analysis")
    workflow.add_edge("news_analysis", "report_generation")
    workflow.add_edge("report_generation", END)

    return workflow.compile()


if __name__ == "__main__":
    TARGET_TICKER = "AAPL"
    COMPANY_FOR_NEWS = "Apple Inc."
    START_DATE = "2023-01-01"
    END_DATE = "2025-01-01"

    full_scaled_df, _, _, _, _, scaler_features_obj, scaler_target_obj, feature_cols = (
        load_and_preprocess_stock_data(
            ticker=TARGET_TICKER,
            start_date=START_DATE,
            end_date=END_DATE,
            sequence_length=LSTM_SEQUENCE_LENGTH,
        )
    )

    market_agent.scaler_features = scaler_features_obj
    market_agent.scaler_target = scaler_target_obj
    market_agent.feature_columns = feature_cols

    latest_market_data_for_agent = full_scaled_df[feature_cols].tail(
        LSTM_SEQUENCE_LENGTH
    )

    app = build_graph()

    initial_state = {
        "ticker": TARGET_TICKER,
        "company_name": COMPANY_FOR_NEWS,
        "market_data": latest_market_data_for_agent,
    }

    print("\n--- Invoking AI Financial Analyst ---")
    result = app.invoke(initial_state)

    print("\n--- AI Financial Analyst Report ---")
    print(result["final_report"])
