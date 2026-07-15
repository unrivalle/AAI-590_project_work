# main.py

from workflow import *
from typing import TypedDict

from langgraph.graph import StateGraph

class AgentState(TypedDict):
    ticker: str
    market: str
    news: list
    report: str


def market_node(state):
    state["market"] = "Market data analyzed"
    return state


def news_node(state):
    state["news"] = ["News 1", "News 2"]
    return state


def report_node(state):
    state["report"] = f"""
Market:
{state['market']}

News:
{state['news']}
"""
    return state


builder = StateGraph(AgentState)

builder.add_node("market", market_node)
builder.add_node("news", news_node)
builder.add_node("report", report_node)

builder.set_entry_point("market")

builder.add_edge("market", "news")
builder.add_edge("news", "report")

graph = builder.compile()

result = graph.invoke({"ticker": "AAPL"})

print(result["report"])
