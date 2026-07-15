# 

from data_loader import load_stock_data

from retrieval import get_news

from agents import MarketAgent, NewsAgent, ReportAgent


market_agent = MarketAgent()

news_agent = NewsAgent()

report_agent = ReportAgent()


stock = load_stock_data("AAPL")

news = get_news("Apple")


market_result = market_agent.analyze(stock)

news_result = news_agent.analyze(news)


final_report = report_agent.generate(
    market_result,
    news_result
)

print(final_report)
