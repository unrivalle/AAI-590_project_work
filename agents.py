# agents.py

class MarketAgent:

    def analyze(self, stock):

        latest = stock["Close"].iloc[-1]

        return f"Latest closing price: {latest:.2f}"


class NewsAgent:

    def analyze(self, news):

        articles = news["articles"][:5]

        summary = []

        for article in articles:

            summary.append(article["title"])

        return summary


class ReportAgent:

    def generate(self, market, news):

        report = f"""

Investment Research Report

-------------------------

Market Analysis

{market}

Recent Headlines

"""

        for headline in news:

            report += f"- {headline}\n"

        return report
