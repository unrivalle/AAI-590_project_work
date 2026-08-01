import os

import numpy as np
import torch

from src.models import LSTMModel
from src.retrieval import analyze_sentiment, get_news


class MarketAgent:
    """
    Agent responsible for analyzing market data and making price predictions.
    Requires a pre-trained LSTM model and scaler.
    """

    def __init__(
        self,
        model_path,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        sequence_length,
        scaler_features,
        scaler_target,
        feature_columns,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel(
            input_size, hidden_size, num_layers, output_size
        ).to(self.device)
        self.sequence_length = sequence_length
        self.scaler_features = scaler_features
        self.scaler_target = scaler_target
        self.feature_columns = feature_columns

        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            print(f"LSTM model loaded from {model_path}")
        else:
            print(
                f"Warning: LSTM model not found at {model_path}. "
                "Please train the model first."
            )
            self.model = None

    def analyze(self, latest_stock_data_df):
        """
        Analyzes the latest stock data to predict the next 'Close' price.
        latest_stock_data_df: DataFrame containing the last `sequence_length` rows of scaled data.
        """
        if self.model is None:
            return "Market prediction not available (model not loaded or trained)."

        if self.scaler_target is None:
            return "Market prediction not available (scaler not initialized)."

        if len(latest_stock_data_df) < self.sequence_length:
            return (
                f"Market prediction not available: requires "
                f"{self.sequence_length} days of data."
            )

        input_data = (
            latest_stock_data_df[self.feature_columns]
            .tail(self.sequence_length)
            .values
        )

        input_tensor = (
            torch.tensor(input_data, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            prediction_scaled = self.model(input_tensor).cpu().numpy()

        prediction = self.scaler_target.inverse_transform(
            prediction_scaled.reshape(-1, 1)
        )[0][0]

        last_known_close_scaled = latest_stock_data_df['Close'].iloc[-1]
        last_known_close = self.scaler_target.inverse_transform(
            np.array(last_known_close_scaled).reshape(-1, 1)
        )[0][0]

        price_change_percent = (
            ((prediction - last_known_close) / last_known_close) * 100
            if last_known_close != 0
            else 0
        )

        return (
            f"Latest known closing price: ${last_known_close:.2f}\n"
            f"Predicted next closing price: ${prediction:.2f}\n"
            f"Expected price change: {price_change_percent:.2f}%"
        )


class NewsAgent:
    """Agent responsible for fetching and analyzing news sentiment."""

    def analyze(self, company_name):
        """Fetches news and performs sentiment analysis."""
        news_data = get_news(company_name)
        headlines = [
            article['title']
            for article in news_data.get('articles', [])
            if 'title' in article
        ]

        if not headlines:
            return "No recent news found for sentiment analysis."

        sentiment_results = analyze_sentiment(headlines)

        if not sentiment_results:
            return "Sentiment analysis could not be performed for news."

        positive_count = sum(
            1 for res in sentiment_results if res['sentiment'] == 'positive'
        )
        negative_count = sum(
            1 for res in sentiment_results if res['sentiment'] == 'negative'
        )
        neutral_count = sum(
            1 for res in sentiment_results if res['sentiment'] == 'neutral'
        )

        total_analyzed = len(sentiment_results)

        summary_lines = ["Recent Headlines & Sentiment:"]
        for res in sentiment_results:
            summary_lines.append(
                f"- {res['headline']} "
                f"(Sentiment: {res['sentiment']}, Score: {res['score']:.2f})"
            )

        summary_lines.append(
            f"\nOverall News Sentiment Summary (Total Analyzed: {total_analyzed}):"
        )
        summary_lines.append(f"  Positive: {positive_count}")
        summary_lines.append(f"  Negative: {negative_count}")
        summary_lines.append(f"  Neutral: {neutral_count}")

        if positive_count > negative_count * 1.5:
            overall_sentiment = "Strongly Bullish"
        elif positive_count > negative_count:
            overall_sentiment = "Mildly Bullish"
        elif negative_count > positive_count * 1.5:
            overall_sentiment = "Strongly Bearish"
        elif negative_count > positive_count:
            overall_sentiment = "Mildly Bearish"
        else:
            overall_sentiment = "Neutral"

        summary_lines.append(f"Overall News Tone: {overall_sentiment}")

        return "\n".join(summary_lines)


class ReportAgent:
    """Agent responsible for generating the final investment research report."""

    def generate(self, market_analysis, news_analysis):
        """Combines market and news analysis into a comprehensive report."""
        report = f"""
#########################################
# AI-POWERED INVESTMENT RESEARCH REPORT #
#########################################

Market Analysis:
----------------
{market_analysis}

News Sentiment Analysis:
------------------------
{news_analysis}

Final Recommendation:
---------------------
"""
        try:
            predicted_price_str = market_analysis.split(
                "Predicted next closing price: $"
            )[1].split('\n')[0]
            predicted_price = float(predicted_price_str)
            last_known_price_str = market_analysis.split(
                "Latest known closing price: $"
            )[1].split('\n')[0]
            last_known_price = float(last_known_price_str)
            price_change_percent = (
                (predicted_price - last_known_price) / last_known_price
            ) * 100
        except (IndexError, ValueError):
            predicted_price = None
            price_change_percent = 0
            report += (
                "Cannot make a specific recommendation due to incomplete market analysis.\n"
            )
            return report

        overall_news_tone = "Neutral"
        if "Overall News Tone: " in news_analysis:
            overall_news_tone = (
                news_analysis.split("Overall News Tone: ")[1].split('\n')[0].strip()
            )

        recommendation = "HOLD"
        if predicted_price is not None:
            if price_change_percent > 1.5 and "Bullish" in overall_news_tone:
                recommendation = "BUY (Strong positive price movement + positive news)"
            elif price_change_percent > 0.5 and "Bullish" in overall_news_tone:
                recommendation = "BUY (Positive price movement + positive news)"
            elif price_change_percent < -1.5 and "Bearish" in overall_news_tone:
                recommendation = "SELL (Strong negative price movement + negative news)"
            elif price_change_percent < -0.5 and "Bearish" in overall_news_tone:
                recommendation = "SELL (Negative price movement + negative news)"
            elif -0.5 <= price_change_percent <= 0.5 and overall_news_tone == "Neutral":
                recommendation = "HOLD (Stable price, neutral news)"
            elif (
                (price_change_percent > 0.5 and "Bearish" in overall_news_tone)
                or (price_change_percent < -0.5 and "Bullish" in overall_news_tone)
            ):
                recommendation = "CAUTION (Conflicting signals from market and news)"
            else:
                recommendation = "HOLD (Moderate signals, no strong conviction)"

        report += recommendation
        report += (
            "\n\nDisclaimer: This report is for informational purposes only "
            "and not financial advice."
        )

        return report
