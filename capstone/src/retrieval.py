import os

import requests
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWSAPI_KEY_HERE":
    print(
        "Warning: NEWS_API_KEY not configured in .env file. "
        "News retrieval will be skipped. Get a key from https://newsapi.org/"
    )
    NEWS_API_KEY = None

tokenizer = None
model = None

try:
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    print("FinBERT model loaded successfully.")
except Exception as e:
    print(f"Error loading FinBERT model: {e}")
    print(
        "Please ensure you have an internet connection and "
        "the transformers library is installed correctly."
    )


def get_news(company, language='en', page_size=5, sort_by='relevancy'):
    """Fetches news headlines for a given company using NewsAPI."""
    if not NEWS_API_KEY:
        print("NewsAPI key not set. Skipping news retrieval.")
        return {"articles": []}

    url = (
        "https://newsapi.org/v2/everything?"
        f"q={company}&"
        f"language={language}&"
        f"pageSize={page_size}&"
        f"sortBy={sort_by}&"
        f"apiKey={NEWS_API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        news_data = response.json()
        print(
            f"Successfully fetched {len(news_data.get('articles', []))} "
            f"news articles for {company}."
        )
        return news_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news for {company}: {e}")
        return {"articles": []}


def analyze_sentiment(headlines):
    """
    Performs sentiment analysis on a list of headlines using FinBERT.
    Returns a list of sentiment labels and their confidence scores.
    """
    if not tokenizer or not model:
        print("FinBERT model not loaded. Skipping sentiment analysis.")
        return []

    if not headlines:
        return []

    try:
        inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        labels = ['positive', 'negative', 'neutral']

        sentiment_results = []
        for i, pred in enumerate(predictions):
            sentiment_label = labels[torch.argmax(pred)]
            confidence_score = pred[torch.argmax(pred)].item()
            sentiment_results.append({
                'headline': headlines[i],
                'sentiment': sentiment_label,
                'score': confidence_score,
            })
        print(f"Sentiment analysis complete for {len(headlines)} headlines.")
        return sentiment_results
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return []
