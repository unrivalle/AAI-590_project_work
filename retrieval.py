# retrieval.py

import requests
import os
os.getenv("API_KEY")

NEWS_API = "API_KEY"

def get_news(company):

    url = f"https://newsapi.org/v2/everything?q={company}&apiKey={NEWS_API}"

    response = requests.get(url)

    return response.json()
