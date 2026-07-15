# data_loader.py

import yfinance as yf

def load_stock_data(ticker="AAPL"):

    df = yf.download(
        ticker,
        start="2023-01-01",
        end="2025-01-01",
        auto_adjust=False
    )

    return df
