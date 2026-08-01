import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def get_clean_data_path(ticker):
    """Return the absolute path to a ticker's cleaned CSV file."""
    return os.path.join(DATA_DIR, f"clean_stock_data_{ticker}.csv")


def calculate_rsi(df, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """Calculates Moving Average Convergence Divergence (MACD)."""
    df['EMA_short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    df.drop(columns=['EMA_short', 'EMA_long'], inplace=True)
    return df


def create_sequences(data, sequence_length):
    """
    Creates sequences of data for LSTM training.
    X contains features (t-sequence_length to t-1), Y contains target (t).
    """
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data.iloc[i:(i + sequence_length)].values
        y = data.iloc[i + sequence_length]['Close']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def load_and_preprocess_stock_data(
    ticker="AAPL",
    start_date="2023-01-01",
    end_date="2025-01-01",
    sequence_length=60,
    split_ratio=0.8,
):
    """
    Downloads, cleans, feature engineers, scales, and creates sequences for stock data.
    Returns:
        - scaled_df: The DataFrame with all features scaled.
        - X_train, y_train, X_test, y_test: LSTM sequences for training and testing.
        - scaler_features, scaler_target: Scaler objects for inverse transformation.
    """
    print(f"Loading data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df["Daily_Return"] = df["Close"].pct_change()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Daily_Return"].rolling(20).std()
    df = calculate_rsi(df)
    df = calculate_macd(df)

    df.dropna(inplace=True)

    features_to_scale = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return',
        'MA20', 'MA50', 'Volatility', 'RSI', 'MACD', 'Signal_Line',
    ]

    existing_features = [f for f in features_to_scale if f in df.columns]
    if not existing_features:
        raise ValueError("No valid features found for scaling. Check DataFrame columns.")

    data_for_scaling = df[existing_features].copy()

    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features_array = scaler_features.fit_transform(data_for_scaling)
    scaled_df = pd.DataFrame(
        scaled_features_array, columns=existing_features, index=df.index
    )

    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target.fit(df[['Close']])

    X, y = create_sequences(scaled_df, sequence_length)

    train_size = int(len(X) * split_ratio)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    print(
        f"Data loading and preprocessing complete. "
        f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}"
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = get_clean_data_path(ticker)
    df.to_csv(csv_path)
    print(f"Cleaned data saved to {csv_path}")

    return (
        scaled_df,
        X_train,
        y_train,
        X_test,
        y_test,
        scaler_features,
        scaler_target,
        existing_features,
    )
