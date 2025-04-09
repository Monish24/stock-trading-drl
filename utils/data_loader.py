import pandas as pd
from pathlib import Path

def load_single_stock(ticker, data_dir=Path("data/raw")):
    """Load and preprocess a single stock's historical data from CSV."""
    filepath = data_dir / f"{ticker}.csv"
    
    try:
        df = pd.read_csv(filepath, skiprows=1)
    except FileNotFoundError:
        raise ValueError(f"No file found for ticker: {ticker} at {filepath}")
    
    df.columns = df.iloc[0]
    df = df[1:]

    df = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high", 
        "Low": "low", "Close": "close", "Volume": "volume"
    })

    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.dropna().reset_index(drop=True)

    df["date"] = pd.to_datetime(df["date"])
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    return df

def load_multiple_stocks(tickers, data_dir=Path("data/raw")):
    """Load and preprocess multiple stocks' historical data as a dictionary."""
    stock_data = {}
    for ticker in tickers:
        stock_data[ticker] = load_single_stock(ticker, data_dir)
    return stock_data
