"""
Fetch US stock data from Yahoo Finance and compute technical indicators.

Usage:
    python -m data.fetch_data
    python -m data.fetch_data --tickers AAPL MSFT NVDA
"""
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    TICKERS, START_DATE, END_DATE, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    SMA_WINDOWS, EMA_WINDOW, RSI_WINDOW, BB_WINDOW, VOLATILITY_WINDOW,
)


def fetch_stock_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data for multiple tickers from Yahoo Finance."""
    print(f"Fetching {len(tickers)} stocks from {start} to {end}...")
    raw = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True)

    records = []
    for ticker in tickers:
        try:
            df = raw[ticker].copy()
            df["ticker"] = ticker
            df.index.name = "date"
            records.append(df.reset_index())
        except Exception as e:
            print(f"  Warning: {ticker} failed — {e}")

    data = pd.concat(records, ignore_index=True)
    data.columns = [c.lower() for c in data.columns]
    print(f"  Fetched {len(data):,} rows for {data['ticker'].nunique()} stocks")
    return data


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to a single stock's DataFrame."""
    df = df.sort_values("date").copy()

    # Returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Moving averages
    for w in SMA_WINDOWS:
        df[f"sma_{w}"] = SMAIndicator(df["close"], window=w).sma_indicator()
    df["ema_12"] = EMAIndicator(df["close"], window=EMA_WINDOW).ema_indicator()

    # RSI
    df["rsi_14"] = RSIIndicator(df["close"], window=RSI_WINDOW).rsi()

    # MACD
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(df["close"], window=BB_WINDOW)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]

    # Volume features
    df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

    # Volatility
    df["volatility_20"] = df["returns"].rolling(window=VOLATILITY_WINDOW).std()

    return df


def process_data(tickers: list[str] = None, start: str = None, end: str = None) -> pd.DataFrame:
    """Full pipeline: fetch → indicators → clean → save."""
    tickers = tickers or TICKERS
    start = start or START_DATE
    end = end or END_DATE

    # Fetch
    stock_data = fetch_stock_data(tickers, start, end)

    # Save raw
    raw_path = RAW_DATA_DIR / "stock_data_raw.csv"
    stock_data.to_csv(raw_path, index=False)
    print(f"  Raw data saved to {raw_path}")

    # Add indicators
    print("Adding technical indicators...")
    stock_data = stock_data.groupby("ticker", group_keys=False).apply(
        add_technical_indicators, include_groups=False
    )
    # Restore the ticker column that gets excluded with include_groups=False
    if "ticker" not in stock_data.columns:
        stock_data = stock_data.reset_index()

    stock_data = stock_data.dropna().reset_index(drop=True)

    # Save processed
    processed_path = PROCESSED_DATA_DIR / "stock_data_processed.csv"
    stock_data.to_csv(processed_path, index=False)
    print(f"  Processed data saved to {processed_path}")
    print(f"  Shape: {stock_data.shape} | Features: {len(stock_data.columns)}")

    return stock_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and process stock data")
    parser.add_argument("--tickers", nargs="+", default=None, help="Stock tickers")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    data = process_data(tickers=args.tickers, start=args.start, end=args.end)
    print(f"\nDone! {len(data):,} rows ready.")
