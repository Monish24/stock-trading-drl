# src/download_all_stocks_yf.py

import yfinance as yf
import pandas as pd
import os
from time import sleep

def download_all_stocks(csv_path="data/nse_50_stocks.csv", start="2019-01-01", end="2024-01-01"):
    df = pd.read_csv(csv_path)
    symbols = df["Symbol"].dropna().unique()

    os.makedirs("data/raw", exist_ok=True)

    for symbol in symbols:
        filename = f"data/raw/{symbol.replace('.', '_')}.csv"
        if os.path.exists(filename):
            print(f"[SKIP] {symbol} already exists.")
            continue
        try:
            print(f"[DOWNLOADING] {symbol}...")
            data = yf.download(symbol, start=start, end=end)
            if not data.empty:
                data.to_csv(filename)
                print(f"[SAVED] {filename}")
            else:
                print(f"[EMPTY] No data for {symbol}")
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
        sleep(1)  # avoid rate limits

if __name__ == "__main__":
    download_all_stocks()
