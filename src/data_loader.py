# src/data_loader.py
import yfinance as yf
import pandas as pd

def download_stock_data(ticker="RELIANCE.NS", start="2019-01-01", end="2024-01-01"):
    data = yf.download(ticker, start=start, end=end)
    data.to_csv(f"data/raw/{ticker.replace('.', '_')}.csv")
    print(f"Saved data/raw/{ticker.replace('.', '_')}.csv")

if __name__ == "__main__":
    download_stock_data("RELIANCE.NS")
