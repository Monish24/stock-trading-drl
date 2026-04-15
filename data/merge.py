"""
Merge stock price data with sentiment scores and prepare train/test splits.

Usage:
    python -m data.merge
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import PROCESSED_DATA_DIR, SENTIMENT_DATA_DIR, TRAIN_END_DATE


SENTIMENT_COLS = [
    "avg_sentiment", "max_sentiment", "min_sentiment",
    "negative_ratio", "positive_ratio",
]


def merge_stock_and_sentiment(
    stock_path: str = None,
    sentiment_path: str = None,
) -> pd.DataFrame:
    """Left-join stock data with daily sentiment. Missing sentiment = neutral (0)."""
    stock_path = stock_path or str(PROCESSED_DATA_DIR / "stock_data_processed.csv")
    sentiment_path = sentiment_path or str(SENTIMENT_DATA_DIR / "sentiment_daily.csv")

    stock_data = pd.read_csv(stock_path, parse_dates=["date"])

    try:
        sentiment = pd.read_csv(sentiment_path, parse_dates=["date_only"])
    except FileNotFoundError:
        print("No sentiment data found — using price features only.")
        for col in SENTIMENT_COLS + ["news_count"]:
            stock_data[col] = 0.0
        return stock_data

    merged = stock_data.merge(
        sentiment,
        left_on=["ticker", "date"],
        right_on=["ticker", "date_only"],
        how="left",
    )
    merged[SENTIMENT_COLS] = merged[SENTIMENT_COLS].fillna(0.0)
    merged["news_count"] = merged["news_count"].fillna(0)
    merged.drop(columns=["date_only"], errors="ignore", inplace=True)

    has_sent = (merged["news_count"] > 0).sum()
    print(f"Merged: {merged.shape} | Rows with sentiment: {has_sent} ({has_sent/len(merged)*100:.1f}%)")
    return merged


def train_test_split(
    df: pd.DataFrame, split_date: str = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date — no data leakage."""
    split_date = split_date or TRAIN_END_DATE
    split = pd.Timestamp(split_date)

    train = df[df["date"] < split].copy().reset_index(drop=True)
    test = df[df["date"] >= split].copy().reset_index(drop=True)

    print(f"Train: {len(train):,} rows ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"Test:  {len(test):,} rows ({test['date'].min().date()} → {test['date'].max().date()})")
    return train, test


if __name__ == "__main__":
    merged = merge_stock_and_sentiment()
    save_path = PROCESSED_DATA_DIR / "merged_data.csv"
    merged.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")

    train, test = train_test_split(merged)
    train.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    print("Train/test splits saved.")
