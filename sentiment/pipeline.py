"""
News sentiment pipeline using FinBERT.
Fetches financial news headlines via GNews and scores them.

Usage:
    python -m sentiment.pipeline
    python -m sentiment.pipeline --tickers AAPL NVDA --start 2024-01-01 --end 2024-06-30
"""
import argparse
import time
import pandas as pd
import torch
from datetime import date, timedelta
from pathlib import Path
from gnews import GNews
from transformers import pipeline as hf_pipeline

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    TICKERS, TICKER_TO_COMPANY, SENTIMENT_MODEL,
    NEWS_MAX_ARTICLES, NEWS_RATE_LIMIT, SENTIMENT_DATA_DIR,
)


class SentimentAnalyzer:
    """FinBERT-based financial sentiment scorer."""

    def __init__(self, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading FinBERT on {device}...")
        device_id = 0 if device == "cuda" else -1
        self.pipe = hf_pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            device=device_id,
        )
        print("  FinBERT ready!")

    def score(self, texts: list[str]) -> list[dict]:
        """Score a list of texts. Returns list of {label, score, numeric_score}."""
        results = self.pipe(texts)
        scored = []
        for res in results:
            numeric = res["score"]
            if res["label"] == "negative":
                numeric = -numeric
            elif res["label"] == "neutral":
                numeric = 0.0
            scored.append({
                "label": res["label"],
                "raw_score": res["score"],
                "sentiment_score": round(numeric, 4),
            })
        return scored


def fetch_news(ticker: str, company: str, start: date, end: date,
               max_articles: int = NEWS_MAX_ARTICLES) -> list[dict]:
    """Fetch news headlines for a stock from GNews."""
    gn = GNews(
        language="en", country="US",
        start_date=start, end_date=end,
        max_results=max_articles,
    )
    articles = gn.get_news(f"{company} stock")
    return articles


def build_monthly_windows(start_date: str, end_date: str) -> list[tuple[date, date]]:
    """Generate (start, end) tuples for each month in the range."""
    windows = []
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    current = start.replace(day=1)

    while current < end:
        month_start = current
        if current.month == 12:
            month_end = date(current.year, 12, 31)
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
            month_end = next_month - timedelta(days=1)

        if month_end > end:
            month_end = end
        windows.append((month_start, month_end))
        current = next_month

    return windows


def run_sentiment_pipeline(
    tickers: list[str] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2025-04-01",
    device: str = None,
) -> pd.DataFrame:
    """Full pipeline: fetch news → score with FinBERT → save."""
    tickers = tickers or TICKERS
    analyzer = SentimentAnalyzer(device=device)

    windows = build_monthly_windows(start_date, end_date)
    print(f"\nFetching sentiment for {len(tickers)} stocks across {len(windows)} months")
    print(f"Estimated time: ~{len(windows) * len(tickers) * 1 // 60} minutes\n")

    all_records = []
    errors = 0

    for i, (w_start, w_end) in enumerate(windows):
        print(f"[{i+1}/{len(windows)}] {w_start} → {w_end}", end=" | ")

        for ticker in tickers:
            company = TICKER_TO_COMPANY.get(ticker, ticker)
            try:
                articles = fetch_news(ticker, company, w_start, w_end)
                if articles:
                    titles = [a.get("title", "") for a in articles]
                    pub_dates = [a.get("published date", "") for a in articles]
                    scores = analyzer.score(titles)

                    for score, title, pub_date in zip(scores, titles, pub_dates):
                        all_records.append({
                            "ticker": ticker,
                            "date": pub_date,
                            "title": title,
                            "sentiment_score": score["sentiment_score"],
                            "sentiment_label": score["label"],
                            "confidence": score["raw_score"],
                        })
            except Exception:
                errors += 1

            time.sleep(NEWS_RATE_LIMIT)

        print(f"Total: {len(all_records)} | Errors: {errors}")

    df = pd.DataFrame(all_records)
    if not df.empty:
        save_path = SENTIMENT_DATA_DIR / "sentiment_raw.csv"
        df.to_csv(save_path, index=False)
        print(f"\nSaved {len(df)} sentiment scores to {save_path}")
    else:
        print("\nNo sentiment data collected!")

    return df


def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate article-level sentiment to daily per-ticker averages."""
    df = df.copy()
    df["parsed_date"] = pd.to_datetime(df["date"], format="mixed")
    df["date_only"] = df["parsed_date"].dt.normalize()

    daily = df.groupby(["ticker", "date_only"]).agg(
        avg_sentiment=("sentiment_score", "mean"),
        max_sentiment=("sentiment_score", "max"),
        min_sentiment=("sentiment_score", "min"),
        news_count=("sentiment_score", "count"),
        negative_ratio=("sentiment_label", lambda x: (x == "negative").mean()),
        positive_ratio=("sentiment_label", lambda x: (x == "positive").mean()),
    ).reset_index()

    save_path = SENTIMENT_DATA_DIR / "sentiment_daily.csv"
    daily.to_csv(save_path, index=False)
    print(f"Daily sentiment saved to {save_path} ({len(daily)} rows)")
    return daily


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sentiment pipeline")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-04-01")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()

    raw_df = run_sentiment_pipeline(
        tickers=args.tickers, start_date=args.start,
        end_date=args.end, device=args.device,
    )
    if not raw_df.empty:
        aggregate_daily_sentiment(raw_df)
