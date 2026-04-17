"""
Main training and evaluation script.

Trains DRL agents (PPO, A2C, SAC) on US stock data with optional sentiment,
evaluates on out-of-sample test data, and compares against buy-and-hold baseline.

Usage:
    python main.py                              # Full pipeline with defaults
    python main.py --algorithms PPO SAC         # Train specific algorithms
    python main.py --no-sentiment               # Price-only (no sentiment features)
    python main.py --timesteps 200000           # More training
    python main.py --skip-data                  # Skip data fetch (use cached)
"""
import os
import argparse
import pandas as pd
from pathlib import Path

from configs.config import (
    TICKERS, INITIAL_CASH, TOTAL_TIMESTEPS,
    PROCESSED_DATA_DIR, TRAIN_END_DATE,
)
from data.fetch_data import process_data
from data.merge import merge_stock_and_sentiment, train_test_split
from envs.trading_env import StockTradingEnv
from agents.drl_agent import DRLAgent
from utils.evaluation import (
    full_evaluation, compute_buy_and_hold, compute_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stock Trading DRL")
    parser.add_argument("--algorithms", nargs="+", default=["PPO", "A2C", "SAC"])
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--no-sentiment", action="store_true", help="Disable sentiment features")
    parser.add_argument("--skip-data", action="store_true", help="Use cached data files")
    parser.add_argument("--split-date", default=TRAIN_END_DATE)
    return parser.parse_args()


def main():
    args = parse_args()
    tickers = args.tickers or TICKERS
    use_sentiment = not args.no_sentiment

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    if args.skip_data:
        print("Loading cached data...")
        merged_path = PROCESSED_DATA_DIR / "merged_data.csv"
        if not merged_path.exists():
            raise FileNotFoundError(f"{merged_path} not found. Run without --skip-data first.")
        merged = pd.read_csv(merged_path, parse_dates=["date"])
    else:
        print("Step 1: Fetching & processing stock data...")
        process_data(tickers=tickers)
        print("\nStep 2: Merging with sentiment data...")
        merged = merge_stock_and_sentiment()
        merged.to_csv(PROCESSED_DATA_DIR / "merged_data.csv", index=False)

    # ── Step 2: Train/Test Split ──────────────────────────────────────────────
    print(f"\nSplitting data at {args.split_date}...")
    train_data, test_data = train_test_split(merged, split_date=args.split_date)

    # ── Step 3: Create Environments ───────────────────────────────────────────
    print("\nCreating trading environments...")
    # Use only tickers that actually exist in the data
    available_tickers = sorted(train_data["ticker"].unique().tolist())
    print(f"  Available tickers: {len(available_tickers)} — {available_tickers}")
    train_env = StockTradingEnv(train_data, tickers=available_tickers, use_sentiment=use_sentiment)
    test_env_template = lambda: StockTradingEnv(test_data, tickers=available_tickers, use_sentiment=use_sentiment)

    print(f"  Assets: {train_env.n_assets}")
    print(f"  Features per asset: {train_env.n_features}")
    print(f"  Observation space: {train_env.observation_space.shape}")
    print(f"  Action space: {train_env.action_space.shape}")
    print(f"  Sentiment: {'enabled' if use_sentiment else 'disabled'}")

    # ── Step 4: Train Agents ──────────────────────────────────────────────────
    all_results = {}

    for algo_name in args.algorithms:
        print(f"\n{'='*60}")
        print(f"  Training {algo_name}")
        print(f"{'='*60}")

        agent = DRLAgent(train_env, algorithm=algo_name)
        agent.train(timesteps=args.timesteps)
        agent.save()

        # Evaluate on fresh test env
        test_env = test_env_template()
        result = agent.evaluate(test_env)
        all_results[algo_name] = result["portfolio_values"]

    # ── Step 5: Buy-and-Hold Baseline ─────────────────────────────────────────
    print("\nComputing buy-and-hold baseline...")
    baseline_env = test_env_template()
    bh_values = compute_buy_and_hold(baseline_env.close_prices, INITIAL_CASH)
    all_results["Buy & Hold"] = bh_values

    # ── Step 6: Evaluation ────────────────────────────────────────────────────
    print("\nFinal evaluation...")
    # Use dates from test env for x-axis
    baseline_env_dates = test_env_template()
    baseline_env_dates.reset()
    dates = baseline_env_dates.dates

    suffix = "with_sentiment" if use_sentiment else "price_only"
    full_evaluation(all_results, dates=dates, save_prefix=f"comparison_{suffix}")

    print("\nDone! Check the results/ folder for plots and metrics.")


if __name__ == "__main__":
    main()
