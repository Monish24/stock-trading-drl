import os
import pandas as pd
import numpy as np
from pathlib import Path
from environment import TradingEnvironment
from agent import TradingAgent
from utils import plot_performance

# Parameters? maybe
DATA_DIR = Path("data/rawrl")
TICKER = "RELIANCE.NS"
EPISODES = 10
INITIAL_BALANCE = 10000  #Rupees

def load_data(ticker):
    filepath = DATA_DIR / f"{ticker}.csv"
    df = pd.read_csv(filepath, skiprows=1)
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.rename(columns={"Date": "date", "Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.dropna().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def main():
    # Load historical data
    data = load_data(TICKER)

    # Create environment
    env = TradingEnvironment(data, initial_balance=INITIAL_BALANCE)

    # Create agent
    agent = TradingAgent(state_size=env.state_size, action_size=env.action_size)

    # Training loop
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward:.2f}")

    # Final performance
    plot_performance(env.history, TICKER)

if __name__ == "__main__":
    main()
