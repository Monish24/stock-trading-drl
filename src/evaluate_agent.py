# src/evaluate_agent.py

import pandas as pd
from stable_baselines3 import PPO
from trading_env import SingleStockTradingEnv

# Load data
df = pd.read_csv("data/raw/RELIANCE_NS.csv", skiprows=2)
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close']).reset_index(drop=True)

# Load environment and model
env = SingleStockTradingEnv(df)
model = PPO.load("models/ppo_reliance")

# Evaluate
obs, _ = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    print(f"Step: {env.current_step}, Action: {action}, Cash: {env.cash}, Shares: {env.shares}, Total: {env.total_asset}")

print("✅ Evaluation done!")
