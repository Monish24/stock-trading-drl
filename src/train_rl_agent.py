# src/train_rl_agent.py

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from trading_env import SingleStockTradingEnv

# Load stock data
df = pd.read_csv("data/raw/RELIANCE_NS.csv", skiprows=2)
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close']).reset_index(drop=True)

# Wrap the env
env = DummyVecEnv([lambda: Monitor(SingleStockTradingEnv(df))])

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=100_000, reset_num_timesteps=True, progress_bar=True, log_interval=10)

# Save model
model.save("models/ppo_reliance")
