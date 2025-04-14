from envs.single_stock_env import SingleStockTradingEnv
from stable_baselines3 import PPO
import pandas as pd

df = pd.read_csv("data/RELIANCE.csv", skiprows=2)
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close']).reset_index(drop=True)

# environment creation
env = SingleStockTradingEnv(df)
model = PPO.load("models/ppo_reliance")

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    print(f"Step: {env.current_step}, Action: {action}, Cash: {env.cash}, Shares: {env.shares}, Total: {env.total_asset}")
