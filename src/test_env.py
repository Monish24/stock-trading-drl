from trading_env import SingleStockTradingEnv
import pandas as pd
import random

# 🧼 Read CSV and skip first two rows (junk header and fake header)
df_raw = pd.read_csv("data/raw/RELIANCE_NS.csv", skiprows=2)

# 💡 Rename the columns manually (based on actual yfinance structure)
df_raw.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# 🧹 Clean and convert data
df_raw['Close'] = pd.to_numeric(df_raw['Close'], errors='coerce')
df_raw = df_raw.dropna(subset=['Close'])
df_raw = df_raw.reset_index(drop=True)

# ✅ Initialize environment
env = SingleStockTradingEnv(df_raw)

obs = env.reset()
done = False
while not done:
    action = random.choice([0, 1, 2])
    obs, reward, done, _ = env.step(action)
    env.render()
