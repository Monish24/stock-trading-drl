# Import the single stock trading environment and PPO algorithm from Stable Baselines3
from envs.single_stock_env import SingleStockTradingEnv
from stable_baselines3 import PPO
import pandas as pd

# Load the stock data for Reliance, skipping the first two rows (usually metadata or headers)
df = pd.read_csv("data/RELIANCE.csv", skiprows=2)

# Rename the columns to standard financial OHLCV format
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Convert the 'Close' column to numeric, coercing errors (e.g., invalid strings become NaN)
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

# Drop rows where 'Close' price is missing, and reset the DataFrame index
df = df.dropna(subset=['Close']).reset_index(drop=True)

# Initialize the trading environment using the processed stock data
env = SingleStockTradingEnv(df)

# Load a pre-trained PPO model from the saved path
model = PPO.load("models/ppo_reliance")

# Reset the environment to start a new episode
obs, _ = env.reset()
done = False

# Run the agent in the environment until the episode ends
while not done:
    # Predict the next action using the trained model (deterministic ensures reproducibility)
    action, _ = model.predict(obs, deterministic=True)

    # Apply the action to the environment and get the result
    obs, reward, done, _, _ = env.step(action)

    # Print the current step, chosen action, cash balance, shares held, and total assets
    print(f"Step: {env.current_step}, Action: {action}, Cash: {env.cash}, Shares: {env.shares}, Total: {env.total_asset}")
