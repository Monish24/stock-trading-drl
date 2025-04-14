# agents/ppo_agent.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# agent class
class PPOStockTrader:
    def __init__(self, env):
        self.env = DummyVecEnv([lambda: env])
        self.model = PPO("MlpPolicy", self.env, verbose=1)

    def train(self, timesteps=10000):
        self.model.learn(total_timesteps=timesteps)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path)

    def predict(self, obs):
        return self.model.predict(obs, deterministic=True)
