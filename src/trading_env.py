# src/trading_env.py

import gymnasium as gym
import numpy as np
import pandas as pd

class SingleStockTradingEnv(gym.Env):
    def __init__(self, data, window_size=10, initial_cash=100000):
        super(SingleStockTradingEnv, self).__init__()

        self.data = data.reset_index()
        self.window_size = window_size
        self.initial_cash = initial_cash

        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size + 2,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cash = self.initial_cash
        self.shares = 0
        self.current_step = self.window_size
        self.total_asset = self.cash
        obs = self._get_observation()
        return obs, {}


    def _get_observation(self):
        window = self.data["Close"].iloc[self.current_step - self.window_size:self.current_step].values
        normalized_window = (window - np.mean(window)) / (np.std(window) + 1e-8)

        obs = np.concatenate([normalized_window, [self.cash / 1e6, self.shares / 100]])
        return obs.astype(np.float32)


    def step(self, action):
        price = self.data["Close"].iloc[self.current_step]

        prev_total_asset = self.cash + self.shares * price

        # Execute action
        if action == 1:  # Buy
            if self.cash >= price:
                self.cash -= price
                self.shares += 1
        elif action == 2:  # Sell
            if self.shares > 0:
                self.cash += price
                self.shares -= 1

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        price = self.data["Close"].iloc[self.current_step]

        self.total_asset = self.cash + self.shares * price
        reward = self.total_asset - prev_total_asset

        # Encourage action
        if action == 0:
            reward -= 0.01  # discourage doing nothing

        obs = self._get_observation()
        return obs, reward, done, False, {}

    def render(self):
        print(f"Step: {self.current_step}, Cash: {self.cash}, Shares: {self.shares}, Total Asset: {self.total_asset}")
