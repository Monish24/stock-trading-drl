import numpy as np
import gym
from gym import spaces

class MultiStockTradingEnv(gym.Env):
    def __init__(self, df, stock_names, initial_balance=100000):
        super(MultiStockTradingEnv, self).__init__()
        self.df = df
        self.stock_names = stock_names
        self.num_stocks = len(stock_names)
        self.initial_balance = initial_balance
        self.max_steps = len(df) - 1

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_stocks,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2 * self.num_stocks + 1,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.num_stocks)
        self.total_value = self.balance
        self.trades = []

        return self._get_observation()

    def _get_observation(self):
        prices = self.df.iloc[self.current_step][self.stock_names].values
        obs = np.concatenate([[self.balance], prices, self.holdings])
        return obs.astype(np.float32)

    def _get_portfolio_value(self, prices):
        return self.balance + np.sum(prices * self.holdings)

    def step(self, actions):
        prices = self.df.iloc[self.current_step][self.stock_names].values
        actions = np.clip(actions, -1, 1)

        for i in range(self.num_stocks):
            action = actions[i]
            if action > 0:
                # Buy
                buy_amount = self.balance * action
                shares_bought = buy_amount // prices[i]
                self.holdings[i] += shares_bought
                self.balance -= shares_bought * prices[i]
            elif action < 0:
                # Sell
                shares_sold = self.holdings[i] * (-action)
                self.holdings[i] -= shares_sold
                self.balance += shares_sold * prices[i]

        self.current_step += 1
        done = self.current_step >= self.max_steps
        next_prices = self.df.iloc[self.current_step][self.stock_names].values
        next_portfolio_value = self._get_portfolio_value(next_prices)

        reward = next_portfolio_value - self.total_value
        self.total_value = next_portfolio_value

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance}")
        print(f"Holdings: {self.holdings}")
        print(f"Total Value: {self.total_value}")
