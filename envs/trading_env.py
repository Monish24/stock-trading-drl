"""
Stock Trading Environment compatible with Gymnasium / Stable-Baselines3.

Supports both single-stock and multi-stock portfolio trading.
Observation = price features + technical indicators + sentiment scores.
Action = continuous portfolio weights across assets + cash.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import INITIAL_CASH, TRANSACTION_COST, WINDOW_SIZE


# Features the agent observes (per stock, per timestep)
PRICE_FEATURES = [
    "returns", "log_returns",
    "sma_5", "sma_20", "sma_60", "ema_12",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width",
    "volume_ratio", "volatility_20",
]

SENTIMENT_FEATURES = [
    "avg_sentiment", "max_sentiment", "min_sentiment",
    "negative_ratio", "positive_ratio", "news_count",
]


class StockTradingEnv(gym.Env):
    """
    Multi-stock portfolio trading environment.

    State: flattened window of features for all stocks + current portfolio weights.
    Action: target portfolio weights (n_stocks + 1 for cash), via softmax.
    Reward: log portfolio return minus transaction costs.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        tickers: list[str] = None,
        initial_cash: float = INITIAL_CASH,
        transaction_cost: float = TRANSACTION_COST,
        window_size: int = WINDOW_SIZE,
        use_sentiment: bool = True,
    ):
        super().__init__()

        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.window_size = window_size

        # Separate data per ticker
        self.tickers = tickers or sorted(data["ticker"].unique().tolist())
        self.n_assets = len(self.tickers)

        # Determine which feature columns to use
        feature_cols = [c for c in PRICE_FEATURES if c in data.columns]
        if use_sentiment:
            feature_cols += [c for c in SENTIMENT_FEATURES if c in data.columns]
        self.feature_cols = feature_cols
        self.n_features = len(feature_cols)

        # Build per-ticker arrays (aligned by date)
        self._build_data(data)

        # Observation: (window * n_assets * n_features) + (n_assets + 1 for current weights)
        obs_size = self.window_size * self.n_assets * self.n_features + (self.n_assets + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action: target weights for each asset (softmax applied internally)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        # State tracking
        self.current_step = 0
        self.portfolio_weights = None
        self.portfolio_value = 0.0
        self.portfolio_values = []
        self.total_transaction_costs = 0.0

    def _build_data(self, data: pd.DataFrame):
        """Align all ticker data to common dates and extract feature matrices."""
        ticker_dfs = {}
        for t in self.tickers:
            tdf = data[data["ticker"] == t].sort_values("date").reset_index(drop=True)
            ticker_dfs[t] = tdf

        # Find common dates
        date_sets = [set(df["date"].values) for df in ticker_dfs.values()]
        common_dates = sorted(set.intersection(*date_sets))
        self.dates = pd.DatetimeIndex(common_dates)
        self.n_steps = len(common_dates)

        # Build feature tensor: (n_steps, n_assets, n_features)
        self.features = np.zeros((self.n_steps, self.n_assets, self.n_features), dtype=np.float32)
        self.close_prices = np.zeros((self.n_steps, self.n_assets), dtype=np.float32)

        for i, ticker in enumerate(self.tickers):
            df = ticker_dfs[ticker]
            df = df[df["date"].isin(common_dates)].sort_values("date").reset_index(drop=True)
            self.features[:, i, :] = df[self.feature_cols].values.astype(np.float32)
            self.close_prices[:, i] = df["close"].values.astype(np.float32)

        # Replace NaN/Inf BEFORE normalizing
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        self.close_prices = np.nan_to_num(self.close_prices, nan=1.0, posinf=1.0, neginf=1.0)

        # Normalize features (z-score per feature across all data)
        self.feat_mean = self.features.mean(axis=0, keepdims=True)
        self.feat_std = self.features.std(axis=0, keepdims=True) + 1e-8
        self.features = (self.features - self.feat_mean) / self.feat_std

        # Final safety check
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
    
    def _get_obs(self) -> np.ndarray:
        """Build observation from current window + portfolio weights."""
        start = self.current_step - self.window_size
        window = self.features[start:self.current_step]  # (window, n_assets, n_features)
        flat = window.flatten()
        obs = np.concatenate([flat, self.portfolio_weights])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        # Start fully in cash
        self.portfolio_weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.portfolio_weights[-1] = 1.0  # 100% cash
        self.portfolio_value = self.initial_cash
        self.portfolio_values = [self.initial_cash]
        self.total_transaction_costs = 0.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # Convert action to target weights (softmax for valid allocation)
        action = np.clip(action, 0, None)
        cash_weight = max(0.0, 1.0 - action.sum())
        target_weights = np.append(action, cash_weight)
        target_weights = target_weights / (target_weights.sum() + 1e-8)

        # Transaction cost = proportional to weight changes
        weight_change = np.abs(target_weights - self.portfolio_weights)
        cost = self.transaction_cost * weight_change.sum() * self.portfolio_value
        self.total_transaction_costs += cost

        # Update weights
        self.portfolio_weights = target_weights

        # Price returns for this step
        if self.current_step < self.n_steps:
            price_returns = (
                self.close_prices[self.current_step]
                / self.close_prices[self.current_step - 1]
            ) - 1.0
        else:
            price_returns = np.zeros(self.n_assets)

        # Portfolio return (weighted sum of asset returns, cash earns nothing)
        asset_weights = self.portfolio_weights[:self.n_assets]
        portfolio_return = np.dot(asset_weights, price_returns)

        # Update portfolio value
        new_value = self.portfolio_value * (1 + portfolio_return) - cost
        self.portfolio_value = max(new_value, 0.0)
        self.portfolio_values.append(self.portfolio_value)

        # Reward: log return (encourages geometric growth)
        reward = np.log(self.portfolio_value / self.portfolio_values[-2] + 1e-8)

        self.current_step += 1
        done = self.current_step >= self.n_steps
        truncated = False

        info = {
            "portfolio_value": self.portfolio_value,
            "transaction_cost": cost,
            "date": self.dates[min(self.current_step - 1, len(self.dates) - 1)],
        }

        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, truncated, info

    def get_portfolio_history(self) -> pd.DataFrame:
        """Return portfolio value over time as a DataFrame."""
        n = min(len(self.portfolio_values), len(self.dates))
        return pd.DataFrame({
            "date": self.dates[:n],
            "portfolio_value": self.portfolio_values[:n],
        })
