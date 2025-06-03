# src/portfolio_trading_env.py
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class PortfolioTradingEnv(gym.Env):
    """
    Multi-stock portfolio trading environment for reinforcement learning.
    
    This environment demonstrates key RL concepts:
    - Sequential decision making under uncertainty
    - Continuous action spaces (portfolio weights)
    - Complex state representations (multiple time series)
    - Risk-adjusted reward functions
    """
    
    def __init__(self, 
                 data_dict: Dict[str, pd.DataFrame], 
                 window_size: int = 20,
                 initial_cash: float = 100000,
                 transaction_cost: float = 0.001,
                 risk_aversion: float = 0.5):
        """
        Initialize the portfolio trading environment.
        
        Args:
            data_dict: Dictionary mapping stock symbols to their price DataFrames
            window_size: Number of historical time steps to include in observations
            initial_cash: Starting cash amount
            transaction_cost: Cost per transaction as fraction of trade value
            risk_aversion: How much to penalize volatility in reward calculation
        """
        super(PortfolioTradingEnv, self).__init__()
        
        # Store environment parameters
        self.data_dict = {symbol: df.reset_index() for symbol, df in data_dict.items()}
        self.symbols = list(data_dict.keys())
        self.n_assets = len(self.symbols)
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.risk_aversion = risk_aversion
        
        # Validate that all data has the same length and dates
        self._validate_data()
        
        # Define action space: continuous portfolio weights
        # Each action represents target allocation percentage for each stock
        # Values should sum to <= 1.0 (remainder stays in cash)
        self.action_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.n_assets,), 
            dtype=np.float32
        )
        
        # Define observation space: price windows + technical indicators + portfolio state
        # For each stock: window_size price returns + 3 technical indicators
        # Plus: current portfolio weights + cash ratio + portfolio volatility
        obs_size = (
            self.n_assets * (self.window_size + 3) +  # Price data + technical indicators
            self.n_assets +  # Current portfolio weights
            2  # Cash ratio + portfolio volatility
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Initialize tracking variables
        self.reset()
    
    def _validate_data(self):
        """Ensure all stock data has consistent length and timestamps."""
        lengths = [len(df) for df in self.data_dict.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All stock data must have the same length")
        self.data_length = lengths[0]
        
        if self.data_length <= self.window_size:
            raise ValueError(f"Data length ({self.data_length}) must be greater than window_size ({self.window_size})")
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset portfolio state
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets)  # Number of shares for each stock
        self.portfolio_weights = np.zeros(self.n_assets)  # Current allocation percentages
        self.current_step = self.window_size
        
        # Initialize tracking for performance metrics
        self.portfolio_values = []
        self.total_transaction_costs = 0.0
        
        # Calculate initial portfolio value
        self._update_portfolio_value()
        
        return self._get_observation(), {}
    
    def _get_current_prices(self) -> np.ndarray:
        """Get current prices for all stocks."""
        prices = np.array([
            self.data_dict[symbol]["Close"].iloc[self.current_step] 
            for symbol in self.symbols
        ])
        return prices
    
    def _calculate_returns(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate price returns over specified window."""
        if window <= 1:
            return np.zeros_like(prices)
        
        past_prices = np.array([
            self.data_dict[symbol]["Close"].iloc[self.current_step - window:self.current_step].values
            for symbol in self.symbols
        ])
        
        # Calculate returns as percentage changes
        returns = np.diff(past_prices, axis=1) / (past_prices[:, :-1] + 1e-8)
        return returns
    
    def _calculate_technical_indicators(self) -> np.ndarray:
        """Calculate technical indicators for each stock."""
        indicators = []
        
        for symbol in self.symbols:
            prices = self.data_dict[symbol]["Close"].iloc[
                max(0, self.current_step - 50):self.current_step
            ].values
            
            if len(prices) < 10:
                # Not enough data for indicators
                indicators.extend([0.0, 0.0, 0.0])
                continue
            
            # Simple moving average ratio (current price / 20-day MA)
            ma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
            ma_ratio = prices[-1] / (ma_20 + 1e-8) - 1.0
            
            # Price volatility (20-day rolling standard deviation of returns)
            if len(prices) >= 20:
                returns = np.diff(prices[-21:]) / (prices[-21:-1] + 1e-8)
                volatility = np.std(returns)
            else:
                volatility = 0.0
            
            # Momentum (10-day price change)
            momentum = (prices[-1] / (prices[-10] + 1e-8) - 1.0) if len(prices) >= 10 else 0.0
            
            indicators.extend([ma_ratio, volatility, momentum])
        
        return np.array(indicators)
    
    def _get_observation(self) -> np.ndarray:
        """Construct the current observation vector."""
        obs_components = []
        
        # Add price return windows for each stock
        returns = self._calculate_returns(self._get_current_prices(), self.window_size)
        for i in range(self.n_assets):
            if returns.shape[1] > 0:
                # Normalize returns to help neural network training
                stock_returns = returns[i]
                normalized_returns = (stock_returns - np.mean(stock_returns)) / (np.std(stock_returns) + 1e-8)
                obs_components.extend(normalized_returns)
            else:
                obs_components.extend([0.0] * (self.window_size - 1))
        
        # Add technical indicators
        technical_indicators = self._calculate_technical_indicators()
        obs_components.extend(technical_indicators)
        
        # Add current portfolio state
        obs_components.extend(self.portfolio_weights)
        
        # Add cash ratio (proportion of portfolio in cash)
        total_value = self.cash + np.sum(self.holdings * self._get_current_prices())
        cash_ratio = self.cash / (total_value + 1e-8)
        obs_components.append(cash_ratio)
        
        # Add portfolio volatility (risk measure)
        portfolio_volatility = self._calculate_portfolio_volatility()
        obs_components.append(portfolio_volatility)
        
        return np.array(obs_components, dtype=np.float32)
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility over recent history."""
        if len(self.portfolio_values) < 10:
            return 0.0
        
        recent_values = self.portfolio_values[-10:]
        returns = np.diff(recent_values) / (np.array(recent_values[:-1]) + 1e-8)
        return np.std(returns)
    
    def _update_portfolio_value(self):
        """Update current portfolio value and tracking metrics."""
        current_prices = self._get_current_prices()
        portfolio_value = self.cash + np.sum(self.holdings * current_prices)
        self.portfolio_values.append(portfolio_value)
        
        # Update portfolio weights
        if portfolio_value > 0:
            asset_values = self.holdings * current_prices
            self.portfolio_weights = asset_values / portfolio_value
        else:
            self.portfolio_weights = np.zeros(self.n_assets)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Target portfolio weights for each stock
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Ensure action is valid (non-negative, sums to <= 1.0)
        action = np.clip(action, 0.0, 1.0)
        if np.sum(action) > 1.0:
            action = action / np.sum(action)  # Normalize to sum to 1.0
        
        # Store previous state for reward calculation
        prev_portfolio_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_cash
        
        # Execute portfolio rebalancing
        transaction_costs = self._rebalance_portfolio(action)
        self.total_transaction_costs += transaction_costs
        
        # Move to next time step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.data_length - 1
        
        # Update portfolio value with new prices
        self._update_portfolio_value()
        current_portfolio_value = self.portfolio_values[-1]
        
        # Calculate reward (risk-adjusted return)
        reward = self._calculate_reward(prev_portfolio_value, current_portfolio_value, transaction_costs)
        
        # Get new observation
        obs = self._get_observation()
        
        # Create info dictionary with useful metrics
        info = {
            'portfolio_value': current_portfolio_value,
            'transaction_costs': transaction_costs,
            'portfolio_weights': self.portfolio_weights.copy(),
            'cash_ratio': self.cash / (current_portfolio_value + 1e-8)
        }
        
        return obs, reward, done, False, info
    
    def _rebalance_portfolio(self, target_weights: np.ndarray) -> float:
        """
        Rebalance portfolio to target weights and return transaction costs.
        
        This method demonstrates how continuous actions get translated into
        discrete market operations (buying/selling shares).
        """
        current_prices = self._get_current_prices()
        current_portfolio_value = self.cash + np.sum(self.holdings * current_prices)
        
        # Calculate target dollar amounts for each stock
        target_values = target_weights * current_portfolio_value
        current_values = self.holdings * current_prices
        
        total_transaction_costs = 0.0
        
        # Execute trades for each stock
        for i, symbol in enumerate(self.symbols):
            target_value = target_values[i]
            current_value = current_values[i]
            price = current_prices[i]
            
            if abs(target_value - current_value) < 0.01:  # Minimum trade threshold
                continue
                
            if target_value > current_value:
                # Need to buy more shares
                trade_value = target_value - current_value
                if trade_value <= self.cash:
                    shares_to_buy = int(trade_value / price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * price
                        transaction_cost = cost * self.transaction_cost
                        self.cash -= (cost + transaction_cost)
                        self.holdings[i] += shares_to_buy
                        total_transaction_costs += transaction_cost
            else:
                # Need to sell shares
                shares_to_sell = int((current_value - target_value) / price)
                shares_to_sell = min(shares_to_sell, int(self.holdings[i]))
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * price
                    transaction_cost = proceeds * self.transaction_cost
                    self.cash += (proceeds - transaction_cost)
                    self.holdings[i] -= shares_to_sell
                    total_transaction_costs += transaction_cost
        
        return total_transaction_costs
    
    def _calculate_reward(self, prev_value: float, current_value: float, transaction_costs: float) -> float:
        """
        Calculate risk-adjusted reward.
        
        This reward function balances returns against risk and transaction costs,
        which is crucial for learning stable trading strategies.
        """
        # Basic return
        raw_return = (current_value - prev_value) / (prev_value + 1e-8)
        
        # Subtract transaction costs (as percentage of portfolio)
        cost_penalty = transaction_costs / (prev_value + 1e-8)
        
        # Risk penalty based on portfolio volatility
        volatility_penalty = self.risk_aversion * self._calculate_portfolio_volatility()
        
        # Combine components
        reward = raw_return - cost_penalty - volatility_penalty
        
        # Scale reward to make learning more stable
        return reward * 100  # Scale up for better gradient signals
    
    def render(self, mode='human'):
        """Display current environment state."""
        current_value = self.portfolio_values[-1] if self.portfolio_values else 0
        total_return = (current_value / self.initial_cash - 1.0) * 100
        
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: ${current_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Cash: ${self.cash:,.2f}")
        print(f"Portfolio Weights: {dict(zip(self.symbols, self.portfolio_weights))}")
        print(f"Total Transaction Costs: ${self.total_transaction_costs:,.2f}")
        print("-" * 50)

# Example usage function
def create_sample_portfolio_env():
    """
    Create a sample environment with synthetic data for demonstration.
    
    In practice, you would replace this with real market data.
    """
    # Generate synthetic stock data
    np.random.seed(42)
    n_days = 1000
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    data_dict = {}
    for symbol in symbols:
        # Create realistic price movements with trends and volatility
        returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
        prices = [100]  # Starting price
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=n_days+1),
            'Close': prices
        })
        data_dict[symbol] = df
    
    return PortfolioTradingEnv(data_dict)

if __name__ == "__main__":
    # Demonstrate the environment
    env = create_sample_portfolio_env()
    obs, info = env.reset()
    
    print("Environment created successfully!")
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Number of assets: {env.n_assets}")
    
    # Take a few random actions to test the environment
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        if done:
            break
