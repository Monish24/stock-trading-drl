"""
DRL Agent wrapper around Stable-Baselines3 algorithms.

Supports: PPO, A2C, SAC (and easy to add DDPG, TD3).

Usage:
    python -m agents.drl_agent  # Quick smoke test
"""
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    PPO_PARAMS, A2C_PARAMS, SAC_PARAMS,
    TOTAL_TIMESTEPS, MODEL_DIR,
)


ALGO_MAP = {
    "PPO": (PPO, PPO_PARAMS),
    "A2C": (A2C, A2C_PARAMS),
    "SAC": (SAC, SAC_PARAMS),
}


class ProgressCallback(BaseCallback):
    """Print progress every N steps."""

    def __init__(self, check_freq: int = 10_000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"  Step {self.n_calls:,} / {self.model._total_timesteps:,}")
        return True


class DRLAgent:
    """Unified interface for training and evaluating DRL trading agents."""

    def __init__(self, env, algorithm: str = "PPO"):
        algorithm = algorithm.upper()
        if algorithm not in ALGO_MAP:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {list(ALGO_MAP.keys())}")

        algo_class, params = ALGO_MAP[algorithm]
        self.algorithm = algorithm
        self.env = env
        self.model = algo_class("MlpPolicy", env, **params)

    def train(self, timesteps: int = None):
        """Train the agent."""
        timesteps = timesteps or TOTAL_TIMESTEPS
        print(f"\nTraining {self.algorithm} for {timesteps:,} timesteps...")
        self.model.learn(
            total_timesteps=timesteps,
            callback=ProgressCallback(check_freq=timesteps // 10),
        )
        print(f"  Training complete!")

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """Get action from trained model."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def evaluate(self, test_env, verbose: bool = True) -> dict:
        """Run the trained agent on a test environment."""
        obs, _ = test_env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = self.predict(obs)
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            done = done or truncated

        final_value = test_env.portfolio_values[-1]
        initial = test_env.initial_cash
        total_return = (final_value / initial - 1.0) * 100

        if verbose:
            print(f"\n{'='*50}")
            print(f"  {self.algorithm} — Out-of-Sample Results")
            print(f"  Final value:  ${final_value:,.2f}")
            print(f"  Return:       {total_return:+.2f}%")
            print(f"  Costs:        ${test_env.total_transaction_costs:,.2f}")
            print(f"  Total reward: {total_reward:.4f}")
            print(f"{'='*50}")

        return {
            "algorithm": self.algorithm,
            "final_value": final_value,
            "total_return": total_return,
            "total_reward": total_reward,
            "transaction_costs": test_env.total_transaction_costs,
            "portfolio_values": test_env.portfolio_values,
        }

    def save(self, name: str = None):
        """Save model to disk."""
        name = name or f"{self.algorithm.lower()}_trader"
        path = MODEL_DIR / name
        self.model.save(str(path))
        print(f"  Model saved to {path}")

    def load(self, name: str = None):
        """Load model from disk."""
        name = name or f"{self.algorithm.lower()}_trader"
        path = MODEL_DIR / name
        algo_class, _ = ALGO_MAP[self.algorithm]
        self.model = algo_class.load(str(path), env=self.env)
        print(f"  Model loaded from {path}")


if __name__ == "__main__":
    # Smoke test with a dummy environment
    print("DRL Agent module loaded successfully.")
    print(f"Available algorithms: {list(ALGO_MAP.keys())}")
