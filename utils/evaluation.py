"""
Evaluation metrics and visualization for trading strategies.

Computes: Sharpe ratio, max drawdown, cumulative returns, etc.
Plots: portfolio performance, comparison charts.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR, RESULTS_DIR


def compute_metrics(portfolio_values: list[float], risk_free_rate: float = None) -> dict:
    """Compute standard trading performance metrics."""
    rf = risk_free_rate if risk_free_rate is not None else RISK_FREE_RATE
    values = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]

    total_return = (values[-1] / values[0] - 1.0) * 100
    n_days = len(returns)
    annual_return = ((values[-1] / values[0]) ** (TRADING_DAYS_PER_YEAR / max(n_days, 1)) - 1.0) * 100
    annual_vol = np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    excess_returns = returns - rf / TRADING_DAYS_PER_YEAR
    sharpe = (np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)) * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Max drawdown
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    max_drawdown = np.max(drawdown) * 100

    # Win rate (days with positive return)
    win_rate = (returns > 0).mean() * 100

    return {
        "total_return_pct": round(total_return, 2),
        "annual_return_pct": round(annual_return, 2),
        "annual_volatility_pct": round(annual_vol, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_drawdown, 2),
        "win_rate_pct": round(win_rate, 2),
        "n_trading_days": n_days,
    }


def compute_buy_and_hold(close_prices: np.ndarray, initial_cash: float) -> list[float]:
    """Equal-weight buy-and-hold baseline across all assets."""
    # Invest equally in all assets at the start
    n_assets = close_prices.shape[1]
    investment_per_asset = initial_cash / n_assets
    shares = investment_per_asset / close_prices[0]

    values = []
    for t in range(len(close_prices)):
        total = np.sum(shares * close_prices[t])
        values.append(total)
    return values


def plot_comparison(results: dict[str, list[float]], dates=None, save_path: str = None):
    """Plot portfolio value curves for multiple strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Portfolio values
    ax = axes[0]
    for name, values in results.items():
        x = dates[:len(values)] if dates is not None else range(len(values))
        ax.plot(x, values, label=name, linewidth=1.5)
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Strategy Comparison — Portfolio Value Over Time")
    ax.legend()
    ax.grid(alpha=0.3)

    # Drawdowns
    ax = axes[1]
    for name, values in results.items():
        values = np.array(values)
        peak = np.maximum.accumulate(values)
        dd = (peak - values) / peak * 100
        x = dates[:len(dd)] if dates is not None else range(len(dd))
        ax.fill_between(x, dd, alpha=0.3, label=name)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def print_comparison_table(all_metrics: dict[str, dict]):
    """Print a formatted comparison table of strategy metrics."""
    df = pd.DataFrame(all_metrics).T
    df.index.name = "Strategy"
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)
    return df


def full_evaluation(
    results: dict[str, list[float]],
    dates=None,
    save_prefix: str = "comparison",
):
    """Run metrics + plotting for all strategies."""
    all_metrics = {}
    for name, values in results.items():
        all_metrics[name] = compute_metrics(values)

    table = print_comparison_table(all_metrics)

    plot_path = RESULTS_DIR / f"{save_prefix}_performance.png"
    plot_comparison(results, dates=dates, save_path=str(plot_path))

    table.to_csv(RESULTS_DIR / f"{save_prefix}_metrics.csv")
    return all_metrics
