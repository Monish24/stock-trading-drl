# Deep Reinforcement Learning for Stock Trading with Sentiment Analysis

A comprehensive system that combines **Deep Reinforcement Learning** (PPO, A2C, SAC) with **NLP-based news sentiment** (FinBERT) for automated US stock trading. Built as a Masters thesis project at Leiden University.

## Research Question

> Can integrating LLM-derived news sentiment into Deep Reinforcement Learning agents improve stock trading performance compared to price-only strategies?

## Architecture

```
News Headlines ──→ FinBERT Sentiment ──→ Sentiment Features ──┐
                                                               ├──→ DRL Agent ──→ Trading Actions
Stock Prices ──→ Technical Indicators ──→ Price Features ──────┘
```

## Project Structure

```
stock-trading-drl/
├── main.py                  # Full training + evaluation pipeline
├── configs/
│   └── config.py            # All hyperparameters & settings
├── data/
│   ├── fetch_data.py        # Yahoo Finance data + technical indicators
│   └── merge.py             # Merge price data with sentiment
├── sentiment/
│   └── pipeline.py          # FinBERT news sentiment pipeline
├── envs/
│   └── trading_env.py       # Gymnasium trading environment
├── agents/
│   └── drl_agent.py         # PPO / A2C / SAC agent wrapper
├── utils/
│   └── evaluation.py        # Metrics, plots, strategy comparison
├── notebooks/               # Colab notebooks for experiments
├── models/saved/            # Trained model checkpoints
├── results/                 # Performance plots & metrics
└── requirements.txt
```

## Quick Start

### 1. Install

```bash
git clone https://github.com/Monish24/stock-trading-drl.git
cd stock-trading-drl
pip install -r requirements.txt
```

### 2. Fetch Data

```bash
# Download stock data + compute technical indicators
python -m data.fetch_data

# Run sentiment pipeline (takes ~30-45 min)
python -m sentiment.pipeline --start 2020-01-01 --end 2025-04-01

# Merge price + sentiment data
python -m data.merge
```

### 3. Train & Evaluate

```bash
# Full pipeline: train PPO, A2C, SAC + compare vs buy-and-hold
python main.py

# Price-only (no sentiment) for comparison
python main.py --no-sentiment

# Train specific algorithms with more steps
python main.py --algorithms PPO SAC --timesteps 200000

# Use cached data (skip re-downloading)
python main.py --skip-data
```

## Key Components

### Trading Environment (`envs/trading_env.py`)
- Gymnasium-compatible multi-stock portfolio environment
- Continuous action space (portfolio weight allocation)
- Realistic transaction costs (0.1% per trade)
- Observation: rolling window of technical indicators + sentiment scores

### DRL Agents (`agents/drl_agent.py`)
- **PPO** (Proximal Policy Optimization) — best general performance
- **A2C** (Advantage Actor-Critic) — faster training
- **SAC** (Soft Actor-Critic) — better exploration

### Sentiment Pipeline (`sentiment/pipeline.py`)
- FinBERT for financial-domain sentiment scoring
- GNews for headline fetching
- Daily aggregation (mean, min, max sentiment + news volume)

### Evaluation (`utils/evaluation.py`)
- Sharpe ratio, max drawdown, annualized return/volatility
- Buy-and-hold baseline comparison
- Side-by-side strategy visualization

## Stock Universe

20 liquid US large-caps: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ, WMT, PG, XOM, BAC, DIS, NFLX, AMD, CRM, PYPL, INTC

## Data

- **Price data**: 2018–2025 daily OHLCV from Yahoo Finance
- **Sentiment**: 2020–2025 news headlines scored by FinBERT
- **Train period**: 2018–2024
- **Test period**: 2024–2025 (out-of-sample)

## Running on Google Colab

The project is designed to run on Colab with a T4 GPU. See `notebooks/` for step-by-step Colab notebooks.

## References

- [FinRL: Deep RL Library for Automated Stock Trading](https://github.com/AI4Finance-Foundation/FinRL)
- [FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models](https://arxiv.org/abs/1908.10063)
- [Deep RL for Automated Stock Trading: An Ensemble Strategy](https://arxiv.org/abs/2511.12120)
- [Sentiment Trading with Large Language Models](https://arxiv.org/abs/2412.19245)

## License

MIT
