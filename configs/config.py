"""
Central configuration for the Stock Trading DRL project.
All hyperparameters, tickers, and paths in one place.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SENTIMENT_DATA_DIR = DATA_DIR / "sentiment"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
RESULTS_DIR = PROJECT_ROOT / "results"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SENTIMENT_DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Stock Universe ────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "XOM", "BAC", "DIS",
    "NFLX", "AMD", "CRM", "PYPL", "INTC",
]

TICKER_TO_COMPANY = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google Alphabet",
    "AMZN": "Amazon", "NVDA": "Nvidia", "META": "Meta Facebook",
    "TSLA": "Tesla", "JPM": "JPMorgan", "V": "Visa",
    "JNJ": "Johnson Johnson", "WMT": "Walmart", "PG": "Procter Gamble",
    "XOM": "ExxonMobil", "BAC": "Bank of America", "DIS": "Disney",
    "NFLX": "Netflix", "AMD": "AMD", "CRM": "Salesforce",
    "PYPL": "PayPal", "INTC": "Intel",
}

# ── Date Ranges ───────────────────────────────────────────────────────────────
START_DATE = "2018-01-01"
END_DATE = "2025-04-01"
TRAIN_END_DATE = "2024-04-01"  # Everything after = test set

# ── Technical Indicator Parameters ────────────────────────────────────────────
SMA_WINDOWS = [5, 20, 60]
EMA_WINDOW = 12
RSI_WINDOW = 14
BB_WINDOW = 20
VOLATILITY_WINDOW = 20

# ── Sentiment ─────────────────────────────────────────────────────────────────
SENTIMENT_MODEL = "ProsusAI/finbert"
NEWS_MAX_ARTICLES = 5
NEWS_RATE_LIMIT = 0.5  # seconds between GNews calls

# ── Trading Environment ──────────────────────────────────────────────────────
INITIAL_CASH = 10_000
TRANSACTION_COST = 0.001  # 0.1% per trade
WINDOW_SIZE = 20

# ── DRL Training ──────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 100_000

PPO_PARAMS = dict(
    learning_rate=3e-4, n_steps=2048, batch_size=64,
    n_epochs=10, gamma=0.99, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.01, verbose=1,
)

A2C_PARAMS = dict(
    learning_rate=7e-4, n_steps=5, gamma=0.99,
    gae_lambda=1.0, ent_coef=0.01, verbose=1,
)

SAC_PARAMS = dict(
    learning_rate=3e-4, buffer_size=100_000,
    batch_size=256, gamma=0.99, tau=0.005,
    ent_coef="auto", verbose=1,
)

# ── Evaluation ────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.04
TRADING_DAYS_PER_YEAR = 252
