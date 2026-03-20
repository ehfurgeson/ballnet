"""
config.py — Single source of truth for the clean/ inference + backtesting engine.

To switch to a new model version, change ACTIVE_MODEL here.
All other scripts (live.py, backtest.py, update.py, predictor.py) import this file.
"""
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Model selection — change this one line to switch models everywhere
# ─────────────────────────────────────────────────────────────────
ACTIVE_MODEL = "v5"  # sub-directory name under models/

# ─────────────────────────────────────────────────────────────────
# Paths — all resolved relative to this file so the folder is portable
# ─────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent
DATA_DIR     = ROOT / "data"
MODEL_DIR    = ROOT / "models" / ACTIVE_MODEL
LOGS_DIR     = ROOT / "logs"
PRED_LOG_DIR = LOGS_DIR / "predictions"
TRADE_LOG_DIR= LOGS_DIR / "trades"
BACKTEST_DIR = LOGS_DIR / "backtest"
KALSHI_HIST  = LOGS_DIR / "kalshi_history"

# Path to GATv2TCN source code (custom module, not pip-installable)
GATV2_SRC    = ROOT / "architecture"

# ─────────────────────────────────────────────────────────────────
# Model architecture constants — must match training exactly
# ─────────────────────────────────────────────────────────────────
SEQ_LENGTH      = 10
FEATURE_COLS    = [
    "PTS", "AST", "REB", "TO", "STL", "BLK", "PLUS_MINUS",
    "TCHS", "PASS", "DIST", "PACE", "USG_PCT", "TS_PCT",
]
PREDICTION_COLS = ["PTS", "AST", "REB", "TO", "STL", "BLK"]
PRED_INDICES    = [FEATURE_COLS.index(c) for c in PREDICTION_COLS]
MIN_MINUTES     = 10.0

QUANTILE_REGRESSION = False
QUANTILES = [0.10, 0.50, 0.90]

# ─────────────────────────────────────────────────────────────────
# Data Splitting — 90% train, 5% val, 5% test
# ─────────────────────────────────────────────────────────────────
SPLIT_TRAIN     = 0.90          # train: 0% - 90%
SPLIT_VAL       = 0.95          # val: 90% - 95%, test: 95% - 100%

# ─────────────────────────────────────────────────────────────────
# Live trading hyperparameters (live.py defaults)
# ─────────────────────────────────────────────────────────────────
MIN_EV           = 2.0    # cents — minimum EV per contract to consider betting
MIN_EDGE         = 0.03   # minimum probability edge over implied probability
KELLY_FRACTION   = 0.25   # fractional Kelly multiplier (0.25 = quarter-Kelly)
MAX_BET_SIZE     = 50     # dollars — max risk per single position
MAX_GAME_PCT     = 0.05   # max fraction of bankroll allocated per game
POLL_INTERVAL    = 120    # seconds between Kalshi market polls
ALERT_COOLDOWN   = 300    # seconds before re-alerting the same prop

# ─────────────────────────────────────────────────────────────────
# Kalshi configuration
# ─────────────────────────────────────────────────────────────────
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES = {
    "KXNBAPTS": "PTS",
    "KXNBAAST": "AST",
    "KXNBAREB": "REB",
    "KXNBABLK": "BLK",
    "KXNBASTL": "STL",
    "KXNBATO":  "TO",
}

# ─────────────────────────────────────────────────────────────────
# Backtesting defaults
# ─────────────────────────────────────────────────────────────────
BACKTEST_INIT_BANKROLL   = 10_000.0
BACKTEST_KELLY_FRAC      = 0.25
BACKTEST_MIN_EV          = 0.2
BACKTEST_MAX_ASK         = 99.0
BACKTEST_MAX_GAME_PCT    = 0.125
BACKTEST_MAX_BETS_OVER   = 1
BACKTEST_MAX_BETS_UNDER  = 1

# NBA annualization constant for Sharpe (170 game-days/season, NOT 252)
NBA_ANNUAL_DAYS = 170

# Per-stat empirical RMSE (used by naive Gaussian baseline in backtest)
STAT_RMSE = {
    "PTS": 5.5, "AST": 1.8, "REB": 2.2,
    "STL": 0.8, "BLK": 0.7, "TO":  1.4,
}
