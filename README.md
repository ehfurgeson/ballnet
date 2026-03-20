# knowball/clean — NBA Player Prop Inference & Backtesting Engine

> **For AI Agents**: Read this entire file before editing any script in this folder.
> All paths, data flows, known gotchas, and design decisions are documented here.

---

## Overview

`clean/` is a self-contained, copy-pasteable NBA player prop prediction engine built
around a trained **GATv2TCN** (Graph Attention Network + Temporal Convolutional Network)
model. It provides:

1. **Inference** — point estimates and calibrated over/under probabilities for any player-stat-threshold triple
2. **Live trading** — automated EV-based order placement on Kalshi prediction markets
3. **Backtesting** — historical simulation of EV strategies with configurable Kelly sizing

The folder is designed to be fully portable. All paths resolve relative to `clean/` using
`Path(__file__).resolve()`, so you can copy it anywhere and it will work.

**To switch to a new model**, change one line in `config.py`:
```python
ACTIVE_MODEL = "v5"   # was "v3"
```
Both `live.py`, `backtest.py`, and `test.py` import `config.py` and will automatically use weights
from `models/v5/`.

---

## Directory Structure

```
clean/
├── README.md                  # This file
├── game_embeddings.md         # ← Game outcome prediction reference (read this for embedding work)
├── config.py                  # ← Single source of truth for all paths + hyperparams
├── predictor.py               # Core predictor class (shared by scripts)
├── update.py                  # ← Daily data refresh (run once before live.py)
├── live.py                    # Live trading on Kalshi (EV filter + Kelly sizing)
├── backtest.py                # Historical backtesting engine (player props)
├── test.py                    # ← Model inference quality test (RMSE/MAE/CORR)
├── kalshi_manual.py           # ← Manual trading CLI for Kalshi
├── analysis.qmd               # ← Quarto analysis for backtest results and predictions
│
├── architecture/              # Model architecture source
│   ├── gatv2tcn.py            # GATv2TCN implementation
│   └── tcn.py                 # TCN block implementation
│
├── data/                      # Runtime data (gitignored: *.pkl, *.npy, *.parquet)
│   ├── raw_boxscores.parquet          # Full NBA game log (built by 01_fetch_data.py)
│   ├── kalshi_pregame_prices.parquet  # ← Pre-game bid/ask prices for player props (05_collect_kalshi_history.py)
│   ├── kalshi_game_markets.parquet    # ← Pre-game prices for game-level markets (06_collect_kalshi_game_markets.py)
│   ├── game_embeddings.parquet        # ← Per-game team embeddings from backbone splice (07_extract_game_embeddings.py)
│   ├── game_home_teams.parquet        # ← {GAME_ID: home_team_abbr} from LeagueGameFinder (cached)
│   ├── X_seq.pkl              # (Days, Players, 13) forward-filled stat tensor
│   ├── X_raw.pkl              # (Days, Players, 13) raw sparse stat tensor (no fill)
│   ├── G_seq.pkl              # List of networkx graphs, one per game-day
│   ├── player_ids.pkl         # Ordered list of player IDs (axis 1 of X_seq)
│   ├── game_dates.pkl         # Ordered list of date strings (axis 0 of X_seq)
│   ├── day_seasons.pkl        # Season label per day (e.g. "2024-25")
│   ├── team_temporal.pkl      # (Days, Players, n_teams) per-day team one-hot arrays
│   ├── pos_temporal.pkl       # (Days, Players, 3) per-day position arrays
│   ├── n_teams.pkl            # int — number of unique teams
│   ├── player_id2team.pkl     # {player_id: "LAL"} — most recent team abbreviation
│   ├── player_id2position.pkl # {player_id: [G,F,C] binary array}
│   ├── mu_per_day.npy         # Causal sliding-window normalization means (Days, 1, 13)
│   └── sd_per_day.npy         # Causal sliding-window normalization std devs (Days, 1, 13)
│
├── models/                    # Trained model weights
│   ├── v5/                    # Current active model
│   │   ├── model.pth          # GATv2TCN state dict
│   │   ├── team_emb.pth       # Linear(n_teams, 2)
│   │   ├── pos_emb.pth        # Linear(3, 2)
│   │   └── conformal_residuals.pkl  # Calibration residuals
│   └── ...                    # v1-v4
│
├── scripts/                   # Setup and training scripts
│   ├── 01_fetch_data.py             # Historical NBA boxscore scrape
│   ├── 02_build_tensors.py          # Build tensors from raw data
│   ├── 03_train.py                  # Training script (MPS/CUDA/Colab)
│   ├── 04_calibrate.py              # Compute conformal_residuals.pkl
│   ├── 05_collect_kalshi_history.py # Fetch pre-game bid/ask prices
│   ├── 06_collect_kalshi_game_markets.py  # Collect game-level market prices
│   ├── 07_extract_game_embeddings.py      # Extract backbone embeddings
│   └── prepare_colab.py             # Package for Colab training
│
├── upload/                    # Google Colab upload bundle
│   ├── train.ipynb            # Colab bootstrap notebook
│   ├── config.py              # Colab path shim
│   ├── scripts/03_train.py    # Training script copy
│   ├── gatv2tcn.py            # Model source
│   ├── tcn.py                 # TCN source
│   └── data/                  # Required data files
│
└── logs/                      # Output logs (gitignored)
    ├── predictions/           # predictions_YYYY-MM-DD.csv
    ├── trades/                # trades_YYYY-MM-DD.csv
    └── backtest/              # backtest_summaries.parquet, backtest_bets.parquet
```

---

## Typical Workflow

### First-time setup
```bash
# 1. Fetch all historical NBA data (takes hours, uses kamikaze restart protocol)
python scripts/01_fetch_data.py

# 2. Build all tensor artifacts from raw data
python scripts/02_build_tensors.py

# 3a. Train on Google Colab (recommended — ~15-30 min on T4/A100)
python scripts/prepare_colab.py
# → Upload clean/upload/ to Google Drive root
# → Open upload/train.ipynb in Colab → Runtime → Run all
# → Download clean_download/ → copy .pth files to clean/models/v2/

# 3b. OR train locally on MPS/CUDA (expect 1-2+ hours)
python scripts/03_train.py

# 4. Calibrate the model (computes conformal residuals)
python scripts/04_calibrate.py

# 5. Collect historical pre-game Kalshi prices for backtesting
#    Public API, no auth required. ~35 min for a full season of data.
python scripts/05_collect_kalshi_history.py --start 2026-01-16 --end 2026-02-22
```

### Daily workflow
```bash
# Fetch any new games and update all tensors
python update.py

# Run live trading (dry-run by default — set LIVE=True in live.py to place real bets)
python live.py

# (Optional) Run manual trading CLI for specific tickers
python kalshi_manual.py

# Incrementally collect yesterday's prices (appends new rows, skips existing tickers)
python scripts/05_collect_kalshi_history.py  # defaults to last 30 days
```

### Running the backtest
```bash
# Full sweep — reads data/kalshi_pregame_prices.parquet
python backtest.py --strategy conformal --configs 100 --top 10

# Use cached opportunities (skip re-extraction, saves ~2 min)
python backtest.py --use-cache --configs 100

# Naive Gaussian baseline (no conformal residuals)
python backtest.py --strategy naive

# After collecting new prices, clear the cache first:
rm data/ops_*_cache.parquet data/quantile_test_cache.parquet
```

### Strategy Validation & Quality Check
```bash
# Run model inference quality test (reports RMSE/MAE/CORR vs targets)
python test.py

# Analyze backtest results and strategy parameters
quarto render analysis.qmd
```

### Retraining (updated data)
```bash
python scripts/02_build_tensors.py   # regenerate tensors
python scripts/prepare_colab.py      # rebuild upload/ bundle with fresh data
# ... train on Colab, copy weights ...
python scripts/04_calibrate.py       # recompute residuals for new weights

# After retraining: re-extract game embeddings with fresh weights
python scripts/07_extract_game_embeddings.py --cutoff YYYY-MM-DD
```

### Game outcome evaluation
```bash
# Collect Kalshi game-level market prices (moneyline, spread, total)
python scripts/06_collect_kalshi_game_markets.py --start 2026-01-16 --end 2026-03-05

# Extract backbone embeddings + train game-outcome linear/MLP head
python scripts/07_extract_game_embeddings.py --cutoff 2026-03-05
# Re-run with cached embeddings after model retrain:
python scripts/07_extract_game_embeddings.py --skip-extract
```

> See `game_embeddings.md` for full results, architecture details, and future directions.

---

## config.py

**The single import every other script depends on.**

```python
ACTIVE_MODEL    = "v5"          # ← Change this to switch models everywhere

ROOT            = Path(__file__).resolve().parent
DATA_DIR        = ROOT / "data"
MODEL_DIR       = ROOT / "models" / ACTIVE_MODEL
GATV2_SRC       = ROOT / "architecture"

FEATURE_COLS    = ['PTS','AST','REB','TO','STL','BLK','PLUS_MINUS',
                   'TCHS','PASS','DIST','PACE','USG_PCT','TS_PCT']  # 13 features
PREDICTION_COLS = ['PTS','AST','REB','TO','STL','BLK']             # 6 output stats
PRED_INDICES    = [0,1,2,3,4,5]   # indices of PREDICTION_COLS within FEATURE_COLS
SEQ_LENGTH      = 10              # days of history used as input window
```

---

## Data Files

### `X_seq.pkl` — shape `(Days, Players, 13)`
The **forward-filled** stat tensor. Zero rows (non-playing days) are filled forward
with each player's last known stats. This is the version used as model input.

> **Gotcha**: `X_seq.pkl` is stored **un-normalized** (raw stat values like PTS=24.0).
> Normalization happens **in memory only** inside `predictor.py` via `mu_per_day`/
> `sd_per_day`. **Never write the normalized version back to disk** — it would cause
> double-normalization on the next load.

### `X_raw.pkl` — shape `(Days, Players, 13)`
Same as `X_seq` but **without forward-fill** (~84% zeros). Used to detect which
players actually played on a given day (non-zero rows).

### `player_id2team.pkl` — `{player_id: "LAL"}`
Maps player ID → team abbreviation string (e.g. `"LAL"`, `"BOS"`).
Generated by `02_build_tensors.py` from the most recent team per player.

> **Gotcha**: This stores **strings** not integers. `03_train.py` and `04_calibrate.py`
> both apply an alphabetical string→int encoding (`sorted(all_teams)`) at runtime to
> get a consistent `n_teams=30` integer mapping.

### `player_id2position.pkl` — `{player_id: [G, F, C]}`
Maps player ID → 3-element binary position vector (e.g. `[1, 0, 0]` for Forward).
Generated by `02_build_tensors.py` using `nba_api` static player data.

### `mu_per_day.npy` / `sd_per_day.npy` — shape `(Days, 1, 13)`
Causal sliding window normalization statistics. To prevent lookahead bias in backtesting, each day leverages an expanding trailing window of up to 150 active days to compute rolling means and standard deviations using purely historical data. `sd` values < 1e-6 are treated as 1.0.

---

## Model Architecture — GATv2TCN

```python
GATv2TCN(
    in_channels        = 17,   # 13 stats + 2 team_emb + 2 pos_emb
    out_channels       = 6,    # PTS, AST, REB, TO, STL, BLK
    len_input          = 10,   # SEQ_LENGTH
    len_output         = 1,
    temporal_filter    = 64,
    out_gatv2conv      = 32,
    dropout_tcn        = 0.25,
    dropout_gatv2conv  = 0.5,
    head_gatv2conv     = 4,
)
```

> **Gotcha**: The correct kwarg names are `len_input`, `len_output`, `out_gatv2conv`,
> `dropout_tcn`, `dropout_gatv2conv`, `head_gatv2conv`. Do NOT use `seq_length` or
> `heads` — those don't exist in the `gatv2tcn.py` constructor and will raise
> `TypeError: unexpected keyword argument`.

**Embedding layers:**
```python
team_emb = nn.Linear(n_teams, 2)   # bias=True (default)
pos_emb  = nn.Linear(3, 2)         # bias=True (default)
```
> **Gotcha**: Always use default `bias=True` when creating these layers to load the
> saved `.pth` files, which include a bias key. Using `bias=False` causes
> `RuntimeError: Unexpected key(s) in state_dict: "bias"`.

**Input construction:**
```python
x_t = cat([X_norm[day, :, :], team_emb(team_one_hot), pos_emb(pos_vec)], dim=1)
# shape: (P=805, 17)
# stacked over SEQ_LENGTH=10 days → (1, P, 17, 10)
```

---

## Training Notes

### Why train on Google Colab?
The model is small (~77K parameters, 76KB `.pth`). However, the full training loop
(300 epochs × 20-day batch × 148 val days) takes **1.5+ hours on Apple MPS** but
only **15-30 minutes on Colab T4/A100**.

### Colab training workflow (`prepare_colab.py`)
```bash
python scripts/prepare_colab.py
```
Builds `upload/` (~44 MB):
- `scripts/03_train.py` — **exact copy** of the canonical training script (parity guarantee)
- `config.py` — auto-generated Colab path shim so `03_train.py` resolves imports correctly
- `train.ipynb` — minimal 4-cell bootstrap notebook that runs `03_train.py` via subprocess
- `gatv2tcn.py` + `tcn.py` — model source
- `data/` — the 5 required pkl files

> **Parity guarantee**: `prepare_colab.py` copies `scripts/03_train.py` directly into the
> upload bundle rather than duplicating its logic. This means any changes to `03_train.py`
> (hyperparameters, normalization, loss function, etc.) are automatically reflected in Colab
> training after re-running `prepare_colab.py`. **Never edit training logic in the notebook
> or in `prepare_colab.py` directly — always edit `03_train.py`.**

After training completes, Colab saves output to `clean_download/` in your Drive:
- `model.pth`, `team_emb.pth`, `pos_emb.pth` → copy to `clean/models/<ACTIVE_MODEL>/`
- Re-run `04_calibrate.py` after copying new weights

> **Gotcha**: Run `prepare_colab.py` fresh each time you retrain with updated data
> — it copies the current pkl files and `03_train.py`, so stale uploads will train
> on stale data with stale code.

### Understanding the loss numbers
Training uses **summed** MSE (not averaged) over all days in the batch/val set:
```python
loss = sum(mse_per_day)   # NOT mean(mse_per_day)
```
This means raw loss values scale linearly with the number of days. Our dataset has
~7× more val days (147) than the original Colab notebook (20), so our val loss will
be ~7× larger by construction. **This is expected and correct.** The per-day loss
converges to the same ~0.032 as the original training. Divide the reported val loss
by ~147 to compare.

### tqdm display during training
The progress bar shows all four quantities every epoch:
```
Training:  35% | 105/300 [train=38.4, val=21.3, best=18.9, saved=★]
```
`★` appears when a new best validation loss is saved.

---

## predictor.py — GATv2Predictor

The core class loaded by both `live.py` and `backtest.py`.

### Setup
```python
p = GATv2Predictor()
p.setup()   # loads all artifacts from data/ and models/<ACTIVE_MODEL>/
```
`setup()` loads: `X_seq`, `G_seq`, `player_ids`, `game_dates`, `mu_per_day`,
`sd_per_day`, `team_temporal`, `pos_temporal`, `n_teams`, `conformal_residuals.pkl`,
and all three `.pth` weight files.

**Conformal residuals (tiered format):**
After loading `conformal_residuals.pkl`, the predictor exposes:
- `self.val_residuals` — `dict[str, list]` keyed as `"PTS_low"`, `"PTS_mid"`, `"PTS_high"`, etc.
- `self.val_bias` — `dict[str, float]` — per-stat mean bias computed at calibration time

**Key public helpers:**
```python
p.get_residual_std("PTS")   # mid-tier std — used by quantile_test.py and live.py SD filter
```

### Inference methods

#### Fast path — use in `backtest.py`
```python
# ONE forward pass for all 805 players, cached per day
pred_matrix = p.predict_all_for_day(day_idx)     # → (P, 6) raw stat units
mc_matrix   = p.predict_all_mc_for_day(day_idx)  # → (20, P, 6) MC-dropout samples
```
Both methods are **memoized** by `day_idx`. Calling them twice for the same day is
a free dict lookup. This is ~150× faster than the per-player approach for backtest.

Use `_get_day_idx_for_date("2025-02-15")` to convert a date string to an index.

#### Convenience wrappers — use in `live.py`
```python
p.predict_point_estimate(player_id, "PTS")
p.predict_conformal_probability(player_id, "PTS", 22.5)
```
These call the day-level batched methods internally so they also benefit from caching
if called multiple times for the same day.

#### Memory management
```python
p.clear_day_cache()   # frees _day_cache and _mc_cache dicts if RAM is tight
```

---

## backtest.py

### Data source
`backtest.py` reads **`data/kalshi_pregame_prices.parquet`** — built by
`scripts/05_collect_kalshi_history.py`. Each row contains a real Kalshi bid/ask
price from the last 1-minute candlestick before tip-off, alongside player, stat,
threshold, and outcome columns.

> **Important**: Before running `backtest.py`, make sure
> `data/kalshi_pregame_prices.parquet` exists. If it doesn't:
> ```bash
> python scripts/05_collect_kalshi_history.py --start YYYY-MM-DD --end YYYY-MM-DD
> ```

The old approach (reading per-day JSON files, using `previous_yes_ask` as a proxy)
has been replaced. `previous_yes_ask` was the price at the last API poll before
permanent data collection — not the actual pre-game price. The new approach fetches
candlestick history for each ticker and records the price of the closest 1-minute
candle before tip-off.

### Date Alignment and Inference
`extract_opportunities()` groups all props by `game_date` (from the parquet). To
prevent target leakage, the model is evaluated on `game_date - 1 day`.

At the start of each inference date context:
1. **1 eval forward pass** → `(805, 6)` prediction matrix (memoized)
2. **20 MC-dropout passes** → `(20, 805, 6)` sample matrix (memoized)
3. All props on that day → **numpy array lookups**, no model calls

Old approach: `150 props × 21 passes = 3,150` forward passes per day.
New approach: **21 forward passes** per day, regardless of number of props (~150× faster).

### Strategy options
```bash
python backtest.py                        # conformal (MC-dropout + residuals)
python backtest.py --strategy naive       # Gaussian approximation using STAT_RMSE
python backtest.py --configs 100          # smaller config sweep (default: 5000)
python backtest.py --top 10              # print top N configs by Sharpe
python backtest.py --use-cache           # skip extraction, use cached ops parquet
```

### Conformal probability details
Per prop under `conformal` strategy:
```python
# 1. Bias-correct the MC samples
samples = mc_matrix[:, pidx, si] + predictor.val_bias[stat]
# 2. Select tier based on corrected mean estimate (low/mid/high)
residuals = predictor._get_residuals_for(stat, mean(samples))
# 3. Build 2000-sample distribution
res  = np.random.choice(residuals, (20, 100))
dist = (samples[:, None] + res).flatten()
p_over = mean(dist > threshold)
```
Residuals come from `04_calibrate.py` — signed errors on the validation set,
mean-centered per stat and stratified by predicted value magnitude.

### EV calculation
```python
ev_yes = (p_over * (100 - yes_ask) - p_under * yes_ask) / 100
ev_no  = (p_under * (100 - no_ask) - p_over  * no_ask)  / 100
```
Values in **[-1, 1]** range (EV per dollar wagered). The `/100` divisor is critical.

---

## scripts/05_collect_kalshi_history.py

Fetches **real pre-game Kalshi bid/ask prices** for all NBA player props in a date
range and saves them to `data/kalshi_pregame_prices.parquet`.

### How it works
1. **NBA tip-off times** — `nba_api.ScoreboardV3` returns `gameTimeUTC` directly
   as an ISO UTC timestamp per game. (Uses V3, not V2 which is deprecated for 2025-26.)
2. **Market discovery** — paginates `GET /markets?series_ticker=KXNBAPTS&min_close_ts=...`
   (public, no auth) to list all settled props in the date range.
3. **Candlestick prices** — for each ticker, calls
   `GET /series/{SERIES}/markets/{ticker}/candlesticks?period_interval=1` to get
   1-minute price history. Finds the **last candle at or before tip-off** and records
   its close price as the pre-game bid/ask.
4. **Saves** to `data/kalshi_pregame_prices.parquet` in **append mode** — re-running
   only adds new tickers, never duplicates existing ones.

### Output columns
| Column | Description |
|--------|-------------|
| `ticker` | Kalshi market ticker |
| `event_ticker` | Game identifier |
| `game_date` | YYYY-MM-DD |
| `player` | Player name from market title |
| `stat` | PTS / AST / REB / BLK / STL |
| `threshold` | The prop line (float) |
| `result` | `yes` or `no` |
| `yes_ask` | Pre-game YES ask in cents (1–99) |
| `yes_bid` | Pre-game YES bid in cents |
| `no_ask` | 100 − yes_bid |
| `no_bid` | 100 − yes_ask |
| `price_ts` | Unix timestamp of the candle used |
| `game_start_ts` | Actual tip-off Unix timestamp from nba_api |
| `price_age_s` | Seconds between candle and tip-off |
| `spread` | yes_ask − yes_bid (market tightness) |

### Usage
```bash
# First run — collect a historical range
python scripts/05_collect_kalshi_history.py --start 2026-01-16 --end 2026-02-22

# Incremental update (defaults to last 30 days, appends new rows)
python scripts/05_collect_kalshi_history.py

# Preview without writing
python scripts/05_collect_kalshi_history.py --start 2026-01-16 --end 2026-01-17 --dry-run
```

> After running, delete `data/ops_*_cache.parquet` before re-running `backtest.py`.

---

## scripts/02_build_tensors.py

Reads `data/raw_boxscores.parquet` and produces all artifacts in `data/`.

### What it generates
| File | Description |
|------|-------------|
| `X_seq.pkl` | Forward-filled stat tensor `(Days, Players, 13)` |
| `X_raw.pkl` | Raw sparse stat tensor (no fill) |
| `G_seq.pkl` | List of networkx graphs |
| `player_ids.pkl` | Ordered player ID list |
| `game_dates.pkl` | Ordered date string list |
| `player_id2team.pkl` | `{pid: "LAL"}` — most recent team |
| `player_id2position.pkl` | `{pid: [G,F,C]}` — position binary vector |
| `mu_per_day.npy` / `sd_per_day.npy` | Per-season normalization stats |

> **Gotcha**: `player_id2team` stores **team abbreviation strings** (`"LAL"`)
> not integers. `03_train.py` and `04_calibrate.py` handle this with an
> alphabetical sort encoding at runtime.

### Pre-flight tensor comparison
Before rebuilding, the script compares the current tensor's player count to the
incoming data's player count. If counts differ significantly, **retraining is
required** — the model's graph structure is keyed to specific player indices.

---

## scripts/03_train.py

### Team/position encoding
```python
# player_id2team.pkl → string→int encoding
all_teams    = sorted(set(team_str_values))  # alphabetical, stable
team_str2int = {t: i for i, t in enumerate(all_teams)}
```
This handles both string abbreviations (our pipeline) and integer team IDs
(original Colab pipeline) automatically.

### Mask (active players only)
Loss is computed **only on players who appear in the target day's graph** (i.e.,
players who actually played that game-day):
```python
mask = G_out[i].unique()   # node indices in next day's edge tensor
loss = mse_loss(pred[mask], y[mask])
```
Players who didn't play are forward-filled in `y` but excluded from the loss.
This prevents the model from wasting capacity learning the fill-forward function.

---

## scripts/04_calibrate.py

Loads model weights and runs inference on the **validation set** (days 50%–75%
of the dataset, matching `03_train.py`) to compute signed residuals against the
forward-shifted target day ($t+1$):
```
residual = actual[t+1] - predicted[t]
```

Residuals are **stratified by predicted value magnitude** into three tiers per stat
(low / mid / high) and **mean-centered** to remove systematic model bias before saving.

**Output format** (`conformal_residuals.pkl`):
```python
{
    "bias": {"PTS": -0.247, "AST": -0.326, ...},   # raw mean residual per stat
    "residuals": {
        "PTS_low": [...],    # mean-centered, for predictions < 12
        "PTS_mid": [...],    # mean-centered, for predictions 12–22
        "PTS_high": [...],   # mean-centered, for predictions ≥ 22
        "AST_low": [...],
        ...                  # all 6 stats × up to 3 tiers
    }
}
```

**Tier boundaries:**
| Stat | Low | Mid | High |
|------|-----|-----|------|
| PTS | <12 | 12–22 | ≥22 |
| AST | <4  | 4–8  | ≥8  |
| REB | <4  | 4–8  | ≥8  |
| STL | <1.5| 1.5–3| ≥3  |
| BLK | <1  | 1–2.5| ≥2.5|
| TO  | <2  | 2–4  | ≥4  |

Tiers with fewer than 30 samples fall back to the mid tier (e.g. STL_high and TO_high
typically have 0 samples — the model rarely predicts these stats that high).

> **Always re-run `04_calibrate.py` after copying new weights from Colab.**

> **Also re-run if tier boundaries are adjusted** — the saved residuals depend on which
> threshold was used to bin predictions at calibration time.

---

## Known Gotchas

| # | Issue | Details |
|---|-------|---------|
| 1 | Double normalization | `X_seq.pkl` is raw. Normalize in-memory only. Never write `X_seq_norm` to disk. |
| 2 | Wrong GATv2TCN kwargs | Use `len_input`, `len_output`, `out_gatv2conv`, `dropout_tcn`, `dropout_gatv2conv`, `head_gatv2conv`. Never `seq_length` or `heads`. |
| 3 | bias=True on embeddings | `nn.Linear(n_teams, 2)` default is `bias=True`. Saved `.pth` files include bias. Never use `bias=False`. |
| 4 | team strings not ints | `player_id2team.pkl` stores `"LAL"` strings. Use alphabetical sort encoding before computing `n_teams`. |
| 5 | Loss scale vs Colab | Our val loss is ~7× larger by construction (147 val days vs 20). Compare per-day loss (divide by ~147). |
| 6 | CWD sensitivity | Always run scripts from `clean/` or via absolute path. `config.py` imports fail if `clean/` is not importable. |
| 7 | Upload/ freshness | Re-run `prepare_colab.py` every time you want to retrain with updated data. It copies fresh pkl files AND `03_train.py`. |
| 8 | Colab/local parity | **Never duplicate training logic** in `prepare_colab.py` or the notebook. All training code lives in `03_train.py`. Edit `03_train.py` → re-run `prepare_colab.py` → upload. |
| 9 | Double-denormalization | The model natively outputs raw stat predictions. Never multiply by `sd_per_day` or add `mu` in `predictor.py` or `04_calibrate.py`. That inflates predictions (24.5 PTS → 260 PTS). |
| 10 | EV formula scale | `calc_ev` **must** divide by 100. Without it, EV values are 100× too large (cents not fraction) and no bets will pass any realistic `min_ev` threshold. |
| 11 | Kalshi candlestick endpoint | For markets settled after 2025-03-01 (the live partition), the correct endpoint is `GET /series/{SERIES}/markets/{ticker}/candlesticks`. Do NOT use `/historical/markets/{ticker}/candlesticks` — it returns 404 for recent markets. |
| 12 | Kalshi price units | The live `/series/…/candlesticks` endpoint returns prices as **integer cents (0–100)**. The historical endpoint returns dollar strings (`"0.5600"`). They are different formats. |
| 13 | nba_api ScoreboardV3 headers | `ScoreboardV3` uses **camelCase** headers: `gameCode`, `gameTimeUTC`. Do not use `ScoreboardV2` names (`GAMECODE`, `GAME_STATUS_TEXT`) — they will raise `ValueError`. Use `ScoreboardV3`; V2 is deprecated for 2025-26 and hangs on SSL for some game dates. |
| 14 | Kalshi close_ts vs game date | NBA games played on date X have markets that close on date X+1. When filtering `/markets?min_close_ts=...&max_close_ts=...`, extend end by **+2 days** to catch all games in the target range. |
| 15 | Backtest cache staleness | After collecting new prices via `05_collect_kalshi_history.py`, delete `data/ops_*_cache.parquet` before re-running `backtest.py`. The cache is keyed by strategy name, not data freshness. |
| 16 | conformal_residuals.pkl format | The file now uses a **tiered format**. Use `predictor._get_residuals_for(stat, pred_val)` or `predictor.get_residual_std(stat)` instead. After re-calibrating, always delete backtest caches. |
| 17 | BackboneWithEmbedding sync | `BackboneWithEmbedding` in `07_extract_game_embeddings.py` replicates `forward()` step-by-step. If `architecture/gatv2tcn.py` is ever modified, the wrapper must be updated in sync. |
| 18 | Game embedding stale after retrain | `data/game_embeddings.parquet` is tied to model weights. Always re-run `python scripts/07_extract_game_embeddings.py` (without `--skip-extract`) after copying new `.pth` files. |
| 19 | LOG_TRANSFORM semantics | The model natively outputs raw stat predictions. `predictor.py` and `04_calibrate.py` must have `LOG_TRANSFORM` set correctly based on the active model's training configuration. |
| 20 | Colab train.ipynb is NOT the canonical script | Training logic lives in `scripts/03_train.py` only. Re-run `prepare_colab.py` before every Colab upload so the bundle includes the current `03_train.py`. |

---

## Dependencies

```
torch torchvision            # model
torch-geometric              # GATv2Conv
networkx                     # graph construction
numpy pandas pyarrow         # data
nba_api                      # schedule data (ScoreboardV3)
scikit-learn statsmodels     # utilities & metrics
scipy                        # conformal probability (norm.cdf)
tqdm seaborn patsy xgboost   # visualization and analysis
matplotlib                   # plotting
requests python-dotenv       # Kalshi API
cryptography                 # RSA-PSS auth
```

The model source (`gatv2tcn.py`, `tcn.py`) lives at:
```
../networks/ballnet/NBA-GNN-prediction/
```
This path is referenced in `config.py` as `GATV2_SRC` and used by `03_train.py`,
`04_calibrate.py`, and `prepare_colab.py` (which copies the files into `upload/`).
