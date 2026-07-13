"""
scripts/02_build_tensors.py
===========================
Converts raw_boxscores.parquet → all tensor/pickle artifacts in data/.

Run after 01_fetch_data.py and before 03_train.py.
After running this, retraining is required if new players were added.

Input:  data/raw_boxscores.parquet
Output: data/X_seq.pkl, X_raw.pkl, G_seq.pkl, player_ids.pkl,
        game_dates.pkl, day_seasons.pkl, team_temporal.pkl,
        pos_temporal.pkl, n_teams.pkl, mu_per_day.npy, sd_per_day.npy
"""
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR, FEATURE_COLS, MIN_MINUTES

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  [build_tensors]  %(message)s")
log = logging.getLogger(__name__)


POS_MAP = {
    "G": [0, 1, 0], "F": [1, 0, 0], "C": [0, 0, 1],
    "F-G": [1, 1, 0], "G-F": [1, 1, 0],
    "F-C": [1, 0, 1], "C-F": [1, 0, 1],
}


def build():
    raw = pd.read_parquet(DATA_DIR / "raw_boxscores.parquet")
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"]).dt.date.astype(str)

    player_ids = sorted(raw["PLAYER_ID"].unique())
    game_dates = sorted(raw["GAME_DATE"].unique())

    # ── Pre-flight: compare against existing tensors ──────────────
    existing_pid_path = DATA_DIR / "player_ids.pkl"
    if existing_pid_path.exists():
        existing_pids = pickle.load(open(existing_pid_path, "rb"))
        existing_P    = len(existing_pids)
        new_P         = len(player_ids)
        added         = set(player_ids) - set(existing_pids)
        removed       = set(existing_pids) - set(player_ids)

        log.info("─" * 55)
        log.info("PRE-FLIGHT CHECK — Existing tensors vs. new build")
        log.info(f"  Existing player_ids.pkl : {existing_P} players")
        log.info(f"  New build (from parquet): {new_P} players")
        if added:
            log.info(f"  ➕ Players added  : {len(added)}  "
                     f"(IDs: {sorted(added)[:10]}{'...' if len(added) > 10 else ''})")
        if removed:
            log.info(f"  ➖ Players removed: {len(removed)} "
                     f"(IDs: {sorted(removed)[:10]}{'...' if len(removed) > 10 else ''})")
        if added or removed:
            log.info("  ⚠️  RETRAIN REQUIRED — player index has changed.")
            log.info("      Run scripts/03_train.py then scripts/04_calibrate.py")
        else:
            log.info("  ✅ Player set unchanged — no retraining required.")
            log.info("     Tensor shapes will differ only in the day dimension.")
        log.info("─" * 55)
    else:
        log.info("No existing tensors found — this is a fresh build.")

    day_seasons = []


    # Season boundaries — sync with 01_fetch_data.py
    SEASONS = [
        ("2022-23", "2022-10-18", "2023-04-09"),
        ("2023-24", "2023-10-24", "2024-04-14"),
        ("2024-25", "2024-10-22", "2025-04-13"),
        ("2025-26", "2025-10-28", "9999-12-31"),
    ]
    def date_to_season(d):
        for name, start, end in SEASONS:
            if start <= d <= end:
                return name
        return SEASONS[-1][0]

    P, F = len(player_ids), len(FEATURE_COLS)
    D = len(game_dates)
    pid2idx = {pid: i for i, pid in enumerate(player_ids)}

    teams_all = sorted(raw["TEAM_ABBREVIATION"].dropna().unique())
    team2idx  = {t: i for i, t in enumerate(teams_all)}
    n_teams   = len(teams_all)

    # Last-known team and position for forward-filling
    last_team: dict[int, int] = {}
    last_pos:  dict[int, list] = {}

    X_raw   = np.zeros((D, P, F), dtype=np.float32)
    X_seq   = np.zeros((D, P, F), dtype=np.float32)
    tt_arr  = np.zeros((D, P, n_teams), dtype=np.float32)
    pt_arr  = np.zeros((D, P, 3),      dtype=np.float32)
    G_seq   = []

    log.info(f"Building tensors: {D} days, {P} players, {F} features")

    # player_id2team / player_id2position — used by 03_train.py (static per-player)
    # We track the most recent team/position seen for each player.
    player_id2team: dict[int, str]             = {}
    player_id2pos:  dict[int, np.ndarray]      = {}
    # Default position vec for players with no position data
    default_pos = np.array([0, 0, 0], dtype=np.int32)

    # Build season-level mu/sd (one vector per season, extended per day)
    # We compute them per-season then forward-extend on a per-day basis after the loop.
    season_stats: dict[str, list] = {}

    for di, gd in enumerate(game_dates):
        day_season = date_to_season(gd)
        day_seasons.append(day_season)
        day_df = raw[raw["GAME_DATE"] == gd]

        x_raw_day = np.zeros((P, F), dtype=np.float32)
        G = nx.Graph()
        G.add_nodes_from(player_ids)

        for gid, grp in day_df.groupby("GAME_ID"):
            active = grp["PLAYER_ID"].tolist()
            for pid in active:
                if pid in pid2idx:
                    pidx = pid2idx[pid]
                    row = grp[grp["PLAYER_ID"] == pid].iloc[0]
                    vals = [float(row.get(c, 0) or 0) for c in FEATURE_COLS]
                    x_raw_day[pidx] = vals
                    tm = row.get("TEAM_ABBREVIATION", "")
                    if tm and tm in team2idx:
                        last_team[pid] = team2idx[tm]
                    # Track last-known team name (string) and position for 03_train.py
                    tm_name = row.get("TEAM_ABBREVIATION", "")
                    if tm_name:
                        player_id2team[pid] = tm_name
                    pos_raw = row.get("POSITION", row.get("START_POSITION", ""))
                    if pos_raw and str(pos_raw).strip():
                        pos_str = str(pos_raw).strip()
                        # Map to [F, G, C] binary vector
                        pv = [int("F" in pos_str), int("G" in pos_str), int("C" in pos_str)]
                        player_id2pos[pid] = np.array(pv, dtype=np.int32)
            from itertools import combinations
            for pA, pB in combinations([p for p in active if p in pid2idx], 2):
                G.add_edge(pA, pB)

        # Build team/pos one-hot for this day
        for pidx, pid in enumerate(player_ids):
            if pid in last_team and last_team[pid] < n_teams:
                tt_arr[di, pidx, last_team[pid]] = 1.0
            if pid in last_pos:
                pt_arr[di, pidx] = last_pos[pid]

        X_raw[di] = x_raw_day
        G_seq.append(G)

        # Accumulate for per-season stats
        season_stats.setdefault(day_season, []).append(x_raw_day)

        if (di + 1) % 50 == 0:
            log.info(f"  Processed {di+1}/{D} days")

    # Forward-fill X_seq (zeros become last known value)
    X_seq[0] = X_raw[0]
    for di in range(1, D):
        X_seq[di] = X_raw[di].copy()
        mask = (X_seq[di] == 0).all(axis=-1)
        X_seq[di, mask] = X_seq[di-1, mask]

    # Compute causal sliding window mu/sd to prevent lookahead bias.
    # We use a trailing window of the last W active days.
    W = 150  # ~1 season worth of active game days
    day_sums = np.zeros((D, F))
    day_sq_sums = np.zeros((D, F))
    day_counts = np.zeros((D, 1))

    for di in range(D):
        x_raw_day = X_raw[di]
        active = (x_raw_day != 0).any(axis=-1)
        active_data = x_raw_day[active]
        if len(active_data) > 0:
            day_sums[di] = active_data.sum(axis=0)
            day_sq_sums[di] = (active_data ** 2).sum(axis=0)
            day_counts[di] = len(active_data)

    cum_sums = np.cumsum(day_sums, axis=0)
    cum_sq_sums = np.cumsum(day_sq_sums, axis=0)
    cum_counts = np.cumsum(day_counts, axis=0)

    causal_sums = np.zeros_like(cum_sums)
    causal_sq_sums = np.zeros_like(cum_sq_sums)
    causal_counts = np.zeros_like(cum_counts)

    for di in range(1, D):
        start_idx = max(0, di - W)
        if start_idx == 0:
            causal_sums[di] = cum_sums[di - 1]
            causal_sq_sums[di] = cum_sq_sums[di - 1]
            causal_counts[di] = cum_counts[di - 1]
        else:
            causal_sums[di] = cum_sums[di - 1] - cum_sums[start_idx - 1]
            causal_sq_sums[di] = cum_sq_sums[di - 1] - cum_sq_sums[start_idx - 1]
            causal_counts[di] = cum_counts[di - 1] - cum_counts[start_idx - 1]

    # For day 0 (no history), fall back to day 0's stats
    causal_sums[0] = day_sums[0]
    causal_sq_sums[0] = day_sq_sums[0]
    causal_counts[0] = day_counts[0]

    safe_counts = np.where(causal_counts < 1, 1, causal_counts)
    mu_per_day = causal_sums / safe_counts
    var = (causal_sq_sums / safe_counts) - (mu_per_day ** 2)
    sd_per_day = np.sqrt(np.maximum(var, 0))
    sd_per_day[sd_per_day < 1e-6] = 1.0

    mu_per_day = mu_per_day[:, None, :]  # (D, 1, F)
    sd_per_day = sd_per_day[:, None, :]  # (D, 1, F)

    log.info("Saving artifacts...")
    with open(DATA_DIR / "X_seq.pkl",         "wb") as f: pickle.dump(X_seq,      f)
    with open(DATA_DIR / "X_raw.pkl",         "wb") as f: pickle.dump(X_raw,      f)
    with open(DATA_DIR / "G_seq.pkl",         "wb") as f: pickle.dump(G_seq,      f)
    with open(DATA_DIR / "player_ids.pkl",    "wb") as f: pickle.dump(player_ids, f)
    with open(DATA_DIR / "game_dates.pkl",    "wb") as f: pickle.dump(game_dates, f)
    with open(DATA_DIR / "day_seasons.pkl",   "wb") as f: pickle.dump(day_seasons, f)
    with open(DATA_DIR / "team_temporal.pkl",       "wb") as f: pickle.dump(tt_arr,         f)
    with open(DATA_DIR / "pos_temporal.pkl",        "wb") as f: pickle.dump(pt_arr,          f)
    with open(DATA_DIR / "n_teams.pkl",             "wb") as f: pickle.dump(n_teams,         f)
    # Fill defaults for players who never had position data recorded
    for pid in player_ids:
        if pid not in player_id2pos:
            player_id2pos[pid] = default_pos
    with open(DATA_DIR / "player_id2team.pkl",      "wb") as f: pickle.dump(player_id2team,  f)
    with open(DATA_DIR / "player_id2position.pkl",  "wb") as f: pickle.dump(player_id2pos,   f)
    np.save(DATA_DIR / "mu_per_day.npy", mu_per_day)
    np.save(DATA_DIR / "sd_per_day.npy", sd_per_day)
    log.info(f"✓ Done. {D} days, {P} players, {n_teams} teams.")
    log.info(f"  player_id2team: {len(player_id2team)} entries")
    log.info(f"  player_id2pos:  {len(player_id2pos)} entries")


if __name__ == "__main__":
    build()
