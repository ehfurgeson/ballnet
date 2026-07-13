"""
scripts/01_fetch_data.py
========================
Full historical NBA boxscore scrape → data/raw_boxscores.parquet

Run once to initialise the dataset, or periodically to backfill any missed games.
For daily incremental updates, use update.py instead (much faster — only fetches
new games since the last cached date).

Usage:
  python scripts/01_fetch_data.py       # run from clean/ directory

Output:
  data/raw_boxscores.parquet
"""

import itertools
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from nba_api.stats.endpoints import (
    boxscoreadvancedv3,
    boxscoreplayertrackv3,
    boxscoretraditionalv3,
    leaguegamelog,
)
from nba_api.library import http as nba_http

# ── Paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR, FEATURE_COLS, MIN_MINUTES

DATA_DIR.mkdir(parents=True, exist_ok=True)
EMPTY_GAMES_PATH = DATA_DIR / "raw_boxscores_empty_games.txt"

# ── Config ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  [fetch_data]  %(message)s")
log = logging.getLogger(__name__)

SEASONS = [
    ("2025-26", "2025-10-28", str(date.today())),
]
TRAD_COLS  = ["GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
              "MIN", "PTS", "AST", "REB", "TO", "STL", "BLK", "PLUS_MINUS"]
ADV_COLS   = ["GAME_ID", "PLAYER_ID", "PACE", "USG_PCT", "TS_PCT"]
API_DELAY  = 2.5
MAX_GAMES_PER_RUN = 199   # kamikaze threshold
KAMIKAZE_CODE     = 42


def reset_session():
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=0, pool_maxsize=0, max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    nba_http.requests_session = session
    return session


def call_api(endpoint_class, *args, retries=6, **kwargs):
    global session
    kwargs.setdefault("timeout", (10, 15))
    for attempt in range(retries):
        try:
            return endpoint_class(*args, **kwargs).get_data_frames()[0]
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = (API_DELAY * (1.5 ** attempt)) + np.random.uniform(2, 8)
            log.warning(f"  Retry {attempt+1}/{retries} in {wait:.1f}s: {exc}")
            time.sleep(wait)


def get_game_ids(season, date_from, date_to):
    time.sleep(API_DELAY + np.random.uniform(1, 5))
    df = call_api(leaguegamelog.LeagueGameLog, season=season, league_id="00",
                  season_type_all_star="Regular Season")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    mask = (df["GAME_DATE"] >= pd.to_datetime(date_from)) & \
           (df["GAME_DATE"] <= pd.to_datetime(date_to))
    return df.loc[mask].set_index("GAME_ID")["GAME_DATE"].dt.strftime("%Y-%m-%d").to_dict()


def fetch_game(game_id):
    try:
        time.sleep(np.random.uniform(2, 4))
        dt = call_api(boxscoretraditionalv3.BoxScoreTraditionalV3, game_id=game_id)
        da = call_api(boxscoreadvancedv3.BoxScoreAdvancedV3,       game_id=game_id)
        dk = call_api(boxscoreplayertrackv3.BoxScorePlayerTrackV3, game_id=game_id)
    except Exception as e:
        log.warning(f"  Skip {game_id}: {e}")
        return None

    trad_map = {"gameId":"GAME_ID","personId":"PLAYER_ID","teamId":"TEAM_ID",
                "teamTricode":"TEAM_ABBREVIATION","minutes":"MIN","points":"PTS",
                "assists":"AST","reboundsTotal":"REB","turnovers":"TO",
                "steals":"STL","blocks":"BLK","plusMinusPoints":"PLUS_MINUS"}
    dt = dt.rename(columns=trad_map)
    if "firstName" in dt.columns:
        dt["PLAYER_NAME"] = (dt["firstName"].fillna("") + " " + dt["familyName"].fillna("")).str.strip()
    da = da.rename(columns={"gameId":"GAME_ID","personId":"PLAYER_ID",
                             "pace":"PACE","usagePercentage":"USG_PCT","trueShootingPercentage":"TS_PCT"})
    dk.columns = dk.columns.str.upper()
    dk = dk.rename(columns={"GAMEID":"GAME_ID","PERSONID":"PLAYER_ID"})
    for want, variants in [("DIST",["DIST","DISTANCE"]),("TCHS",["TCHS","TOUCHES"]),("PASS",["PASS","PASSES"])]:
        for v in variants:
            if v in dk.columns:
                dk = dk.rename(columns={v: want}); break

    avt = [c for c in TRAD_COLS if c in dt.columns]
    ava = [c for c in ADV_COLS  if c in da.columns]
    avk = ["GAME_ID","PLAYER_ID"] + [c for c in ["DIST","TCHS","PASS"] if c in dk.columns]
    df  = dt[avt].merge(da[ava], on=["GAME_ID","PLAYER_ID"], how="left")
    df  = df.merge(dk[avk], on=["GAME_ID","PLAYER_ID"], how="left")

    def parse_min(m):
        if pd.isna(m) or str(m).strip() == "": return 0.0
        if ":" in str(m): p = str(m).split(":"); return float(p[0]) + float(p[1])/60
        return float(m)
    df["MIN"] = df["MIN"].apply(parse_min)
    return df[df["MIN"] >= MIN_MINUTES].copy()


def main():
    global session
    session = reset_session()
    raw_path = DATA_DIR / "raw_boxscores.parquet"

    if raw_path.exists():
        existing = pd.read_parquet(raw_path)
        all_frames = [existing]
        done = set(existing["GAME_ID"].unique())
        log.info(
            f"Resuming: {len(done)} games already cached "
            f"({len(existing)} rows) from {raw_path}"
        )
    else:
        all_frames, done = [], set()

    # Also treat previously-seen empty games as done so we don't refetch forever.
    if EMPTY_GAMES_PATH.exists():
        try:
            with open(EMPTY_GAMES_PATH, "r") as f:
                empty_ids = {line.strip() for line in f if line.strip()}
            done.update(empty_ids)
            if empty_ids:
                log.info(f"Found {len(empty_ids)} empty games already marked as done.")
        except Exception as e:
            log.warning(f"Could not read {EMPTY_GAMES_PATH}: {e}")

    games_this_run = 0
    for season, d_from, d_to in SEASONS:
        log.info(f"Season: {season}")
        game_map = get_game_ids(season, d_from, d_to)
        new_ids  = [g for g in game_map if g not in done]
        log.info(f"  {len(new_ids)} games to fetch")
        for idx, gid in enumerate(new_ids):
            log.info(f"  [{idx+1}/{len(new_ids)}] {gid}")
            df_game = fetch_game(gid)
            if df_game is not None:
                df_game["SEASON"] = season
                df_game["GAME_DATE"] = game_map[gid]
                if df_game.empty:
                    log.info(
                        f"  Game {gid} has no players with >= {MIN_MINUTES} minutes; "
                        "marking as done without rows."
                    )
                    done.add(gid)
                    try:
                        with open(EMPTY_GAMES_PATH, "a") as f:
                            f.write(str(gid) + "\n")
                    except Exception as e:
                        log.warning(f"Could not append {gid} to {EMPTY_GAMES_PATH}: {e}")
                else:
                    all_frames.append(df_game)
                    done.add(gid)
            if (idx + 1) % 25 == 0:
                combined = pd.concat(all_frames, ignore_index=True)
                log.info(
                    f"Saving checkpoint after {idx+1} games "
                    f"({combined['GAME_ID'].nunique()} unique games, "
                    f"{len(combined)} rows) to {raw_path}"
                )
                combined.to_parquet(raw_path, index=False)
                session.close(); session = reset_session()
                time.sleep(120)
            games_this_run += 1
            if games_this_run >= MAX_GAMES_PER_RUN:
                combined = pd.concat(all_frames, ignore_index=True)
                log.info(
                    f"Kamikaze save: {combined['GAME_ID'].nunique()} games, "
                    f"{len(combined)} rows to {raw_path}"
                )
                combined.to_parquet(raw_path, index=False)
                log.info(f"Kamikaze: {MAX_GAMES_PER_RUN} games. Restarting...")
                sys.exit(KAMIKAZE_CODE)

    combined = pd.concat(all_frames, ignore_index=True)
    log.info(
        f"Final save: {combined['GAME_ID'].nunique()} games, "
        f"{len(combined)} rows to {raw_path}"
    )
    combined.to_parquet(raw_path, index=False)
    log.info(f"✓ Done. Saved to {raw_path}")


if __name__ == "__main__":
    main()
