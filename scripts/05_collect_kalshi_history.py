"""
scripts/05_collect_kalshi_history.py
=====================================
Collects pre-game Kalshi bid/ask prices for NBA player props.

For a given date range:
  1. Fetches NBA tip-off times from nba_api (ScoreboardV3 — fast, no auth).
  2. Lists all NBA player-prop markets from Kalshi for each series/date.
  3. For each market, fetches 1-min candlestick history and finds the last
     candle before tip-off (= the pre-game price).
  4. Saves everything to data/kalshi_pregame_prices.parquet.

No external JSON files needed. Run this scripts any time to extend the dataset.

Output columns (clean flat parquet):
  ticker          — Kalshi market ticker
  event_ticker    — Kalshi event ticker (e.g. KXNBAREB-26JAN16MINHOU)
  game_date       — YYYY-MM-DD string
  player          — player display name from market title
  stat            — PTS / AST / REB / BLK / STL / TO
  threshold       — the line (float, e.g. 24.5)
  result          — 'yes' | 'no' | '' (outcome)
  yes_ask         — pre-game YES ask in cents (int, 1-99)
  yes_bid         — pre-game YES bid in cents (int, 0-98)
  no_ask          — 100 - yes_bid
  no_bid          — 100 - yes_ask
  price_ts        — Unix timestamp of the candle used (int)
  game_start_ts   — Unix timestamp of actual tip-off from nba_api (int)
  price_age_s     — (game_start_ts - price_ts) in seconds (smaller = fresher)
  spread          — yes_ask - yes_bid in cents (market tightness, smaller = more liquid)

Usage:
  # Collect last 30 days:
  python scripts/05_collect_kalshi_history.py

  # Specific date range:
  python scripts/05_collect_kalshi_history.py --start 2026-01-16 --end 2026-02-22

  # Dry-run (show schedule, no API candle calls):
  python scripts/05_collect_kalshi_history.py --start 2026-01-16 --end 2026-01-18 --dry-run
"""
import argparse
import logging
import re
import sys
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import KALSHI_BASE, KALSHI_SERIES, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [05_kalshi]  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH   = DATA_DIR / "kalshi_pregame_prices.parquet"
LOOKBACK_SECS = 4 * 3600    # fetch candles starting 4h before tip-off
RATE_SLEEP    = 0.40         # seconds between Kalshi candle API calls (safer for rate limits)


# ─────────────────────────────────────────────────────────────────
# Step 1: NBA tip-off times via nba_api ScoreboardV3
# ─────────────────────────────────────────────────────────────────

def get_tipoffs_for_dates(dates: List[str]) -> Dict[str, int]:
    """
    Fetch NBA tip-off times for each YYYY-MM-DD date.
    Returns dict: '{date}:{away_abbr}:{home_abbr}' → tipoff_unix_ts.

    Uses ScoreboardV3 which provides gameTimeUTC directly.
    ScoreboardV3 headers are camelCase: gameCode, gameTimeUTC, gameEt, ...
    """
    from nba_api.stats.endpoints import scoreboardv3

    tipoffs: Dict[str, int] = {}

    for date_str in sorted(set(dates)):
        log.info(f"  nba_api: fetching schedule for {date_str}...")
        try:
            sb      = scoreboardv3.ScoreboardV3(game_date=date_str,
                                                league_id="00", timeout=30)
            games   = sb.game_header.get_dict()
            headers = tuple(games["headers"])
            rows    = games["data"]
        except Exception as e:
            log.warning(f"  {date_str}: nba_api failed ({e}), skipping.")
            continue

        # ScoreboardV3 uses camelCase: gameCode='20260116/NOPIND', gameTimeUTC='2026-01-17T00:00:00Z'
        try:
            code_idx = headers.index("gameCode")
            utc_idx  = headers.index("gameTimeUTC")
        except ValueError as e:
            log.warning(f"  {date_str}: unexpected headers {headers[:6]}... ({e})")
            continue

        for row in rows:
            gamecode     = row[code_idx]   # e.g. '20260116/NOPIND'
            game_time_utc = row[utc_idx]   # e.g. '2026-01-17T00:00:00Z'

            # Gamecode format: 'YYYYMMDD/AWYHOM' — always 3-char away, rest = home
            # e.g. '20260116/NOPIND' → away='NOP', home='IND'
            slash = gamecode.find("/")
            if slash < 0:
                continue
            suffix    = gamecode[slash+1:]
            away_abbr = suffix[:3]
            home_abbr = suffix[3:]

            # Parse ISO UTC timestamp directly — no text parsing needed
            try:
                dt = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
                tipoff_unix = int(dt.timestamp())
            except (ValueError, AttributeError):
                log.warning(f"  {date_str}: bad gameTimeUTC '{game_time_utc}' for {away_abbr}@{home_abbr}")
                continue

            key = f"{date_str}:{away_abbr}:{home_abbr}"
            tipoffs[key] = tipoff_unix
            log.info(f"    {away_abbr} @ {home_abbr}  tip-off {game_time_utc}")

        time.sleep(0.5)  # nba_api rate-limit courtesy pause

    log.info(f"  Collected {len(tipoffs)} game tip-offs across {len(dates)} dates.")
    return tipoffs


# ─────────────────────────────────────────────────────────────────
# Step 2: Match event_ticker → tip-off time
# ─────────────────────────────────────────────────────────────────

# Kalshi event ticker: KXNBAREB-26JAN16MINHOU
_TICKER_RE  = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})([A-Z]{3})([A-Z]+)$")
_MONTH_MAP  = {m: i+1 for i, m in enumerate(
    ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}


def parse_event_ticker(event_ticker: str) -> Optional[Tuple[str, str, str]]:
    """Returns (game_date, away_short, home_short) or None."""
    m = _TICKER_RE.search(event_ticker)
    if not m:
        return None
    yy, mon, dd, away, home = m.groups()
    month = _MONTH_MAP.get(mon.upper())
    if not month:
        return None
    return f"{2000+int(yy)}-{month:02d}-{int(dd):02d}", away, home


def find_tipoff_for_event(event_ticker: str,
                          tipoffs: Dict[str, int]) -> Optional[int]:
    """
    Match event_ticker to tipoffs dict.
    Tries exact match first, then substring match (Kalshi city codes ≠ nba_api abbreviations).
    """
    parsed = parse_event_ticker(event_ticker)
    if not parsed:
        return None
    date_str, away_short, home_short = parsed

    # Exact match
    exact = f"{date_str}:{away_short}:{home_short}"
    if exact in tipoffs:
        return tipoffs[exact]

    # Fuzzy match: same date, abbreviation prefix overlap
    for key, ts in tipoffs.items():
        k_date, k_away, k_home = key.split(":")
        if k_date != date_str:
            continue
        away_match = away_short[:3] in k_away or k_away[:3] in away_short
        home_match = home_short[:3] in k_home or k_home[:3] in home_short
        if away_match and home_match:
            return ts

    return None


# ─────────────────────────────────────────────────────────────────
# Step 3: List all Kalshi markets for a date range
# ─────────────────────────────────────────────────────────────────

_TITLE_RE   = re.compile(r"^(.+?):\s*([\d.]+)\+", re.IGNORECASE)


def list_markets_for_dates(start_dt: datetime, end_dt: datetime) -> List[dict]:
    """
    List all finalized/settled Kalshi NBA player-prop markets for a date range.
    Uses the public GET /markets endpoint filtered by series and close_ts.
    No authentication required.
    """
    start_ts = int(start_dt.timestamp())
    end_ts   = int(end_dt.timestamp())
    all_markets: List[dict] = []

    for series, stat in KALSHI_SERIES.items():
        cursor = None
        page   = 0
        while True:
            params = {
                "series_ticker": series,
                "min_close_ts":  start_ts,
                "max_close_ts":  end_ts,
                "limit":         200,
            }
            if cursor:
                params["cursor"] = cursor
            try:
                resp = requests.get(f"{KALSHI_BASE}/markets", params=params, timeout=15)
                if resp.status_code == 429:
                    log.warning(f"  Rate limited (429) on {series}. Waiting 5s...")
                    time.sleep(5)
                    continue
                resp.raise_for_status()
                body    = resp.json()
                markets = body.get("markets", [])
                for m in markets:
                    # Extract player name + threshold from title
                    title = m.get("title", "")
                    match = _TITLE_RE.match(title)
                    if not match:
                        continue
                    player    = match.group(1).strip()
                    threshold = float(match.group(2))
                    all_markets.append({
                        "ticker":       m["ticker"],
                        "event_ticker": m.get("event_ticker", ""),
                        "player":       player,
                        "stat":         stat,
                        "threshold":    threshold,
                        "result":       m.get("result", ""),
                    })
                cursor = body.get("cursor")
                page  += 1
                if not cursor or not markets:
                    break
            except Exception as e:
                log.warning(f"  {series} page {page}: fetch failed ({e})")
                break

        log.info(f"  {series}: {sum(1 for x in all_markets if x['stat']==stat)} markets")

    log.info(f"  Total markets listed: {len(all_markets)}")
    return all_markets


# ─────────────────────────────────────────────────────────────────
# Step 4: Fetch pre-game candlestick price per ticker
# ─────────────────────────────────────────────────────────────────

def get_pregame_price(ticker: str,
                      tipoff_ts: int) -> Optional[dict]:
    """
    Fetch 1-min candlestick history for ticker in [tipoff - LOOKBACK_SECS, tipoff].
    Returns the last candle at or before tip-off.
    Prices are integer cents (0–100).
    """
    series    = ticker.split("-")[0]
    start_ts  = tipoff_ts - LOOKBACK_SECS
    url       = f"{KALSHI_BASE}/series/{series}/markets/{ticker}/candlesticks"
    params    = {"start_ts": start_ts, "end_ts": tipoff_ts, "period_interval": 1}

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            log.debug(f"  candle fetch failed for {ticker}: {resp.status_code} {resp.text}")
            return None
        candles = resp.json().get("candlesticks", [])
    except Exception as e:
        log.debug(f"  candle fetch failed for {ticker}: {e}")
        return None

    if not candles:
        return None

    # Last candle before or at tip-off
    valid = [c for c in candles if c.get("end_period_ts", 0) <= tipoff_ts]
    if not valid:
        valid = [candles[0]]   # fallback: earliest available

    best = max(valid, key=lambda c: c.get("end_period_ts", 0))

    # Try newer V2 schema (close_dollars) then fallback to old V2 schema (close)
    ya = best.get("yes_ask", {}).get("close_dollars")
    yb = best.get("yes_bid", {}).get("close_dollars")
    
    if ya is not None and yb is not None:
        # close_dollars is float-like string or float (e.g. 0.99)
        ya, yb = round(float(ya) * 100), round(float(yb) * 100)
    else:
        ya = best.get("yes_ask", {}).get("close")
        yb = best.get("yes_bid", {}).get("close")
        if ya is not None and yb is not None:
            ya, yb = int(ya), int(yb)
        else:
            return None

    # Skip fully-settled markets (price already at 100/0 before game)
    if ya == 100 and yb == 0:
        return None

    return {
        "yes_ask":    ya,
        "yes_bid":    yb,
        "no_ask":     100 - yb,
        "no_bid":     100 - ya,
        "price_ts":   best["end_period_ts"],
        "spread":     ya - yb,           # bid-ask spread in cents
    }


# ─────────────────────────────────────────────────────────────────
# Step 5: Orchestrate & save
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     default=None,
                        help="Start date YYYY-MM-DD (default: 30 days ago)")
    parser.add_argument("--end",       default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="List markets and tip-offs but skip candle fetching.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output; default appends new rows.")
    args = parser.parse_args()

    today     = date.today()
    end_date  = date.fromisoformat(args.end)   if args.end   else today
    start_date= date.fromisoformat(args.start) if args.start else today - timedelta(days=30)

    # Convert to UTC datetimes bracketing each calendar day broadly
    start_dt = datetime(start_date.year, start_date.month, start_date.day,
                        0, 0, tzinfo=timezone.utc)
    end_dt   = datetime(end_date.year,   end_date.month,   end_date.day,
                        23, 59, 59, tzinfo=timezone.utc)

    log.info(f"Date range: {start_date} → {end_date}")
    log.info(f"Output:     {OUTPUT_PATH}")

    # ── Load existing data (for append mode) ─────────────────────
    existing_tickers: set = set()
    df_existing = None
    if OUTPUT_PATH.exists() and not args.overwrite:
        df_existing = pd.read_parquet(OUTPUT_PATH)
        existing_tickers = set(df_existing["ticker"].tolist())
        log.info(f"Loaded {len(df_existing)} existing rows. Will skip {len(existing_tickers)} tickers already collected.")

    # ── Step 1: Fetch tip-off times ───────────────────────────────
    dates = []
    d = start_date
    while d <= end_date:
        dates.append(d.isoformat())
        d += timedelta(days=1)

    log.info(f"\nFetching NBA tip-off times for {len(dates)} dates...")
    tipoffs = get_tipoffs_for_dates(dates)
    if not tipoffs:
        log.error("No tip-off times retrieved. Check nba_api connectivity.")
        sys.exit(1)

    # ── Step 2: List Kalshi markets ───────────────────────────────
    log.info(f"\nListing Kalshi markets for date range...")
    # Extend end by +2 days: NBA games played on date X close on the morning of X+1.
    market_end_dt = datetime(end_date.year, end_date.month, end_date.day,
                             23, 59, 59, tzinfo=timezone.utc) + timedelta(days=2)
    markets = list_markets_for_dates(start_dt, market_end_dt)

    # Filter out tickers we already have
    new_markets = [m for m in markets if m["ticker"] not in existing_tickers]
    log.info(f"{len(new_markets)} new markets to fetch prices for "
             f"({len(markets) - len(new_markets)} already in dataset).")

    if args.dry_run:
        log.info(f"\n[DRY RUN] Would fetch candles for {len(new_markets)} markets.")
        log.info(f"Sample markets:")
        for m in new_markets[:5]:
            tipoff_ts = find_tipoff_for_event(m["event_ticker"], tipoffs)
            log.info(f"  {m['player']} {m['threshold']}+ {m['stat']}"
                     f"  tip-off={'found' if tipoff_ts else 'MISSING'}")
        return

    # ── Step 3: Fetch pre-game prices ────────────────────────────
    rows: List[dict] = []
    no_tipoff  = no_candles = collected = 0

    # Group by event_ticker to amortize tip-off lookups
    from itertools import groupby
    new_markets.sort(key=lambda x: x["event_ticker"])

    total = len(new_markets)
    for i, market in enumerate(new_markets, 1):
        if i % 100 == 0 or i == 1:
            log.info(f"  [{i}/{total}] Fetching candles...")

        tipoff_ts = find_tipoff_for_event(market["event_ticker"], tipoffs)
        if tipoff_ts is None:
            no_tipoff += 1
            continue

        parsed = parse_event_ticker(market["event_ticker"])
        game_date = parsed[0] if parsed else ""

        prices = get_pregame_price(market["ticker"], tipoff_ts)
        time.sleep(RATE_SLEEP)

        if prices is None:
            no_candles += 1
            continue

        rows.append({
            "ticker":        market["ticker"],
            "event_ticker":  market["event_ticker"],
            "game_date":     game_date,
            "player":        market["player"],
            "stat":          market["stat"],
            "threshold":     market["threshold"],
            "result":        market["result"],
            "yes_ask":       prices["yes_ask"],
            "yes_bid":       prices["yes_bid"],
            "no_ask":        prices["no_ask"],
            "no_bid":        prices["no_bid"],
            "price_ts":      prices["price_ts"],
            "game_start_ts": tipoff_ts,
            "price_age_s":   tipoff_ts - prices["price_ts"],
            "spread":        prices["spread"],
        })
        collected += 1

    log.info(f"\nResults: {collected} collected | {no_tipoff} no tip-off | {no_candles} no candles")

    if not rows and df_existing is None:
        log.warning("No new rows collected. Exiting without saving.")
        return

    # ── Step 4: Save ──────────────────────────────────────────────
    df_new = pd.DataFrame(rows)
    if df_existing is not None and not df_new.empty:
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    elif df_existing is not None:
        df_out = df_existing
    else:
        df_out = df_new

    df_out.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"\n✓ Saved {len(df_out)} total rows to {OUTPUT_PATH}")
    log.info(f"  New rows added: {len(df_new)}")
    log.info(f"  Date coverage: {df_out['game_date'].min()} → {df_out['game_date'].max()}")
    log.info(f"  Next: delete data/ops_*_cache.parquet and re-run backtest.py")


if __name__ == "__main__":
    main()
