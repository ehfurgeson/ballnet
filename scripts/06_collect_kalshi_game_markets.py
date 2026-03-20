"""
scripts/06_collect_kalshi_game_markets.py
==========================================
Collects pre-game Kalshi prices for NBA game-level markets:
  - KXNBAGAME   → moneyline (outright winner per team)
  - KXNBASPREAD → spread markets (team wins by N+ points)
  - KXNBATOTAL  → game total points (optional, included by default)

For each market, fetches the last 1-min candlestick before tip-off
(same approach as 05_collect_kalshi_history.py for player props).

Output: data/kalshi_game_markets.parquet

Output columns:
  ticker          — Kalshi ticker
  event_ticker    — game identifier (e.g. KXNBAGAME-26MAR08CHAPHX)
  market_type     — 'moneyline' | 'spread' | 'total'
  game_date       — YYYY-MM-DD
  home_team       — 3-char Kalshi home abbreviation
  away_team       — 3-char Kalshi away abbreviation
  subject_team    — team the market refers to (spread/moneyline) or '' for total
  line            — spread threshold or total line (float), or 0.0 for moneyline
  result          — 'yes' | 'no' | '' (outcome)
  yes_ask         — pre-game YES ask in cents (int, 1-99)
  yes_bid         — pre-game YES bid in cents (int, 0-98)
  no_ask          — 100 - yes_bid
  no_bid          — 100 - yes_ask
  price_ts        — Unix timestamp of candle used
  game_start_ts   — Unix timestamp of tip-off from nba_api
  price_age_s     — (game_start_ts - price_ts) in seconds
  spread          — yes_ask - yes_bid (market tightness)

Usage:
  # Specific date range:
  python scripts/06_collect_kalshi_game_markets.py --start 2026-01-16 --end 2026-03-01

  # Default: last 30 days
  python scripts/06_collect_kalshi_game_markets.py

  # Preview without writing candle data:
  python scripts/06_collect_kalshi_game_markets.py --start 2026-01-16 --end 2026-01-18 --dry-run

  # Skip totals:
  python scripts/06_collect_kalshi_game_markets.py --no-totals
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
from config import KALSHI_BASE, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [06_game_mkts]  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH   = DATA_DIR / "kalshi_game_markets.parquet"
LOOKBACK_SECS = 4 * 3600     # fetch candles starting 4h before tip-off
RATE_SLEEP    = 0.40          # seconds between candle API calls (safer for rate limits)

# Series → market_type mapping
GAME_SERIES = {
    "KXNBAGAME":   "moneyline",
    "KXNBASPREAD": "spread",
    "KXNBATOTAL":  "total",
}

# ─────────────────────────────────────────────────────────────────
# Ticker/title parsing
# ─────────────────────────────────────────────────────────────────

# Event ticker: KXNBAGAME-26MAR08CHAPHX  or  KXNBASPREAD-26MAR07GSWOKC
_EVENT_RE  = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})([A-Z]{3})([A-Z]+)$")
_MONTH_MAP = {m: i + 1 for i, m in enumerate(
    ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}

# Moneyline ticker suffix: -PHX  (team abbrev who wins)
# Spread ticker suffix:   -OKC7  (team abbrev + integer half-point line × 2)
#   e.g. OKC7 → OKC wins by > 7.5 pts (the integer is line * 2 - 0.5 adjustment built in)
# Total ticker suffix:    -234   (integer = threshold × 2, so 234 → 234.5 pts? or 234 pts exactly?)
_SPREAD_SUFFIX_RE = re.compile(r"^([A-Z]+?)(\d+)$")


def parse_event_ticker(event_ticker: str) -> Optional[Tuple[str, str, str]]:
    """Return (game_date, away_3char, home_3char) or None."""
    m = _EVENT_RE.search(event_ticker)
    if not m:
        return None
    yy, mon, dd, away, home = m.groups()
    month = _MONTH_MAP.get(mon.upper())
    if not month:
        return None
    return f"{2000+int(yy)}-{month:02d}-{int(dd):02d}", away, home


def parse_market_ticker(ticker: str, market_type: str) -> Tuple[str, float]:
    """
    Parse the subject team and line from the ticker suffix.

    Returns (subject_team, line) where:
      moneyline: subject_team = winning team abbrev, line = 0.0
      spread:    subject_team = favored team abbrev, line = N + 0.5 (e.g. '7' → 7.5)
      total:     subject_team = '', line = N + 0.5 (e.g. '234' → 234.5)
    """
    # Everything after the last '-'
    suffix = ticker.rsplit("-", 1)[-1]   # e.g. 'PHX', 'OKC7', '234'

    if market_type == "moneyline":
        # Suffix is purely alphabetic = team abbreviation
        return suffix, 0.0

    if market_type == "total":
        # Suffix is purely numeric; Kalshi encodes as integer where the real line = int + 0.5
        try:
            return "", float(suffix) + 0.5
        except ValueError:
            return "", 0.0

    # spread: 'OKC7' → team='OKC', line=7.5
    m = _SPREAD_SUFFIX_RE.match(suffix)
    if m:
        team = m.group(1)
        try:
            line = float(m.group(2)) + 0.5
        except ValueError:
            line = 0.0
        return team, line
    return suffix, 0.0


# ─────────────────────────────────────────────────────────────────
# Tip-off times (reused from 05_collect_kalshi_history.py)
# ─────────────────────────────────────────────────────────────────

def get_tipoffs_for_dates(dates: List[str]) -> Dict[str, int]:
    """
    Returns dict: '{date}:{away_abbr}:{home_abbr}' → tipoff_unix_ts.
    Tries ScoreboardV3 (camelCase headers, available in nba_api ≥ 1.4) and
    falls back to ScoreboardV2 (UPPER_CASE headers, different game-code field)
    automatically so both old and new installs work.
    """
    # Detect which scoreboard version is available once
    try:
        from nba_api.stats.endpoints import scoreboardv3 as _sb_mod
        _SB_VERSION = 3
    except ImportError:
        log.error("ScoreboardV3 is required. Please update nba_api.")
        sys.exit(1)
    log.info(f"  Using ScoreboardV{_SB_VERSION}")

    tipoffs: Dict[str, int] = {}
    for date_str in sorted(set(dates)):
        log.info(f"  nba_api: fetching schedule for {date_str}...")
        try:
            if _SB_VERSION == 3:
                sb    = _sb_mod.ScoreboardV3(game_date=date_str, league_id="00", timeout=30)
                games = sb.game_header.get_dict()
            else:
                sb    = _sb_mod.ScoreboardV2(game_date=date_str, league_id="00", timeout=30)
                # V2 exposes game_header as the first data set
                games = sb.game_header.get_dict()
            headers = tuple(games["headers"])
            rows    = games["data"]
        except Exception as e:
            log.warning(f"  {date_str}: nba_api failed ({e}), skipping.")
            continue

        # V3 uses camelCase; V2 uses UPPER_CASE and "GAMECODE" / "GAME_STATUS_TEXT"
        # We need the game code column and the UTC tip-off column.
        code_col = "gameCode"    if _SB_VERSION == 3 else "GAMECODE"
        utc_col  = "gameTimeUTC" if _SB_VERSION == 3 else "GAME_DATE_EST"

        try:
            code_idx = headers.index(code_col)
            utc_idx  = headers.index(utc_col)
        except ValueError:
            # Some V2 builds use slightly different names — try common alternatives
            for alt in ("GAME_CODE", "gamecode"):
                if alt in headers:
                    code_idx = headers.index(alt); break
            else:
                log.warning(f"  {date_str}: cannot find game-code column in {headers[:6]}…")
                continue
            for alt in ("GAME_DATE_EST", "GAME_TIME_UTC", "gameTimeUTC"):
                if alt in headers:
                    utc_idx = headers.index(alt); break
            else:
                log.warning(f"  {date_str}: cannot find tip-off time column in {headers[:6]}…")
                continue

        for row in rows:
            gamecode = row[code_idx]
            game_utc = row[utc_idx]
            slash = str(gamecode).find("/")
            if slash < 0:
                continue
            suffix    = gamecode[slash + 1:]
            away_abbr = suffix[:3]
            home_abbr = suffix[3:]
            try:
                dt = datetime.fromisoformat(str(game_utc).replace("Z", "+00:00"))
                tipoff_unix = int(dt.timestamp())
            except (ValueError, AttributeError):
                continue
            key = f"{date_str}:{away_abbr}:{home_abbr}"
            tipoffs[key] = tipoff_unix
            log.info(f"    {away_abbr} @ {home_abbr}  tip-off {game_utc}")

        time.sleep(0.5)

    log.info(f"  Collected {len(tipoffs)} tip-offs across {len(dates)} dates.")
    return tipoffs


def find_tipoff_for_event(event_ticker: str, tipoffs: Dict[str, int]) -> Optional[int]:
    """Match event_ticker to tipoffs dict (exact then fuzzy)."""
    parsed = parse_event_ticker(event_ticker)
    if not parsed:
        return None
    date_str, away_short, home_short = parsed

    exact = f"{date_str}:{away_short}:{home_short}"
    if exact in tipoffs:
        return tipoffs[exact]

    # Fuzzy: same date, prefix overlap
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
# Market listing
# ─────────────────────────────────────────────────────────────────

def list_game_markets(start_dt: datetime, end_dt: datetime,
                      include_totals: bool = True) -> List[dict]:
    """List all game-level markets in the date range."""
    start_ts = int(start_dt.timestamp())
    end_ts   = int(end_dt.timestamp())
    all_markets: List[dict] = []

    for series, market_type in GAME_SERIES.items():
        if not include_totals and market_type == "total":
            continue
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
                    ev = m.get("event_ticker", "")
                    parsed = parse_event_ticker(ev)
                    if not parsed:
                        continue
                    game_date, away, home = parsed
                    subject_team, line = parse_market_ticker(m["ticker"], market_type)
                    all_markets.append({
                        "ticker":       m["ticker"],
                        "event_ticker": ev,
                        "market_type":  market_type,
                        "game_date":    game_date,
                        "away_team":    away,
                        "home_team":    home,
                        "subject_team": subject_team,
                        "line":         line,
                        "result":       m.get("result", ""),
                    })
                cursor = body.get("cursor")
                page  += 1
                if not cursor or not markets:
                    break
            except Exception as e:
                log.warning(f"  {series} page {page}: fetch failed ({e})")
                break

        count = sum(1 for x in all_markets if x["market_type"] == market_type)
        log.info(f"  {series} ({market_type}): {count} markets total")

    log.info(f"  Total game markets listed: {len(all_markets)}")
    return all_markets


# ─────────────────────────────────────────────────────────────────
# Candlestick price fetch (identical logic to 05_collect_kalshi_history.py)
# ─────────────────────────────────────────────────────────────────

def get_pregame_price(ticker: str, tipoff_ts: int) -> Optional[dict]:
    """
    Fetch the last 1-min candle before tip-off.
    Returns dict with yes_ask/yes_bid/no_ask/no_bid/price_ts/spread, or None.
    """
    series   = ticker.split("-")[0]
    start_ts = tipoff_ts - LOOKBACK_SECS
    url      = f"{KALSHI_BASE}/series/{series}/markets/{ticker}/candlesticks"
    params   = {"start_ts": start_ts, "end_ts": tipoff_ts, "period_interval": 1}

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

    valid = [c for c in candles if c.get("end_period_ts", 0) <= tipoff_ts]
    if not valid:
        valid = [candles[0]]

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

    # Skip already-settled markets (price at 100/0 before game)
    if ya == 100 and yb == 0:
        return None

    return {
        "yes_ask":  ya,
        "yes_bid":  yb,
        "no_ask":   100 - yb,
        "no_bid":   100 - ya,
        "price_ts": best["end_period_ts"],
        "spread":   ya - yb,
    }


# ─────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect pre-game Kalshi game market prices.")
    parser.add_argument("--start",      default=None,
                        help="Start date YYYY-MM-DD (default: 30 days ago)")
    parser.add_argument("--end",        default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="List markets and tip-offs but skip candle fetching.")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Overwrite existing output; default appends new rows.")
    parser.add_argument("--no-totals",  action="store_true",
                        help="Skip KXNBATOTAL game total markets.")
    args = parser.parse_args()

    today      = date.today()
    end_date   = date.fromisoformat(args.end)   if args.end   else today
    start_date = date.fromisoformat(args.start) if args.start else today - timedelta(days=30)

    start_dt = datetime(start_date.year, start_date.month, start_date.day, 0, 0, tzinfo=timezone.utc)
    end_dt   = datetime(end_date.year,   end_date.month,   end_date.day, 23, 59, 59, tzinfo=timezone.utc)

    log.info(f"Date range: {start_date} → {end_date}")
    log.info(f"Output:     {OUTPUT_PATH}")

    # ── Load existing data (for append mode) ─────────────────────
    existing_tickers: set = set()
    df_existing = None
    if OUTPUT_PATH.exists() and not args.overwrite:
        df_existing = pd.read_parquet(OUTPUT_PATH)
        existing_tickers = set(df_existing["ticker"].tolist())
        log.info(f"Loaded {len(df_existing)} existing rows. "
                 f"Skipping {len(existing_tickers)} existing tickers.")

    # ── Step 1: Tip-off times ─────────────────────────────────────
    # ± 1 day buffer: Kalshi tickers use US east-coast dates but nba_api
    # ScoreboardV3 schedules late-night games under the next UTC calendar date.
    # Fetching one extra day on each side ensures all tip-offs are captured.
    dates = []
    d = start_date - timedelta(days=1)
    while d <= end_date + timedelta(days=1):
        dates.append(d.isoformat())
        d += timedelta(days=1)

    log.info(f"\nFetching NBA tip-off times for {len(dates)} dates...")
    tipoffs = get_tipoffs_for_dates(dates)
    if not tipoffs:
        log.error("No tip-off times retrieved. Check nba_api connectivity.")
        sys.exit(1)

    # ── Step 2: List markets ──────────────────────────────────────
    log.info("\nListing Kalshi game markets...")
    # +2 days: markets close the morning after the game
    market_end_dt = end_dt + timedelta(days=2)
    markets = list_game_markets(start_dt, market_end_dt, include_totals=not args.no_totals)

    # Filter already-collected tickers
    new_markets = [m for m in markets if m["ticker"] not in existing_tickers]
    log.info(f"{len(new_markets)} new markets to fetch "
             f"({len(markets) - len(new_markets)} already in dataset).")

    if args.dry_run:
        log.info(f"\n[DRY RUN] Would fetch candles for {len(new_markets)} markets.")
        log.info("Sample markets:")
        for m in new_markets[:8]:
            tipoff_ts = find_tipoff_for_event(m["event_ticker"], tipoffs)
            log.info(f"  [{m['market_type']:10}] {m['away_team']} @ {m['home_team']} "
                     f"| subject={m['subject_team']:5} line={m['line']:.1f} "
                     f"| tip-off={'found' if tipoff_ts else 'MISSING'}")
        return

    # ── Step 3: Fetch pre-game prices ────────────────────────────
    rows: List[dict] = []
    no_tipoff = no_candles = collected = 0

    new_markets.sort(key=lambda x: x["event_ticker"])
    total = len(new_markets)

    for i, market in enumerate(new_markets, 1):
        if i % 100 == 0 or i == 1:
            log.info(f"  [{i}/{total}] Fetching candles...")

        tipoff_ts = find_tipoff_for_event(market["event_ticker"], tipoffs)
        if tipoff_ts is None:
            no_tipoff += 1
            continue

        prices = get_pregame_price(market["ticker"], tipoff_ts)
        time.sleep(RATE_SLEEP)

        if prices is None:
            no_candles += 1
            continue

        rows.append({
            "ticker":        market["ticker"],
            "event_ticker":  market["event_ticker"],
            "market_type":   market["market_type"],
            "game_date":     market["game_date"],
            "away_team":     market["away_team"],
            "home_team":     market["home_team"],
            "subject_team":  market["subject_team"],
            "line":          market["line"],
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
    if not df_out.empty:
        log.info(f"  Date coverage: {df_out['game_date'].min()} → {df_out['game_date'].max()}")
        log.info(f"  Market types: {df_out['market_type'].value_counts().to_dict()}")
    log.info("  Next: run game_outcome_test.py --mode kalshi-compare")


if __name__ == "__main__":
    main()
