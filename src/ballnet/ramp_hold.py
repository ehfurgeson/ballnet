"""Locked ramp–hold: min_n = n_base * min(ramp_week, 5).

as_of_week is the latest REG week included in YTD (partial slates allowed).
ramp_week / completed_week is the last fully scored REG week, floored at 1 and
capped at as_of_week — so Thursday of week N does not double the sample bar.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ballnet.paths import RAW_DIR, YTD_DIR


def min_n(n_base: int | float, ramp_week: int) -> float:
    """ramp_week is completed REG weeks, not weeks since debut and not as_of_week."""
    if ramp_week < 1:
        raise ValueError(f"ramp_week must be >= 1, got {ramp_week}")
    return float(n_base) * min(ramp_week, 5)


def ramp_week(as_of_week: int, completed_week: int) -> int:
    """Clamp last fully completed week into the live YTD slice."""
    if as_of_week < 1:
        raise ValueError(f"as_of_week must be >= 1, got {as_of_week}")
    if completed_week < 1:
        return 1
    return min(int(completed_week), int(as_of_week))


def last_completed_reg_week_from_frame(sched: pl.DataFrame) -> int:
    """Highest consecutive REG week where every scheduled game has scores.

    Returns 0 when no week is fully scored (week-1 Thursday).
    """
    df = sched
    if "game_type" in df.columns:
        df = df.filter(pl.col("game_type") == "REG")
    elif "season_type" in df.columns:
        df = df.filter(pl.col("season_type") == "REG")
    if df.height == 0 or "week" not in df.columns:
        return 0
    if "home_score" not in df.columns or "away_score" not in df.columns:
        return 0

    scored = pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    by_week = (
        df.group_by(pl.col("week").cast(pl.Int32))
        .agg(pl.len().alias("n"), scored.sum().alias("n_final"))
        .sort("week")
    )
    completed = 0
    for row in by_week.iter_rows(named=True):
        week = int(row["week"])
        if week < 1:
            continue
        if week != completed + 1:
            break
        n = int(row["n"])
        n_final = int(row["n_final"] or 0)
        if n > 0 and n_final >= n:
            completed = week
        else:
            break
    return completed


def last_completed_reg_week(season: int) -> int | None:
    """None when the cached schedule parquet is missing (caller falls back)."""
    path = RAW_DIR / f"schedules_{season}.parquet"
    if not path.exists():
        return None
    return last_completed_reg_week_from_frame(pl.read_parquet(path))


def completed_week_from_schedule(season: int, as_of_week: int) -> int:
    """Stage C source of truth: schedule finals, not a prior YTD parquet."""
    last = last_completed_reg_week(season)
    if last is None:
        return as_of_week
    return ramp_week(as_of_week, last)


def completed_week_from_ytd(season: int, as_of_week: int) -> int | None:
    """Read the week Stage C actually used, so skip-pipeline matches qualification."""
    for group in ("qb", "backfield", "kicker"):
        for suffix in ("_pct", ""):
            path = YTD_DIR / f"ytd_{group}_{season}_w{as_of_week}{suffix}.parquet"
            week = _completed_week_column(path)
            if week is not None:
                return week
    return None


def completed_week_for(season: int, as_of_week: int) -> int:
    """Publish-time completed week: parquet first, else live schedule."""
    from_ytd = completed_week_from_ytd(season, as_of_week)
    if from_ytd is not None:
        return from_ytd
    return completed_week_from_schedule(season, as_of_week)


def _completed_week_column(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        df = pl.read_parquet(path, columns=["completed_week"])
    except Exception:
        return None
    if df.height == 0:
        return None
    val = df.get_column("completed_week")[0]
    if val is None:
        return None
    return int(val)
