"""Stage A — fetch nflverse sources and cache as parquet."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import nflreadpy as nfl
import polars as pl

from ballnet.paths import RAW_DIR, ensure_data_dirs

# Minimal schemas so Stage B can left-join when a source starts later than the season.
_EMPTY: dict[str, pl.DataFrame] = {
    "ngs_passing": pl.DataFrame(
        schema={
            "player_gsis_id": pl.Utf8,
            "season": pl.Int32,
            "week": pl.Int32,
            "attempts": pl.Float64,
        }
    ),
    "ngs_receiving": pl.DataFrame(
        schema={
            "player_gsis_id": pl.Utf8,
            "season": pl.Int32,
            "week": pl.Int32,
            "avg_yac": pl.Float64,
        }
    ),
    "ngs_rushing": pl.DataFrame(
        schema={
            "player_gsis_id": pl.Utf8,
            "season": pl.Int32,
            "week": pl.Int32,
            "efficiency": pl.Float64,
        }
    ),
    "pfr_pass": pl.DataFrame(
        schema={
            "pfr_player_id": pl.Utf8,
            "season": pl.Int32,
            "week": pl.Int32,
            "times_pressured": pl.Float64,
        }
    ),
    "pfr_rush": pl.DataFrame(
        schema={
            "pfr_player_id": pl.Utf8,
            "season": pl.Int32,
            "week": pl.Int32,
            "rushing_broken_tackles": pl.Float64,
        }
    ),
    "pfr_rec": pl.DataFrame(
        schema={
            "pfr_player_id": pl.Utf8,
            "season": pl.Int32,
            "week": pl.Int32,
            "receiving_drop": pl.Float64,
        }
    ),
    "pfr_def": pl.DataFrame(
        schema={
            "pfr_player_id": pl.Utf8,
            "season": pl.Int32,
            "week": pl.Int32,
            "def_targets": pl.Float64,
        }
    ),
    "ftn": pl.DataFrame(schema={"season": pl.Int32, "week": pl.Int32}),
    "participation": pl.DataFrame(schema={"season": pl.Int32, "week": pl.Int32}),
    "ff_opportunity": pl.DataFrame(
        schema={
            "player_id": pl.Utf8,
            "season": pl.Int32,
            "week": pl.Int32,
            "rec_yards_gained_exp": pl.Float64,
        }
    ),
    "snaps": pl.DataFrame(
        schema={
            "pfr_player_id": pl.Utf8,
            "season": pl.Int32,
            "week": pl.Int32,
            "offense_snaps": pl.Float64,
        }
    ),
}


@dataclass(frozen=True)
class FetchResult:
    name: str
    path: str
    rows: int
    cols: int
    seconds: float
    notes: str = ""


def _write(name: str, df: pl.DataFrame) -> Path:
    ensure_data_dirs()
    path = RAW_DIR / f"{name}.parquet"
    df.write_parquet(path)
    return path


def _filter_reg(df: pl.DataFrame) -> pl.DataFrame:
    if "season_type" in df.columns:
        return df.filter(pl.col("season_type") == "REG")
    if "game_type" in df.columns:
        return df.filter(pl.col("game_type") == "REG")
    return df


def _drop_ngs_week0(df: pl.DataFrame) -> pl.DataFrame:
    if "week" not in df.columns:
        return df
    return df.filter(pl.col("week") > 0)


def fetch_season(season: int, *, force: bool = False) -> list[FetchResult]:
    """Download every source family needed for the Knowball viz spine."""
    ensure_data_dirs()
    results: list[FetchResult] = []

    def run(
        name: str,
        loader: Callable[[], pl.DataFrame],
        notes: str = "",
        *,
        optional: bool = False,
        empty_key: str | None = None,
    ) -> pl.DataFrame:
        out = RAW_DIR / f"{name}.parquet"
        if out.exists() and not force:
            df = pl.read_parquet(out)
            results.append(
                FetchResult(name, str(out), df.height, len(df.columns), 0.0, notes + " (cache)")
            )
            return df
        t0 = time.perf_counter()
        try:
            df = loader()
            note = notes
        except Exception as e:
            if not optional:
                raise
            df = _EMPTY[empty_key] if empty_key else pl.DataFrame()
            note = f"{notes} UNAVAILABLE: {type(e).__name__}: {e}"
        elapsed = time.perf_counter() - t0
        path = _write(name, df)
        results.append(FetchResult(name, str(path), df.height, len(df.columns), elapsed, note))
        return df

    run(
        f"player_stats_{season}",
        lambda: _filter_reg(nfl.load_player_stats(seasons=[season], summary_level="week")),
        "box weekly REG",
    )

    for st, empty_key in (
        ("passing", "ngs_passing"),
        ("receiving", "ngs_receiving"),
        ("rushing", "ngs_rushing"),
    ):
        run(
            f"ngs_{st}_{season}",
            lambda st=st: _drop_ngs_week0(
                _filter_reg(nfl.load_nextgen_stats(seasons=[season], stat_type=st))
            ),
            f"NGS {st}",
            optional=True,
            empty_key=empty_key,
        )

    run(
        f"snap_counts_{season}",
        lambda: _filter_reg(nfl.load_snap_counts(seasons=[season])),
        "snap_counts",
        optional=True,
        empty_key="snaps",
    )

    for st, empty_key in (
        ("pass", "pfr_pass"),
        ("rush", "pfr_rush"),
        ("rec", "pfr_rec"),
        ("def", "pfr_def"),
    ):
        run(
            f"pfr_{st}_{season}",
            lambda st=st: _filter_reg(
                nfl.load_pfr_advstats(seasons=[season], stat_type=st, summary_level="week")
            ),
            f"pfr_advstats {st}",
            optional=True,
            empty_key=empty_key,
        )

    run(
        f"ff_opportunity_{season}",
        lambda: nfl.load_ff_opportunity(seasons=[season], stat_type="weekly"),
        "ff_opportunity weekly",
        optional=True,
        empty_key="ff_opportunity",
    )

    run("ff_playerids", lambda: nfl.load_ff_playerids(), "gsis↔pfr map")

    run(
        f"rosters_{season}",
        lambda: nfl.load_rosters(seasons=[season]),
        "rosters",
    )

    run(
        f"schedules_{season}",
        lambda: nfl.load_schedules(seasons=[season]),
        "schedules",
    )

    run(
        f"ftn_charting_{season}",
        lambda: nfl.load_ftn_charting(seasons=[season]),
        "play-grain FTN",
        optional=True,
        empty_key="ftn",
    )
    run(
        f"participation_{season}",
        lambda: nfl.load_participation(seasons=[season]),
        "play-grain participation",
        optional=True,
        empty_key="participation",
    )

    run(
        f"pbp_{season}",
        lambda: _filter_reg(nfl.load_pbp(seasons=[season])),
        "play-grain PBP REG",
        optional=True,
    )

    return results
