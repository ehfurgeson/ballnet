"""Stage B — canonical player-week spine (left joins; preserve nulls)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import polars as pl

from ballnet.paths import RAW_DIR, SPINE_DIR, ensure_data_dirs
from ballnet.positions import map_position


@dataclass(frozen=True)
class SpineResult:
    path: str
    rows: int
    cols: int
    seconds: float
    join_coverage: dict[str, float]


def _prefix(df: pl.DataFrame, prefix: str, keep: set[str]) -> pl.DataFrame:
    renames = {c: f"{prefix}{c}" for c in df.columns if c not in keep}
    return df.rename(renames)


def _id_map() -> pl.DataFrame:
    ids = pl.read_parquet(RAW_DIR / "ff_playerids.parquet")
    return (
        ids.select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("pfr_id").alias("pfr_player_id"),
        )
        .filter(pl.col("player_id").is_not_null() & pl.col("pfr_player_id").is_not_null())
        .unique(subset=["player_id"], keep="first")
    )


def _as_i32(df: pl.DataFrame, *cols: str) -> pl.DataFrame:
    casts = [pl.col(c).cast(pl.Int32, strict=False) for c in cols if c in df.columns]
    return df.with_columns(casts) if casts else df


def _dedupe_keys(df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    present = [k for k in keys if k in df.columns]
    if len(present) != len(keys) or df.height == 0:
        return df
    return df.unique(subset=present, keep="first")


def _safe_join(
    spine: pl.DataFrame,
    right: pl.DataFrame,
    on: list[str],
    label: str,
) -> pl.DataFrame:
    missing = [c for c in on if c not in right.columns]
    if missing:
        return spine
    right = _dedupe_keys(right, on)
    before = spine.height
    out = spine.join(right, on=on, how="left")
    assert out.height == before, f"{label} join exploded rows"
    return out


def build_spine(season: int) -> SpineResult:
    """One row per (player_id, season, week) with box + NGS + snaps + PFR + opportunity."""
    ensure_data_dirs()
    t0 = time.perf_counter()

    box = pl.read_parquet(RAW_DIR / f"player_stats_{season}.parquet")
    # Normalize interception / sack names used in Knowball catalog notes
    if "passing_interceptions" in box.columns and "interceptions" not in box.columns:
        box = box.rename({"passing_interceptions": "interceptions"})
    if "sacks_suffered" in box.columns and "sacks_taken" not in box.columns:
        box = box.rename({"sacks_suffered": "sacks_taken"})

    box = _as_i32(box, "season", "week")

    codes = [
        map_position(p) for p in box.get_column("position").to_list()
    ]
    box = box.with_columns(
        pl.Series("position_code", [c[0] for c in codes]),
        pl.Series("position_group", [c[1] for c in codes]),
    )

    id_map = _id_map()
    spine = box.join(id_map, on="player_id", how="left")

    # NGS joins on GSIS + season + week
    for st, pref in (("passing", "ngs_pass_"), ("receiving", "ngs_rec_"), ("rushing", "ngs_rush_")):
        ngs = pl.read_parquet(RAW_DIR / f"ngs_{st}_{season}.parquet")
        ngs = _as_i32(ngs, "season", "week")
        keep = {"player_gsis_id", "season", "week"}
        if "player_gsis_id" not in ngs.columns:
            continue
        ngs = _prefix(ngs, pref, keep).rename({"player_gsis_id": "player_id"})
        spine = _safe_join(spine, ngs, ["player_id", "season", "week"], f"NGS {st}")

    # Snaps on PFR id + season + week
    snaps = pl.read_parquet(RAW_DIR / f"snap_counts_{season}.parquet")
    snaps = _as_i32(snaps, "season", "week")
    snap_keep = {"pfr_player_id", "season", "week"}
    if "pfr_player_id" in snaps.columns:
        snaps = _prefix(snaps, "snap_", snap_keep)
        spine = _safe_join(spine, snaps, ["pfr_player_id", "season", "week"], "snap")

    # PFR advanced
    for st, pref in (("pass", "pfr_pass_"), ("rush", "pfr_rush_"), ("rec", "pfr_rec_"), ("def", "pfr_def_")):
        pfr = pl.read_parquet(RAW_DIR / f"pfr_{st}_{season}.parquet")
        pfr = _as_i32(pfr, "season", "week")
        keep = {"pfr_player_id", "season", "week"}
        if "pfr_player_id" not in pfr.columns:
            continue
        pfr = _prefix(pfr, pref, keep)
        spine = _safe_join(spine, pfr, ["pfr_player_id", "season", "week"], f"pfr {st}")

    # Fantasy opportunity
    opp = pl.read_parquet(RAW_DIR / f"ff_opportunity_{season}.parquet")
    if "season_type" in opp.columns:
        opp = opp.filter(pl.col("season_type") == "REG")
    opp = _as_i32(opp, "season", "week")
    keep = {"player_id", "season", "week"}
    if "player_id" in opp.columns:
        opp = _prefix(opp, "ffopp_", keep)
        spine = _safe_join(spine, opp, ["player_id", "season", "week"], "ff_opportunity")

    # Join coverage: share of spine rows with non-null key marker from each family
    coverage = {
        "ngs_pass": _cov(spine, "ngs_pass_attempts"),
        "ngs_rec": _cov(spine, "ngs_rec_avg_yac"),
        "ngs_rush": _cov(spine, "ngs_rush_efficiency"),
        "snaps": _cov(spine, "snap_offense_snaps"),
        "pfr_pass": _cov(spine, "pfr_pass_times_pressured"),
        "pfr_rush": _cov(spine, "pfr_rush_rushing_broken_tackles"),
        "pfr_rec": _cov(spine, "pfr_rec_receiving_drop"),
        "pfr_def": _cov(spine, "pfr_def_def_targets"),
        "ff_opportunity": _cov(spine, "ffopp_rec_yards_gained_exp"),
        "pfr_id_mapped": float(spine.filter(pl.col("pfr_player_id").is_not_null()).height)
        / max(spine.height, 1),
        "position_mapped": float(spine.filter(pl.col("position_code").is_not_null()).height)
        / max(spine.height, 1),
    }

    out = SPINE_DIR / f"player_week_{season}.parquet"
    spine.write_parquet(out)
    elapsed = time.perf_counter() - t0
    return SpineResult(str(out), spine.height, len(spine.columns), elapsed, coverage)


def _cov(df: pl.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(df.filter(pl.col(col).is_not_null()).height) / max(df.height, 1)
