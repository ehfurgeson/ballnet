"""Stage H — weekly highlight boards from the Stage B spine (not Stage G)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import polars as pl

from ballnet.catalog.registry import stats_for_group
from ballnet.paths import HIGHLIGHTS_DIR, SPINE_DIR, ensure_data_dirs
from ballnet.publish import PUBLISHABLE_GROUPS
from ballnet.scoring import MIN_PEER_N, gaussian_tail_one_in_n, oriented_z_score

_JSON_DUMP_KW: dict[str, Any] = {"separators": (",", ":"), "ensure_ascii": False}

TOP_N = 25
PER_GROUP_N = 8

# OL / punter / returner: sparse or unusable weekly box columns for curated z-scores.
HIGHLIGHT_GROUPS: tuple[str, ...] = tuple(
    g for g in PUBLISHABLE_GROUPS if g not in ("ol", "punter")
)


@dataclass(frozen=True)
class HighlightStat:
    """One catalog id scored from a single spine week."""

    stat_id: str
    label: str
    higher_is_better: bool
    # Build a numeric column named `_value` (and optional `_volume` for floors).
    build: Callable[[pl.DataFrame], pl.DataFrame]
    volume_floor: float | None = None
    # Rare event counts (FF, INT) need a floor so a single fluke does not dominate.
    min_value: float | None = None


def _allowlist_for_group(group: str) -> list[HighlightStat]:
    """Curated single-game stats — volumes + clear rates; skip snap%/cushion noise."""

    def meta(stat_id: str) -> tuple[str, bool]:
        for s in stats_for_group(group):
            if s.id == stat_id:
                return s.id.replace("_", " "), s.higher_is_better
        return stat_id.replace("_", " "), True

    def simple(
        stat_id: str,
        spine_col: str,
        *,
        volume_col: str | None = None,
        floor: float | None = None,
        min_value: float | None = None,
    ) -> HighlightStat:
        label, hib = meta(stat_id)

        def build(df: pl.DataFrame) -> pl.DataFrame:
            out = df.with_columns(pl.col(spine_col).cast(pl.Float64).alias("_value"))
            if volume_col and volume_col in df.columns:
                out = out.with_columns(pl.col(volume_col).cast(pl.Float64).alias("_volume"))
            else:
                out = out.with_columns(pl.lit(None).cast(pl.Float64).alias("_volume"))
            return out

        return HighlightStat(stat_id, label, hib, build, floor, min_value)

    def ratio(
        stat_id: str,
        num: str,
        den: str,
        *,
        floor: float,
    ) -> HighlightStat:
        label, hib = meta(stat_id)

        def build(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(
                pl.when(pl.col(den).cast(pl.Float64) > 0)
                .then(pl.col(num).cast(pl.Float64) / pl.col(den).cast(pl.Float64))
                .otherwise(None)
                .alias("_value"),
                pl.col(den).cast(pl.Float64).alias("_volume"),
            )

        return HighlightStat(stat_id, label, hib, build, floor)

    def combined_tackles(stat_id: str = "tackles_combined") -> HighlightStat:
        label, hib = meta(stat_id)

        def build(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(
                (
                    pl.col("def_tackles_solo").fill_null(0).cast(pl.Float64)
                    + pl.col("def_tackle_assists").fill_null(0).cast(pl.Float64)
                ).alias("_value"),
                pl.lit(None).cast(pl.Float64).alias("_volume"),
            )

        return HighlightStat(stat_id, label, hib, build, None)

    def cpoe() -> HighlightStat:
        label, hib = meta("cpoe")

        def build(df: pl.DataFrame) -> pl.DataFrame:
            ngs = "ngs_pass_completion_percentage_above_expectation"
            exprs = []
            if ngs in df.columns:
                exprs.append(pl.col(ngs).cast(pl.Float64))
            exprs.append(pl.col("passing_cpoe").cast(pl.Float64))
            return df.with_columns(
                pl.coalesce(exprs).alias("_value"),
                pl.col("attempts").cast(pl.Float64).alias("_volume"),
            )

        return HighlightStat("cpoe", label, hib, build, 10.0)

    if group == "qb":
        return [
            simple("passing_yards", "passing_yards", volume_col="attempts", floor=10),
            simple("passing_tds", "passing_tds", volume_col="attempts", floor=10),
            simple("passing_epa", "passing_epa", volume_col="attempts", floor=10),
            cpoe(),
            ratio("completion_pct", "completions", "attempts", floor=15),
            simple("rushing_yards", "rushing_yards", volume_col="carries", floor=3),
        ]
    if group == "backfield":
        return [
            simple("rushing_yards", "rushing_yards", volume_col="carries", floor=8),
            simple("rushing_tds", "rushing_tds", volume_col="carries", floor=5),
            simple("rushing_epa", "rushing_epa", volume_col="carries", floor=8),
            ratio("yards_per_carry", "rushing_yards", "carries", floor=8),
            simple("receptions", "receptions", volume_col="targets", floor=2),
            simple("receiving_yards", "receiving_yards", volume_col="targets", floor=2),
        ]
    if group == "pass_catcher":
        return [
            simple("receiving_yards", "receiving_yards", volume_col="targets", floor=3),
            simple("receptions", "receptions", volume_col="targets", floor=3),
            simple("receiving_tds", "receiving_tds", volume_col="targets", floor=2),
            simple("receiving_epa", "receiving_epa", volume_col="targets", floor=3),
            simple("targets", "targets", volume_col="targets", floor=4),
        ]
    if group == "def_front":
        return [
            combined_tackles(),
            simple("sacks", "def_sacks", min_value=1.5),
            simple("tackles_for_loss", "def_tackles_for_loss", min_value=2),
            simple("qb_hits", "def_qb_hits", min_value=3),
            simple("forced_fumbles", "def_fumbles_forced", min_value=2),
            simple("interceptions", "def_interceptions", min_value=2),
        ]
    if group == "secondary":
        return [
            simple("interceptions", "def_interceptions", min_value=2),
            simple("passes_defended", "def_pass_defended", min_value=2),
            combined_tackles(),
            simple("forced_fumbles", "def_fumbles_forced", min_value=2),
        ]
    if group == "kicker":
        return [
            simple("fg_made", "fg_made", volume_col="fg_att", floor=1),
            ratio("fg_pct", "fg_made", "fg_att", floor=2),
            simple("fg_long", "fg_long", volume_col="fg_att", floor=1),
        ]
    return []


def _score_stat(
    week_df: pl.DataFrame,
    group: str,
    spec: HighlightStat,
) -> list[dict[str, Any]]:
    if week_df.is_empty():
        return []
    # Drop rows missing required spine columns used by build.
    built = spec.build(week_df)
    built = built.filter(pl.col("_value").is_not_null() & pl.col("_value").is_finite())
    if spec.volume_floor is not None:
        built = built.filter(
            pl.col("_volume").is_not_null() & (pl.col("_volume") >= spec.volume_floor)
        )
    if built.height < MIN_PEER_N:
        return []

    peers = built["_value"].to_list()
    rows: list[dict[str, Any]] = []
    for rec in built.iter_rows(named=True):
        val = float(rec["_value"])
        if spec.min_value is not None and val < spec.min_value:
            continue
        z = oriented_z_score(val, peers, higher_is_better=spec.higher_is_better)
        if z is None:
            continue
        rows.append(
            {
                "playerId": rec["player_id"],
                "name": rec.get("player_display_name") or "",
                "position": rec.get("position_code") or rec.get("position") or "",
                "team": rec.get("team") or "",
                "opponent": rec.get("opponent_team") or "",
                "positionGroup": group,
                "statId": spec.stat_id,
                "statLabel": spec.label,
                "value": val,
                "zScore": round(z, 3),
                "peerN": len(peers),
                "oneInN": gaussian_tail_one_in_n(z),
            }
        )
    return rows


def build_highlights_board(season: int, week: int) -> dict[str, Any]:
    """Compute weekly board payload from spine (does not write)."""
    path = SPINE_DIR / f"player_week_{season}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing spine {path}")

    spine = pl.read_parquet(path)
    week_df = spine.filter(
        (pl.col("week") == week) & pl.col("position_group").is_not_null()
    )

    all_rows: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = {g: [] for g in HIGHLIGHT_GROUPS}

    for group in HIGHLIGHT_GROUPS:
        gdf = week_df.filter(pl.col("position_group") == group)
        for spec in _allowlist_for_group(group):
            # Fix higher_is_better / label from catalog for this group.
            for s in stats_for_group(group):
                if s.id == spec.stat_id:
                    spec = HighlightStat(
                        spec.stat_id,
                        s.id.replace("_", " "),
                        s.higher_is_better,
                        spec.build,
                        spec.volume_floor,
                        spec.min_value,
                    )
                    break
            scored = _score_stat(gdf, group, spec)
            all_rows.extend(scored)

    all_rows.sort(key=lambda r: r["zScore"], reverse=True)

    # Cap per-group contribution so sparse defensive events cannot crowd out offense.
    balanced: list[dict[str, Any]] = []
    for group in HIGHLIGHT_GROUPS:
        group_rows = [r for r in all_rows if r["positionGroup"] == group][:PER_GROUP_N]
        balanced.extend(group_rows)
    balanced.sort(key=lambda r: r["zScore"], reverse=True)

    top: list[dict[str, Any]] = []
    for i, row in enumerate(balanced[:TOP_N], start=1):
        entry = dict(row)
        entry["rank"] = i
        top.append(entry)

    for group in HIGHLIGHT_GROUPS:
        group_rows = [r for r in all_rows if r["positionGroup"] == group][:PER_GROUP_N]
        for i, row in enumerate(group_rows, start=1):
            entry = dict(row)
            entry["rank"] = i
            by_group[group].append(entry)

    return {
        "schemaVersion": 1,
        "season": season,
        "week": week,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "top": top,
        "byGroup": by_group,
    }


def publish_highlights(season: int, week: int) -> Path:
    """Write `data/highlights/{season}/w{week}.json`."""
    ensure_data_dirs()
    payload = build_highlights_board(season, week)
    out = HIGHLIGHTS_DIR / str(season) / f"w{week}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, **_JSON_DUMP_KW)
    return out
