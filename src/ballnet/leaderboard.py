"""Publish Knowball search leaderboards from Stage E percentile panels."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from ballnet.catalog.registry import POSITION_GROUPS, stats_for_group
from ballnet.paths import LEADERBOARDS_DIR, YTD_DIR, ensure_data_dirs
from ballnet.percentiles import attach_percentiles

# Mirror publish.PUBLISHABLE_GROUPS without importing publish (cycle risk).
_DEFAULT_GROUPS: tuple[str, ...] = tuple(g for g in POSITION_GROUPS if g != "returner")

_JSON_DUMP_KW: dict[str, Any] = {"separators": (",", ":"), "ensure_ascii": False}


@dataclass(frozen=True)
class LeaderboardPublishResult:
    path: str
    position_group: str
    stats: int
    players: int
    seconds: float


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, **_JSON_DUMP_KW)


def _row_sort_key(row: dict[str, Any]) -> tuple[int, float, str]:
    """Qualified + high oriented percentile first; nulls last."""
    pct = row.get("percentile")
    if pct is None:
        return (1, 0.0, row["playerId"])
    return (0, -float(pct), row["playerId"])


def build_leaderboard_payload(
    season: int,
    as_of_week: int,
    position_group: str,
    *,
    long: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Build one group leaderboard JSON from Stage E long panel."""
    if position_group not in POSITION_GROUPS:
        raise NotImplementedError(f"leaderboard unknown group {position_group}")

    if long is None:
        pct_path = YTD_DIR / f"ytd_{position_group}_{season}_w{as_of_week}_pct.parquet"
        if not pct_path.exists():
            attach_percentiles(season, as_of_week, position_group=position_group)
        long = pl.read_parquet(pct_path)

    catalog = [
        s for s in stats_for_group(position_group) if not s.always_unavailable
    ]
    catalog_ids = {s.id for s in catalog}
    stats_out: dict[str, list[dict[str, Any]]] = {s.id: [] for s in catalog}

    if long.height > 0:
        for rec in long.to_dicts():
            sid = rec.get("stat_id")
            if sid not in catalog_ids:
                continue
            value = rec.get("player_value")
            pct = rec.get("percentile")
            denom = rec.get("denom_ytd")
            stats_out[sid].append(
                {
                    "playerId": rec["player_id"],
                    "name": rec.get("player_display_name") or "",
                    "position": rec.get("position_code") or "",
                    "team": rec.get("team") or "",
                    "value": float(value) if value is not None else None,
                    "percentile": float(pct) if pct is not None else None,
                    "denomYtd": float(denom) if denom is not None else None,
                    "qualified": bool(rec.get("qualified")),
                }
            )

    for sid, rows in stats_out.items():
        rows.sort(key=_row_sort_key)

    return {
        "schemaVersion": 1,
        "season": season,
        "asOfWeek": as_of_week,
        "positionGroup": position_group,
        "stats": stats_out,
    }


def publish_leaderboard_group(
    season: int,
    as_of_week: int,
    position_group: str,
) -> LeaderboardPublishResult:
    """Write `leaderboards/{season}/w{week}/{group}.json`."""
    ensure_data_dirs()
    t0 = time.perf_counter()
    payload = build_leaderboard_payload(season, as_of_week, position_group)
    out = LEADERBOARDS_DIR / str(season) / f"w{as_of_week}" / f"{position_group}.json"
    _write_json(out, payload)
    player_ids = {
        row["playerId"]
        for rows in payload["stats"].values()
        for row in rows
    }
    return LeaderboardPublishResult(
        path=str(out),
        position_group=position_group,
        stats=len(payload["stats"]),
        players=len(player_ids),
        seconds=time.perf_counter() - t0,
    )


def publish_leaderboards(
    season: int,
    as_of_week: int,
    *,
    groups: Iterable[str] | None = None,
) -> list[LeaderboardPublishResult]:
    """Publish leaderboard JSON for every selected group in a season slice."""
    selected = list(groups) if groups is not None else list(_DEFAULT_GROUPS)
    return [
        publish_leaderboard_group(season, as_of_week, group) for group in selected
    ]
