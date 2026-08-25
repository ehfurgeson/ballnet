"""Stage G — publish local PlayerPageJson + league shapes + search index."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from ballnet.catalog.registry import POSITION_GROUPS, stats_for_group
from ballnet.catalog.types import StatDefinition
from ballnet.density import load_densities
from ballnet.paths import INDEX_DIR, LEAGUE_DIR, PAGES_DIR, YTD_DIR, ensure_data_dirs
from ballnet.percentiles import attach_percentiles

# Compact JSON — player pages are scalars only; curves live under data/league/.
_JSON_DUMP_KW: dict[str, Any] = {"separators": (",", ":"), "ensure_ascii": False}

PUBLISHABLE_GROUPS: tuple[str, ...] = tuple(
    g for g in POSITION_GROUPS if g != "returner"
)

# NFL REG week count: 17 through 2020, 18 from 2021.
_REG_WEEKS_18_FROM = 2021


def default_as_of_week(season: int) -> int:
    """Final REG week for season-end YTD publish."""
    return 18 if season >= _REG_WEEKS_18_FROM else 17


@dataclass(frozen=True)
class PublishResult:
    path: str
    current_path: str | None
    stats: int
    seconds: float


@dataclass(frozen=True)
class GroupPublishResult:
    position_group: str
    players: int
    paths: int
    seconds: float


@dataclass(frozen=True)
class BatchPublishResult:
    season: int
    as_of_week: int
    groups: list[GroupPublishResult]
    index_path: str | None
    current_index_path: str | None
    players: int
    bios: list[dict[str, Any]]
    seconds: float


@dataclass(frozen=True)
class RangePublishResult:
    start: int
    end: int
    seasons: list[BatchPublishResult]
    index_path: str
    current_index_path: str
    seasons_index_path: str
    players: int
    seconds: float


def _snapshot_for_row(
    row: dict[str, Any],
    *,
    stat: StatDefinition,
) -> dict[str, Any] | None:
    """Build one scalar JsonStatSnapshot. League curve lives in data/league/."""
    if stat.always_unavailable:
        return None

    snap: dict[str, Any] = {
        "id": row["stat_id"],
        "qualified": bool(row["qualified"]),
        "kind": row["kind"],
    }

    value = row.get("player_value")
    snap["playerValue"] = float(value) if value is not None else None

    pct = row.get("percentile")
    snap["percentile"] = float(pct) if pct is not None else None

    if row.get("denom_ytd") is not None:
        snap["denomYtd"] = float(row["denom_ytd"])

    reason = row.get("unavailable_reason")
    if reason:
        snap["unavailableReason"] = reason

    return snap


def _page_dict(
    *,
    season: int,
    as_of_week: int,
    player_id: str,
    meta: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    catalog_stats: list[StatDefinition],
    seasons: list[int] | None = None,
) -> dict[str, Any]:
    stats_out: list[dict[str, Any]] = []
    for stat in catalog_stats:
        if stat.always_unavailable:
            continue
        row = by_id.get(stat.id)
        if row is None:
            continue
        snap = _snapshot_for_row(row, stat=stat)
        if snap is not None:
            stats_out.append(snap)

    return {
        "schemaVersion": 1,
        "player": {
            "id": player_id,
            "name": meta["player_display_name"],
            "position": meta["position_code"],
            "team": meta["team"],
            "seasons": seasons if seasons is not None else [season],
        },
        "season": season,
        "asOfWeek": as_of_week,
        "stats": stats_out,
    }


def _league_stat_payload(dens: dict[str, Any]) -> dict[str, Any]:
    """Stage D snake_case → Knowball camelCase KDE shape (no player overlay)."""
    out: dict[str, Any] = {
        "kind": dens["kind"],
        "xMin": dens["x_min"],
        "xMax": dens["x_max"],
        "yMax": dens["y_max"],
        "curve": dens.get("curve") or [],
    }
    if dens.get("n_sample") is not None:
        out["nSample"] = int(dens["n_sample"])
    if dens.get("lower_bound") is not None:
        out["lowerBound"] = dens["lower_bound"]
    if dens.get("upper_bound") is not None:
        out["upperBound"] = dens["upper_bound"]
    return out


def publish_league_group(
    season: int,
    as_of_week: int,
    position_group: str,
) -> Path:
    """Write `league/{season}/w{week}/{group}.json` from Stage D KDE curves."""
    if position_group not in POSITION_GROUPS:
        raise NotImplementedError(f"league publish unknown group {position_group}")
    ensure_data_dirs()
    dens_by_id = load_densities(season, as_of_week, position_group=position_group)
    stats: dict[str, Any] = {
        sid: _league_stat_payload(dens) for sid, dens in dens_by_id.items()
    }
    payload = {
        "schemaVersion": 1,
        "season": season,
        "asOfWeek": as_of_week,
        "positionGroup": position_group,
        "stats": stats,
    }
    out = LEAGUE_DIR / str(season) / f"w{as_of_week}" / f"{position_group}.json"
    _write_json(out, payload)
    return out


def publish_league_slice(
    season: int,
    as_of_week: int,
    *,
    groups: Iterable[str] | None = None,
) -> list[Path]:
    """Publish Knowball league JSON for every group in a season slice."""
    selected = list(groups) if groups is not None else list(PUBLISHABLE_GROUPS)
    return [publish_league_group(season, as_of_week, g) for g in selected]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, **_JSON_DUMP_KW)


def publish_player_page(
    season: int,
    as_of_week: int,
    player_id: str,
    *,
    position_group: str = "qb",
    also_current: bool = True,
) -> PublishResult:
    """Write `pages/{season}/w{week}/{player_id}.json` from Stage C/E (scalars only)."""
    if position_group not in POSITION_GROUPS:
        raise NotImplementedError(f"publish unknown group {position_group}")
    ensure_data_dirs()
    t0 = time.perf_counter()

    pct_path = YTD_DIR / f"ytd_{position_group}_{season}_w{as_of_week}_pct.parquet"
    if not pct_path.exists():
        attach_percentiles(season, as_of_week, position_group=position_group)
    long = pl.read_parquet(pct_path)
    player_rows = long.filter(pl.col("player_id") == player_id)
    if player_rows.height == 0:
        raise ValueError(f"player {player_id} not in YTD panel for {season} w{as_of_week}")

    meta = player_rows.row(0, named=True)
    by_id = {r["stat_id"]: r for r in player_rows.to_dicts()}
    page = _page_dict(
        season=season,
        as_of_week=as_of_week,
        player_id=player_id,
        meta=meta,
        by_id=by_id,
        catalog_stats=stats_for_group(position_group),
    )

    out = PAGES_DIR / str(season) / f"w{as_of_week}" / f"{player_id}.json"
    _write_json(out, page)

    current_path: str | None = None
    if also_current:
        cur = PAGES_DIR / "current" / f"{player_id}.json"
        _write_json(cur, page)
        current_path = str(cur)

    return PublishResult(
        path=str(out),
        current_path=current_path,
        stats=len(page["stats"]),
        seconds=time.perf_counter() - t0,
    )


def publish_group(
    season: int,
    as_of_week: int,
    position_group: str,
    *,
    also_current: bool = True,
) -> tuple[GroupPublishResult, list[dict[str, Any]]]:
    """Publish every player in a group's Stage E panel. Returns bios for the index."""
    if position_group not in POSITION_GROUPS:
        raise NotImplementedError(f"publish unknown group {position_group}")
    ensure_data_dirs()
    t0 = time.perf_counter()

    pct_path = YTD_DIR / f"ytd_{position_group}_{season}_w{as_of_week}_pct.parquet"
    if not pct_path.exists():
        attach_percentiles(season, as_of_week, position_group=position_group)
    long = pl.read_parquet(pct_path)
    if long.height == 0:
        return (
            GroupPublishResult(
                position_group=position_group,
                players=0,
                paths=0,
                seconds=time.perf_counter() - t0,
            ),
            [],
        )

    catalog_stats = stats_for_group(position_group)
    week_dir = PAGES_DIR / str(season) / f"w{as_of_week}"
    week_dir.mkdir(parents=True, exist_ok=True)
    if also_current:
        (PAGES_DIR / "current").mkdir(parents=True, exist_ok=True)

    bios: list[dict[str, Any]] = []
    paths = 0
    # One partition pass — avoid re-filtering the long panel per player.
    for player_id, pdf in long.group_by("player_id", maintain_order=True):
        pid = player_id[0] if isinstance(player_id, tuple) else player_id
        rows = pdf.to_dicts()
        meta = rows[0]
        by_id = {r["stat_id"]: r for r in rows}
        page = _page_dict(
            season=season,
            as_of_week=as_of_week,
            player_id=pid,
            meta=meta,
            by_id=by_id,
            catalog_stats=catalog_stats,
        )
        _write_json(week_dir / f"{pid}.json", page)
        paths += 1
        if also_current:
            _write_json(PAGES_DIR / "current" / f"{pid}.json", page)
            paths += 1
        bios.append(
            {
                "id": pid,
                "name": meta["player_display_name"],
                "position": meta["position_code"],
                "team": meta["team"],
                "seasons": [season],
            }
        )

    return (
        GroupPublishResult(
            position_group=position_group,
            players=len(bios),
            paths=paths,
            seconds=time.perf_counter() - t0,
        ),
        bios,
    )


def _merge_bios(
    into: dict[str, dict[str, Any]],
    bios: Iterable[dict[str, Any]],
) -> None:
    """Union seasons; latest bio wins for name/position/team."""
    for p in bios:
        pid = p["id"]
        seasons = set(p.get("seasons") or [])
        if pid in into:
            seasons |= set(into[pid].get("seasons") or [])
        into[pid] = {
            "id": pid,
            "name": p["name"],
            "position": p["position"],
            "team": p["team"],
            "seasons": sorted(seasons),
        }


def write_players_index(
    players: Iterable[dict[str, Any]],
    *,
    merge_existing: bool = False,
) -> Path:
    """Write `index/players.json` with schemaVersion envelope."""
    ensure_data_dirs()
    by_id: dict[str, dict[str, Any]] = {}
    if merge_existing:
        existing = INDEX_DIR / "players.json"
        if existing.exists():
            prev = json.loads(existing.read_text(encoding="utf-8"))
            _merge_bios(by_id, prev.get("players") or [])
    _merge_bios(by_id, players)
    ordered = sorted(by_id.values(), key=lambda p: (p["name"].lower(), p["id"]))
    out = INDEX_DIR / "players.json"
    _write_json(out, {"schemaVersion": 1, "players": ordered})
    return out


def write_current_index(season: int, as_of_week: int) -> Path:
    """Write `index/current.json` mirroring meta_current_week."""
    ensure_data_dirs()
    out = INDEX_DIR / "current.json"
    _write_json(
        out,
        {"schemaVersion": 1, "season": season, "asOfWeek": as_of_week},
    )
    return out


def write_seasons_index(slices: Iterable[dict[str, int]]) -> Path:
    """Write `index/seasons.json` — published (season, asOfWeek) pairs."""
    ensure_data_dirs()
    ordered = sorted(
        ({"season": int(s["season"]), "asOfWeek": int(s["asOfWeek"])} for s in slices),
        key=lambda s: s["season"],
    )
    out = INDEX_DIR / "seasons.json"
    _write_json(out, {"schemaVersion": 1, "seasons": ordered})
    return out


def publish_all(
    season: int,
    as_of_week: int,
    *,
    groups: Iterable[str] | None = None,
    also_current: bool = True,
    write_index: bool = True,
    merge_index: bool = False,
) -> BatchPublishResult:
    """Batch Stage G for every player in each group + league shapes + optional index."""
    ensure_data_dirs()
    t0 = time.perf_counter()
    selected = list(groups) if groups is not None else list(PUBLISHABLE_GROUPS)
    group_results: list[GroupPublishResult] = []
    all_bios: list[dict[str, Any]] = []

    publish_league_slice(season, as_of_week, groups=selected)

    for group in selected:
        result, bios = publish_group(
            season, as_of_week, group, also_current=also_current
        )
        group_results.append(result)
        all_bios.extend(bios)

    index_path: str | None = None
    current_path: str | None = None
    if write_index:
        index_path = str(
            write_players_index(all_bios, merge_existing=merge_index)
        )
        current_path = str(write_current_index(season, as_of_week))

    return BatchPublishResult(
        season=season,
        as_of_week=as_of_week,
        groups=group_results,
        index_path=index_path,
        current_index_path=current_path,
        players=len({b["id"] for b in all_bios}),
        bios=all_bios,
        seconds=time.perf_counter() - t0,
    )


def publish_range(
    start: int,
    end: int,
    *,
    groups: Iterable[str] | None = None,
    as_of_week: int | None = None,
    also_current_latest: bool = True,
) -> RangePublishResult:
    """Publish season-end pages for each season in [start, end] and merge the index.

    Uses week 17 for 2016–2020 and week 18 for 2021+ unless `as_of_week` is set.
    Only the latest season writes `pages/current/` and `index/current.json`.
    """
    if end < start:
        raise ValueError("end must be >= start")
    ensure_data_dirs()
    t0 = time.perf_counter()
    seasons = list(range(start, end + 1))
    season_results: list[BatchPublishResult] = []
    bios_by_id: dict[str, dict[str, Any]] = {}
    slices: list[dict[str, int]] = []

    for i, season in enumerate(seasons):
        week = as_of_week if as_of_week is not None else default_as_of_week(season)
        also_current = also_current_latest and i == len(seasons) - 1
        if also_current:
            # Drop prior-season current pointers so search doesn't resolve stale years.
            cur_dir = PAGES_DIR / "current"
            if cur_dir.exists():
                shutil.rmtree(cur_dir)
            cur_dir.mkdir(parents=True, exist_ok=True)
        batch = publish_all(
            season,
            week,
            groups=groups,
            also_current=also_current,
            write_index=False,
        )
        season_results.append(batch)
        _merge_bios(bios_by_id, batch.bios)
        slices.append({"season": season, "asOfWeek": week})

    index_path = write_players_index(bios_by_id.values(), merge_existing=False)
    last = season_results[-1]
    current_path = write_current_index(last.season, last.as_of_week)
    seasons_path = write_seasons_index(slices)

    return RangePublishResult(
        start=start,
        end=end,
        seasons=season_results,
        index_path=str(index_path),
        current_index_path=str(current_path),
        seasons_index_path=str(seasons_path),
        players=len(bios_by_id),
        seconds=time.perf_counter() - t0,
    )


def sync_index_to_knowball(knowball_web: Path) -> dict[str, str]:
    """Copy search/current/seasons index into Knowball for local wiring."""
    ensure_data_dirs()
    dest = knowball_web / "public" / "viz" / "index"
    dest.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    names = ("players.json", "current.json", "seasons.json")
    for name in names:
        src = INDEX_DIR / name
        if not src.exists():
            if name == "seasons.json":
                continue
            raise FileNotFoundError(f"missing {src}; run publish-all first")
        target = dest / name
        shutil.copy2(src, target)
        copied[name] = str(target)
    data_dest = knowball_web / "src" / "data" / "ballnet"
    data_dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = INDEX_DIR / name
        if not src.exists():
            continue
        target = data_dest / name
        shutil.copy2(src, target)
        copied[f"src/{name}"] = str(target)
    return copied


def copy_page_to_knowball(
    page_path: Path,
    knowball_fixture: Path,
) -> None:
    """Copy published JSON into Knowball for a temporary visual fixture."""
    knowball_fixture.parent.mkdir(parents=True, exist_ok=True)
    knowball_fixture.write_text(page_path.read_text())
