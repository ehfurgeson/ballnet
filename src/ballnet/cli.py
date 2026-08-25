"""CLI for Stage A–G pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ballnet.ingest import fetch_season
from ballnet.panel import build_spine
from ballnet.stage_c import build_ytd, player_stat_table
from ballnet.catalog.registry import POSITION_GROUPS, STAT_IDS_BY_GROUP
from ballnet.density import build_densities
from ballnet.percentiles import attach_percentiles
from ballnet.publish import (
    PUBLISHABLE_GROUPS,
    default_as_of_week,
    publish_all,
    publish_league_slice,
    publish_player_page,
    publish_range,
    sync_index_to_knowball,
    write_seasons_index,
)
from ballnet.storage_upload import (
    upload_index,
    upload_season_league,
    upload_season_pages,
)
import polars as pl

# Demo players for local visual review (2024 REG through week 18).
DEMO_PLAYERS: list[tuple[str, str, str]] = [
    ("qb", "00-0039918", "Caleb Williams"),
    ("backfield", "00-0034844", "Saquon Barkley"),
    ("pass_catcher", "00-0036900", "Ja'Marr Chase"),
    ("ol", "00-0032380", "Laremy Tunsil"),
    ("def_front", "00-0033868", "Myles Garrett"),
    ("secondary", "00-0033281", "Marlon Humphrey"),
    ("kicker", "00-0037692", "Brandon Aubrey"),
    ("punter", "00-0039745", "Tory Taylor"),
]


def _season_list(args: argparse.Namespace) -> list[int]:
    if getattr(args, "seasons", None):
        return list(args.seasons)
    start = getattr(args, "start", None)
    end = getattr(args, "end", None)
    if start is not None and end is not None:
        if end < start:
            raise SystemExit("--end must be >= --start")
        return list(range(start, end + 1))
    season = getattr(args, "season", None)
    if season is not None:
        return [season]
    raise SystemExit("Provide --season YEAR or --start/--end")


def _add_season_args(p: argparse.ArgumentParser, *, require_one: bool = True) -> None:
    g = p.add_mutually_exclusive_group(required=require_one)
    g.add_argument("--season", type=int, help="Single season")
    g.add_argument(
        "--start",
        type=int,
        help="First season (inclusive); requires --end",
    )
    p.add_argument("--end", type=int, help="Last season (inclusive); requires --start")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ballnet", description="Knowball viz data pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="Stage A: download nflverse sources")
    _add_season_args(p_fetch)
    p_fetch.add_argument("--force", action="store_true")

    p_spine = sub.add_parser("spine", help="Stage A (if needed) + Stage B weekly panel")
    _add_season_args(p_spine)
    p_spine.add_argument("--force-fetch", action="store_true")

    p_backfill = sub.add_parser(
        "backfill",
        help="Fetch + spine for a season range; print coverage summary",
    )
    p_backfill.add_argument("--start", type=int, required=True)
    p_backfill.add_argument("--end", type=int, required=True)
    p_backfill.add_argument("--force-fetch", action="store_true")

    p_ytd = sub.add_parser("ytd", help="Stage C: catalog YTD panel")
    p_ytd.add_argument("--season", type=int, required=True)
    p_ytd.add_argument("--as-of-week", type=int, required=True)
    p_ytd.add_argument("--group", default="qb", choices=list(POSITION_GROUPS))
    p_ytd.add_argument(
        "--player",
        default=None,
        help="Optional GSIS player_id to print (e.g. Caleb Williams 00-0039918)",
    )

    p_dens = sub.add_parser("densities", help="Stage D: league_ytd densities from Stage C")
    p_dens.add_argument("--season", type=int, required=True)
    p_dens.add_argument("--as-of-week", type=int, required=True)
    p_dens.add_argument("--group", default="qb", choices=list(POSITION_GROUPS))

    p_pct = sub.add_parser("percentiles", help="Stage E: oriented percentiles on Stage C")
    p_pct.add_argument("--season", type=int, required=True)
    p_pct.add_argument("--as-of-week", type=int, required=True)
    p_pct.add_argument("--group", default="qb", choices=list(POSITION_GROUPS))

    p_pub = sub.add_parser("publish", help="Stage G: local PlayerPageJson for one player")
    p_pub.add_argument("--season", type=int, required=True)
    p_pub.add_argument("--as-of-week", type=int, required=True)
    p_pub.add_argument("--player", type=str, required=True, help="GSIS player_id")
    p_pub.add_argument("--group", default="qb", choices=list(POSITION_GROUPS))
    p_pub.add_argument(
        "--no-current",
        action="store_true",
        help="Skip writing pages/current/{player_id}.json",
    )

    p_demos = sub.add_parser(
        "publish-demos",
        help="Stage C–G for one demo player per position group (skips returner)",
    )
    p_demos.add_argument("--season", type=int, default=2024)
    p_demos.add_argument("--as-of-week", type=int, default=18)

    p_all = sub.add_parser(
        "publish-all",
        help="Stage C–G for every player in all groups + index/players.json",
    )
    p_all.add_argument("--season", type=int, required=True)
    p_all.add_argument("--as-of-week", type=int, required=True)
    p_all.add_argument(
        "--group",
        action="append",
        choices=list(POSITION_GROUPS),
        dest="groups",
        help="Limit to group(s); repeatable. Default: all except returner",
    )
    p_all.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Reuse existing YTD/density/percentile parquet (Stage G + index only)",
    )
    p_all.add_argument(
        "--no-current",
        action="store_true",
        help="Skip writing pages/current/{player_id}.json",
    )
    p_all.add_argument(
        "--sync-knowball",
        type=Path,
        default=None,
        help="Copy index JSON into Knowball web/ (e.g. ../knowball/web)",
    )

    p_range = sub.add_parser(
        "publish-range",
        help="Stage C–G for every season in [start, end]; merge multi-season index",
    )
    p_range.add_argument("--start", type=int, required=True)
    p_range.add_argument("--end", type=int, required=True)
    p_range.add_argument(
        "--as-of-week",
        type=int,
        default=None,
        help="Override final week for all seasons (default: 17 before 2021, else 18)",
    )
    p_range.add_argument(
        "--group",
        action="append",
        choices=list(POSITION_GROUPS),
        dest="groups",
        help="Limit to group(s); repeatable. Default: all except returner",
    )
    p_range.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Reuse existing YTD/density/percentile parquet (Stage G + index only)",
    )
    p_range.add_argument(
        "--no-current",
        action="store_true",
        help="Skip writing pages/current/ and leave index/current.json unchanged",
    )
    p_range.add_argument(
        "--sync-knowball",
        type=Path,
        default=None,
        help="Copy index JSON into Knowball web/ after the range completes",
    )

    p_league = sub.add_parser(
        "publish-league-range",
        help="Regenerate league/{season}/w{week}/{group}.json only (KDE curves)",
    )
    p_league.add_argument("--start", type=int, required=True)
    p_league.add_argument("--end", type=int, required=True)
    p_league.add_argument(
        "--as-of-week",
        type=int,
        default=None,
        help="Override week for every season (default: 17 before 2021, else 18)",
    )
    p_league.add_argument(
        "--group",
        action="append",
        choices=list(POSITION_GROUPS),
        dest="groups",
        help="Limit to group(s); repeatable. Default: all except returner",
    )

    p_up = sub.add_parser(
        "upload-storage",
        help="Upload local index/pages/league JSON to Supabase Storage (knowball-public)",
    )
    p_up.add_argument(
        "--season",
        type=int,
        default=None,
        help="Upload pages/{season}/w{week}/*.json and league/{season}/w{week}/*.json",
    )
    p_up.add_argument(
        "--as-of-week",
        type=int,
        default=None,
        help="Week folder (default: 17 before 2021, else 18)",
    )
    p_up.add_argument(
        "--index",
        action="store_true",
        help="Upload data/index/{players,current,seasons}.json to index/",
    )
    p_up.add_argument(
        "--pages-only",
        action="store_true",
        help="With --season, upload pages only (skip league/)",
    )
    p_up.add_argument(
        "--league-only",
        action="store_true",
        help="With --season, upload league/ only (skip pages)",
    )
    p_up.add_argument(
        "--also-current",
        action="store_true",
        help="Also copy season pages to pages/current/ (doubles Storage use)",
    )
    p_up.add_argument("--bucket", default="knowball-public")
    p_up.add_argument("--workers", type=int, default=8)

    args = parser.parse_args(argv)

    if args.cmd == "fetch":
        if getattr(args, "start", None) is not None and args.end is None:
            raise SystemExit("--start requires --end")
        seasons = _season_list(args)
        for season in seasons:
            print(f"=== fetch {season} ===")
            for r in fetch_season(season, force=args.force):
                print(
                    f"{r.name:28} rows={r.rows:7d} cols={r.cols:3d} "
                    f"{r.seconds:6.2f}s  {r.notes}"
                )
        return

    if args.cmd == "spine":
        if getattr(args, "start", None) is not None and args.end is None:
            raise SystemExit("--start requires --end")
        seasons = _season_list(args)
        for season in seasons:
            print(f"=== spine {season} ===")
            fetch_season(season, force=args.force_fetch)
            spine = build_spine(season)
            print(f"spine: {spine.path}")
            print(f"rows={spine.rows} cols={spine.cols} seconds={spine.seconds:.2f}")
            print("join_coverage:")
            print(json.dumps({k: round(v, 4) for k, v in spine.join_coverage.items()}, indent=2))
        return

    if args.cmd == "backfill":
        if args.end < args.start:
            raise SystemExit("--end must be >= --start")
        seasons = list(range(args.start, args.end + 1))
        t0 = time.perf_counter()
        summary: list[dict] = []
        failures: list[str] = []
        for season in seasons:
            print(f"=== backfill {season} ===", flush=True)
            try:
                fetch_results = fetch_season(season, force=args.force_fetch)
                fetch_secs = sum(r.seconds for r in fetch_results)
                spine = build_spine(season)
                row = {
                    "season": season,
                    "fetch_seconds": round(fetch_secs, 2),
                    "spine_rows": spine.rows,
                    "spine_cols": spine.cols,
                    "spine_seconds": round(spine.seconds, 3),
                    "join_coverage": {k: round(v, 4) for k, v in spine.join_coverage.items()},
                }
                summary.append(row)
                cov = spine.join_coverage
                print(
                    f"  ok rows={spine.rows} spine={spine.seconds:.2f}s "
                    f"pos={cov.get('position_mapped', 0):.3f} "
                    f"pfr={cov.get('pfr_id_mapped', 0):.3f} "
                    f"snaps={cov.get('snaps', 0):.3f}",
                    flush=True,
                )
            except Exception as e:
                failures.append(f"{season}: {type(e).__name__}: {e}")
                print(f"  FAIL {season}: {e}", flush=True)

        report = {
            "start": args.start,
            "end": args.end,
            "total_seconds": round(time.perf_counter() - t0, 2),
            "seasons_ok": [r["season"] for r in summary],
            "failures": failures,
            "summary": summary,
        }
        print(json.dumps(report, indent=2))
        if failures:
            sys.exit(1)
        # Gate on coverage for every season
        bad_seasons = [
            r["season"]
            for r in summary
            if r["join_coverage"].get("position_mapped", 0) < 0.5
            or r["join_coverage"].get("pfr_id_mapped", 0) < 0.5
        ]
        if bad_seasons:
            print(f"WARNING: low coverage seasons {bad_seasons}", file=sys.stderr)
            sys.exit(2)
        return

    if args.cmd == "ytd":
        result = build_ytd(args.season, args.as_of_week, position_group=args.group)
        long = pl.read_parquet(result.path)
        report: dict = {
            "season": args.season,
            "as_of_week": args.as_of_week,
            "group": args.group,
            "path": result.path,
            "rows": result.rows,
            "players": result.players,
            "seconds": round(result.seconds, 3),
            "catalog_ids": len(STAT_IDS_BY_GROUP[args.group]),
            "qualified_rows": int(long.filter(pl.col("qualified")).height),
            "reasons": long.group_by("unavailable_reason").len().to_dicts(),
        }
        if args.player:
            table = player_stat_table(long, args.player)
            if table.height == 0:
                print(f"ERROR: player {args.player} not found in YTD panel", file=sys.stderr)
                print(json.dumps(report, indent=2))
                sys.exit(1)
            expected = set(STAT_IDS_BY_GROUP[args.group])
            # alwaysUnavailable ids are omitted from Stage C by design
            from ballnet.catalog.registry import stats_for_group

            expected -= {s.id for s in stats_for_group(args.group) if s.always_unavailable}
            missing_ids = sorted(expected - set(table["stat_id"].to_list()))
            report["player"] = {
                "player_id": args.player,
                "name": long.filter(pl.col("player_id") == args.player)[
                    "player_display_name"
                ][0],
                "stats": table.to_dicts(),
                "missing_catalog_ids": missing_ids,
                "all_present": len(missing_ids) == 0,
            }
        print(json.dumps(report, indent=2, default=str))
        if args.player and report["player"]["missing_catalog_ids"]:
            sys.exit(1)
        return

    if args.cmd == "densities":
        result = build_densities(args.season, args.as_of_week, position_group=args.group)
        print(
            json.dumps(
                {
                    "season": args.season,
                    "as_of_week": args.as_of_week,
                    "group": args.group,
                    "path": result.path,
                    "rows": result.rows,
                    "seconds": round(result.seconds, 3),
                },
                indent=2,
            )
        )
        return

    if args.cmd == "percentiles":
        # Densities must exist
        dens = build_densities(args.season, args.as_of_week, position_group=args.group)
        result = attach_percentiles(args.season, args.as_of_week, position_group=args.group)
        print(
            json.dumps(
                {
                    "season": args.season,
                    "as_of_week": args.as_of_week,
                    "group": args.group,
                    "densities": dens.path,
                    "path": result.path,
                    "rows": result.rows,
                    "seconds": round(result.seconds, 3),
                },
                indent=2,
            )
        )
        return

    if args.cmd == "publish":
        # Ensure D+E exist for this slice
        build_densities(args.season, args.as_of_week, position_group=args.group)
        attach_percentiles(args.season, args.as_of_week, position_group=args.group)
        result = publish_player_page(
            args.season,
            args.as_of_week,
            args.player,
            position_group=args.group,
            also_current=not args.no_current,
        )
        # Quick validation summary
        page = json.loads(Path(result.path).read_text())
        ready = [
            s
            for s in page["stats"]
            if s.get("qualified") and s.get("percentile") is not None
        ]
        print(
            json.dumps(
                {
                    "path": result.path,
                    "current_path": result.current_path,
                    "stats": result.stats,
                    "ready_stats": len(ready),
                    "player": page["player"],
                    "season": page["season"],
                    "asOfWeek": page["asOfWeek"],
                    "schemaVersion": page.get("schemaVersion"),
                    "seconds": round(result.seconds, 3),
                    "sample": [
                        {
                            "id": s["id"],
                            "playerValue": s.get("playerValue"),
                            "percentile": s.get("percentile"),
                            "qualified": s.get("qualified"),
                            "unavailableReason": s.get("unavailableReason"),
                        }
                        for s in page["stats"][:8]
                    ],
                },
                indent=2,
            )
        )
        return

    if args.cmd == "publish-demos":
        summary: list[dict] = []
        for group, player_id, name in DEMO_PLAYERS:
            print(f"=== {group} {name} ({player_id}) ===", flush=True)
            ytd = build_ytd(args.season, args.as_of_week, position_group=group)
            dens = build_densities(args.season, args.as_of_week, position_group=group)
            pct = attach_percentiles(args.season, args.as_of_week, position_group=group)
            pub = publish_player_page(
                args.season,
                args.as_of_week,
                player_id,
                position_group=group,
            )
            page = json.loads(Path(pub.path).read_text())
            ready = sum(
                1
                for s in page["stats"]
                if s.get("qualified") and s.get("percentile") is not None
            )
            gray = sum(1 for s in page["stats"] if s.get("unavailableReason"))
            summary.append(
                {
                    "group": group,
                    "player_id": player_id,
                    "name": name,
                    "path": pub.path,
                    "stats": pub.stats,
                    "ready": ready,
                    "with_reason": gray,
                    "ytd_players": ytd.players,
                    "density_stats": dens.rows,
                    "pct_rows": pct.rows,
                }
            )
        print(json.dumps({"season": args.season, "as_of_week": args.as_of_week, "demos": summary}, indent=2))
        return

    if args.cmd == "publish-all":
        groups = args.groups or list(PUBLISHABLE_GROUPS)
        pipeline_summary: list[dict] = []
        if not args.skip_pipeline:
            for group in groups:
                print(f"=== pipeline {group} ===", flush=True)
                ytd = build_ytd(args.season, args.as_of_week, position_group=group)
                dens = build_densities(args.season, args.as_of_week, position_group=group)
                pct = attach_percentiles(args.season, args.as_of_week, position_group=group)
                pipeline_summary.append(
                    {
                        "group": group,
                        "ytd_players": ytd.players,
                        "density_stats": dens.rows,
                        "pct_rows": pct.rows,
                    }
                )
                print(
                    f"  ytd_players={ytd.players} dens={dens.rows} pct_rows={pct.rows}",
                    flush=True,
                )

        print("=== publish-all Stage G + index ===", flush=True)
        batch = publish_all(
            args.season,
            args.as_of_week,
            groups=groups,
            also_current=not args.no_current,
        )
        write_seasons_index(
            [{"season": batch.season, "asOfWeek": batch.as_of_week}]
        )
        sync_paths: dict[str, str] | None = None
        if args.sync_knowball is not None:
            sync_paths = sync_index_to_knowball(args.sync_knowball.resolve())

        print(
            json.dumps(
                {
                    "season": batch.season,
                    "as_of_week": batch.as_of_week,
                    "players": batch.players,
                    "seconds": round(batch.seconds, 3),
                    "index_path": batch.index_path,
                    "current_index_path": batch.current_index_path,
                    "pipeline": pipeline_summary,
                    "groups": [
                        {
                            "group": g.position_group,
                            "players": g.players,
                            "paths": g.paths,
                            "seconds": round(g.seconds, 3),
                        }
                        for g in batch.groups
                    ],
                    "sync_knowball": sync_paths,
                },
                indent=2,
            )
        )
        return

    if args.cmd == "publish-league-range":
        if args.end < args.start:
            raise SystemExit("--end must be >= --start")
        groups = args.groups or list(PUBLISHABLE_GROUPS)
        reports: list[dict] = []
        t0 = time.perf_counter()
        for season in range(args.start, args.end + 1):
            week = (
                args.as_of_week
                if args.as_of_week is not None
                else default_as_of_week(season)
            )
            paths = publish_league_slice(season, week, groups=groups)
            reports.append(
                {
                    "season": season,
                    "as_of_week": week,
                    "groups": len(paths),
                    "paths": [str(p) for p in paths],
                }
            )
            print(f"=== league {season} w{week}: {len(paths)} groups ===", flush=True)
        print(
            json.dumps(
                {
                    "start": args.start,
                    "end": args.end,
                    "seconds": round(time.perf_counter() - t0, 3),
                    "seasons": reports,
                },
                indent=2,
            )
        )
        return

    if args.cmd == "publish-range":
        if args.end < args.start:
            raise SystemExit("--end must be >= --start")
        groups = args.groups or list(PUBLISHABLE_GROUPS)
        seasons = list(range(args.start, args.end + 1))
        pipeline_summary: list[dict] = []

        if not args.skip_pipeline:
            for season in seasons:
                week = (
                    args.as_of_week
                    if args.as_of_week is not None
                    else default_as_of_week(season)
                )
                print(f"=== pipeline {season} w{week} ===", flush=True)
                for group in groups:
                    ytd = build_ytd(season, week, position_group=group)
                    dens = build_densities(season, week, position_group=group)
                    pct = attach_percentiles(season, week, position_group=group)
                    row = {
                        "season": season,
                        "as_of_week": week,
                        "group": group,
                        "ytd_players": ytd.players,
                        "density_stats": dens.rows,
                        "pct_rows": pct.rows,
                    }
                    pipeline_summary.append(row)
                    print(
                        f"  {group}: ytd_players={ytd.players} dens={dens.rows} "
                        f"pct_rows={pct.rows}",
                        flush=True,
                    )

        print("=== publish-range Stage G + merged index ===", flush=True)
        range_result = publish_range(
            args.start,
            args.end,
            groups=groups,
            as_of_week=args.as_of_week,
            also_current_latest=not args.no_current,
        )
        sync_paths = None
        if args.sync_knowball is not None:
            sync_paths = sync_index_to_knowball(args.sync_knowball.resolve())

        print(
            json.dumps(
                {
                    "start": range_result.start,
                    "end": range_result.end,
                    "players": range_result.players,
                    "seconds": round(range_result.seconds, 3),
                    "index_path": range_result.index_path,
                    "current_index_path": range_result.current_index_path,
                    "seasons_index_path": range_result.seasons_index_path,
                    "pipeline": pipeline_summary,
                    "seasons": [
                        {
                            "season": s.season,
                            "as_of_week": s.as_of_week,
                            "players": s.players,
                            "seconds": round(s.seconds, 3),
                            "groups": [
                                {
                                    "group": g.position_group,
                                    "players": g.players,
                                    "seconds": round(g.seconds, 3),
                                }
                                for g in s.groups
                            ],
                        }
                        for s in range_result.seasons
                    ],
                    "sync_knowball": sync_paths,
                },
                indent=2,
            )
        )
        return

    if args.cmd == "upload-storage":
        if not args.index and args.season is None:
            raise SystemExit("Provide --index and/or --season YEAR")
        if args.pages_only and args.league_only:
            raise SystemExit("Use only one of --pages-only / --league-only")
        reports: list[dict] = []
        if args.index:
            print("=== upload index/ ===", flush=True)
            r = upload_index(bucket=args.bucket, workers=args.workers)
            reports.append(
                {
                    "what": "index",
                    "uploaded": r.uploaded,
                    "failed": r.failed,
                    "bytes": r.bytes,
                    "seconds": round(r.seconds, 3),
                    "errors": r.errors,
                }
            )
            print(
                f"  uploaded={r.uploaded} failed={r.failed} "
                f"bytes={r.bytes} seconds={r.seconds:.1f}",
                flush=True,
            )
        if args.season is not None:
            week = (
                args.as_of_week
                if args.as_of_week is not None
                else default_as_of_week(args.season)
            )
            do_pages = not args.league_only
            do_league = not args.pages_only
            if do_pages:
                print(
                    f"=== upload pages/{args.season}/w{week}/ "
                    f"(also_current={args.also_current}) ===",
                    flush=True,
                )
                r = upload_season_pages(
                    args.season,
                    week,
                    bucket=args.bucket,
                    also_current=args.also_current,
                    workers=args.workers,
                )
                reports.append(
                    {
                        "what": f"pages/{args.season}/w{week}",
                        "uploaded": r.uploaded,
                        "failed": r.failed,
                        "bytes": r.bytes,
                        "seconds": round(r.seconds, 3),
                        "errors": r.errors,
                    }
                )
                print(
                    f"  uploaded={r.uploaded} failed={r.failed} "
                    f"bytes={r.bytes} seconds={r.seconds:.1f}",
                    flush=True,
                )
            if do_league:
                print(
                    f"=== upload league/{args.season}/w{week}/ ===",
                    flush=True,
                )
                r = upload_season_league(
                    args.season,
                    week,
                    bucket=args.bucket,
                    workers=args.workers,
                )
                reports.append(
                    {
                        "what": f"league/{args.season}/w{week}",
                        "uploaded": r.uploaded,
                        "failed": r.failed,
                        "bytes": r.bytes,
                        "seconds": round(r.seconds, 3),
                        "errors": r.errors,
                    }
                )
                print(
                    f"  uploaded={r.uploaded} failed={r.failed} "
                    f"bytes={r.bytes} seconds={r.seconds:.1f}",
                    flush=True,
                )
        failed = sum(x["failed"] for x in reports)
        print(json.dumps({"bucket": args.bucket, "uploads": reports}, indent=2))
        if failed:
            sys.exit(1)
        return

    parser.error(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()
