# Ballnet

Public data pipeline and football data-science engine for [Knowball](https://github.com/ehfurgeson/knowball).

Ballnet ingests nflverse data, applies qualification (ramp–hold), computes league KDE densities and percentiles, and publishes JSON that Knowball renders.

## Status

Stages **A–G** are implemented for all catalog position groups. Full local publish covers **2016–2025** season-end slices (~5412 players in the merged index). League shapes are **KDE-only**. Public JSON lives on Supabase Storage (`knowball-public`).

**Agents:** read [`docs/HANDOFF.md`](docs/HANDOFF.md). Weekly refresh: [`docs/WEEKLY_OPS.md`](docs/WEEKLY_OPS.md).

Contract: knowball `.plans/ballnet-etl-knowball-visualizations.md`.

## Setup

```bash
uv sync
```

## Pipeline

```bash
uv sync

# NGS-era backfill (fetch + spine for each season)
uv run ballnet backfill --start 2016 --end 2025

# One season, every player + index (+ C–E unless --skip-pipeline)
uv run ballnet publish-all --season 2025 --as-of-week 18 --sync-knowball ../knowball/web

# Upload index + that season's pages + league curves
uv run ballnet upload-storage --index --season 2025

# Full multi-season publish (week 17 before 2021, else 18) + merged index
uv run ballnet publish-range --start 2016 --end 2025 --sync-knowball ../knowball/web
```

Parquet lands under gitignored `data/raw/`, `data/spine/`, `data/ytd/`, `data/dists/`. Page JSON under `data/pages/`; league under `data/league/`; search under `data/index/`.

