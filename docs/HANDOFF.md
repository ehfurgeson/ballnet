# Ballnet handoff — next agent

Read this first, then `src/ballnet/BOUNDARIES.md` (+ `catalog/BOUNDARIES.md`), then knowball’s ETL contract. Do **not** invent a parallel contract.

## Product context

| Repo | Role |
|---|---|
| **knowball** (public) | Next.js viz only — renders JSON |
| **ballnet** (this repo, public) | nflverse ingest → densities → percentiles → publish JSON |
| **ffoptim** (private) | Draft optimizer — out of scope here |

| Doc | Why |
|---|---|
| knowball `.plans/ballnet-etl-knowball-visualizations.md` | Locked publish / JSON / store contract |
| `docs/adr/2026-08-24-kde-every-stat.md` | KDE is the only league density shape |
| `docs/WEEKLY_OPS.md` | Post-game / weekly refresh + upload plan |
| knowball `.plans/NFL Stats Sliders.md` | Sources, min-n, zero-mass |
| knowball `web/src/lib/catalog/` | `stat.id` authority |
| knowball `web/src/lib/payload.ts`, `distribution.ts` | JSON + CDF shapes |

## Done (do not redo)

Human visual review of demos is **approved**. Local D→E→G math is good. **Full local Stage G** covers **2016–2025** season-end slices with a merged search index (**5412** players). Storage holds scalar pages + league KDE + index (no `pages/current/`). Histograms / rug `samples` are retired.

| Stage | What | How to run / artifacts |
|---|---|---|
| **A** | Fetch nflverse → `data/raw/*.parquet` | `uv run ballnet fetch --season YEAR` or `backfill --start 2016 --end 2025` |
| **B** | Weekly spine `(player_id, season, week)` | `uv run ballnet spine --season YEAR` → `data/spine/player_week_{season}.parquet` |
| **C** | Catalog YTD long panel — **all** position groups | `uv run ballnet ytd --season … --as-of-week … --group GROUP` |
| **D** | `league_ytd` KDE densities | `uv run ballnet densities …` → `data/dists/` |
| **E** | Oriented percentiles | `uv run ballnet percentiles …` → `data/ytd/*_pct.parquet` |
| **G** | All players for one slice | `uv run ballnet publish-all --season … --as-of-week …` |
| **G range** | Multi-season pages + merged index | `uv run ballnet publish-range --start 2016 --end 2025` |
| **H** | Weekly highlight board + game-level KDEs | `uv run ballnet highlights --season … --week …` → `data/highlights/` + `data/dists/league_weekly/` |

Spines exist for **2016–2025**. `data/` is gitignored (~4GB+ pages).

### Published local slices

| Season | asOfWeek | Pages |
|---|---|---|
| 2016–2020 | 17 (17-week REG) | ~1.8–2.0k each |
| 2021–2025 | 18 | ~1.9–2.1k each |

| Artifact | Location |
|---|---|
| Page JSON (scalars) | `data/pages/{season}/w{week}/{gsis}.json` |
| League KDE curves | `data/league/{season}/w{week}/{group}.json` |
| Weekly highlights | `data/highlights/{season}/w{week}.json` (Stage H) |
| Game-level KDEs | `data/dists/league_weekly/{season}/w{week}/{group}.json` (Stage H allowlist) |
| Current pages | `data/pages/current/` = latest season only (2025 w18) |
| Search index | `data/index/players.json` (`schemaVersion: 1`, seasons unioned) |
| Current pointer | `data/index/current.json` → `{ season: 2025, asOfWeek: 18 }` |
| Season map | `data/index/seasons.json` → published `(season, asOfWeek)` list |

Returner still has no spine rows (skipped). **Do not** embed league curves on player pages.

### Supabase Storage (free tier)

Bucket `knowball-public` (public read) — **2016–2025 uploaded** (scalar pages + league + index; no `pages/current/`).

| Path | Status |
|---|---|
| `index/{players,current,seasons}.json` | uploaded |
| `pages/{season}/w{week}/*.json` | uploaded (all published seasons) |
| `league/{season}/w{week}/{group}.json` | uploaded (KDE curves; required for charts) |
| `highlights/{season}/w{week}.json` | Stage H weekly boards (upload with `--highlights`) |
| `dists/league_weekly/{season}/w{week}/{group}.json` | Stage H single-game KDEs (uploaded with `--highlights`) |
| `pages/current/` | **skipped** (Knowball uses `index/current.json`) |

CLI: `uv run ballnet upload-storage --index --season YEAR` (uploads pages + league).

### Knowball local wiring

- Search / bios / season map: synced via `--sync-knowball ../knowball/web` → `web/src/data/ballnet/{players,current,seasons}.json`
- Page + league: Storage when remote configured; for local smoke set `VIZ_PREFER_LOCAL=1` and `BALLNET_DATA_DIR` to this repo’s `data/`. Clear those to use Storage.
- Do **not** import page JSON into the Next bundle. **No** Supabase client in Knowball.

---

## Your next work (ordered)

### 1. Deploy Knowball + weekly refresh

- Point Knowball at Storage (no `VIZ_PREFER_LOCAL`) for production.
- Follow `docs/WEEKLY_OPS.md` for post-game updates; implement CLI `refresh` when ready to automate.

### 2. Spine / catalog gaps (do not block deploy unless asked)

| Gap | Symptom | Note |
|---|---|---|
| OL `pfr_player_id` null on spine | snap % / snaps_played `missing_source` | Ask before changing join keys |
| No `returner` rows | KR/PR catalogs empty | Roster positions are usually WR/RB; need an explicit rule — **ask first** |
| PBP / participation / FTN not exploded | `red_zone_*`, `route_pct` always `missing_source` | Cached in `data/raw/` already |
| Pre-2018 PFR advanced cols missing on spine | many PFR stats `missing_source` | Stage C null-fills; expected |
| Catalog domains are weekly-ish | Season YTD volumes expand `xMin`/`xMax` | Intentional per contract |

---

## Key gotchas (already burned)

**Joins / spine**

- Cast `season`/`week` to Int32 before joins (`ff_opportunity` is often str).
- Dedupe enrichment keys before left-join (2021 snap dup exploded rows).
- Snap/PFR null when `pfr_player_id` missing — looks like low coverage, often not a join bug.
- nflverse `SAF`/`MLB`/`DL` → Knowball `S`/`ILB`/`DT`; `LS` unmapped.

**YTD / rates**

- Never average weekly rates; ratio-of-sums (and recompute passer rating from components via `rating.py`).
- CPOE: prefer NGS volume-weighted pp — **do not** average weekly box `passing_cpoe`.
- NGS aggressiveness / expected completion / 8+ defenders: scale **0–100 → 0–1**.
- Never impute 0 for missing NGS/PFR/snaps → `missing_source`.
- Pre-2018 spines omit many PFR columns (ingest empty stubs). Stage C null-fills before aggregate.
- `wide_to_long` must use an **explicit Polars schema** — null `unavailable_reason` early in the list otherwise fails schema inference.

**Densities / percentiles**

- Every catalog id: Gaussian KDE + reflection at catalog bounds; grid on `[xMin,xMax]`; ∫y dx ≈ 1. Catalog `kind: discrete` is metadata only (formatting/UI).
- Percentiles use the same inclusive CDF as knowball `kdeCdf`, then orient with `higherIsBetter`.
- KDE `y` is **density** (can be ≫ 1 on narrow domains). UI must not label it as “% of league.”
- Season-YTD volume samples often exceed catalog weekly domains → `_expand_domain` widens charts for that `(season, week, group, stat)`.

**Catalog / publish**

- Ids must match knowball; registry in `catalog/registry.py`.
- `alwaysUnavailable` (OL pass-pro) omitted from Stage C; Knowball still grays from its catalog.
- Play-grain ids stay in catalog but Stage C hard-codes `missing_source` until spine has them.
- `publish` writes **scalar** player pages; league KDE curves are separate under `data/league/`.
- `publish-range` uses week **17** for 2016–2020 and **18** for 2021+; merges `seasons[]` in the index; only the latest season writes `pages/current/` (clears prior pointers first).

**CLI**

```bash
uv sync
uv run ballnet backfill --start 2016 --end 2025
uv run ballnet publish-all --season 2024 --as-of-week 18
uv run ballnet publish-range --start 2016 --end 2025
uv run ballnet publish-league-range --start 2016 --end 2025
uv run ballnet upload-storage --index --season 2025
```

Use **uv** only.

---

## Layout

```
src/ballnet/
  catalog/          # qb, backfield, pass_catcher, ol, def_front, secondary, special_teams + registry
  ingest.py         # Stage A
  panel.py          # Stage B
  stage_c.py        # Stage C (all groups)
  density.py        # Stage D (KDE-only)
  percentiles.py    # Stage E
  publish.py        # Stage G local JSON + index + range
  storage_upload.py # Supabase Storage
  cli.py
  BOUNDARIES.md
data/               # gitignored: raw/, spine/, ytd/, dists/, league/, pages/, index/
```

---

## Later (non-blocking)

1. CLI `refresh` wrapper per `docs/WEEKLY_OPS.md`.
2. Shared catalog export so ballnet cannot drift from knowball.
3. Tests: Stage C fixtures + density ∫y dx ≈ 1.
4. Package layout split if the module grows.
5. Keep `archive/` gitignored; scrub history before making GitHub public if needed.

Do **not** “clean up” by rewriting Stage B join semantics or inventing new `stat.id`s.
