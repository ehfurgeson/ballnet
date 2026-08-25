# Boundaries: `src/ballnet` pipeline

## Always

- Persist Stage B as parquet under `data/spine/` before building densities or publish JSON.
- Filter `season_type` / `game_type` to REG for box, NGS, snaps, and PFR weeklies.
- Drop NGS `week == 0` (season totals) from the weekly spine.
- Left-join enrichment tables; preserve nulls (`missing_source` later — never impute 0).
- Use GSIS `player_id` as the canonical key; map PFR via `ff_playerids`.
- Apply ramp–hold as `min_n = n_base * min(as_of_week, 4)` with `as_of_week` = viewed NFL week.

## Ask First

- Changing join keys or spine grain `(player_id, season, week)`.
- Adding Supabase / Storage publish credentials or schema migrations.
- Expanding play-grain sources (PBP / FTN / participation) into the spine.

## Never

- Put fantasy draft optimization or ffoptim logic in this package.
- Write raw weeklies into a public `viz` schema.
- Invent Knowball catalog `stat.id` values — ids come from the knowball catalog / ETL brief.
- Average weekly rates for YTD; use ratio-of-sums (and recompute passer rating from components).

## Silent Failures & Gotchas

- `load_ff_opportunity` often types `season`/`week` as strings — cast join keys to Int32 before joining or Polars raises `SchemaError`.
- nflverse uses `SAF` / `MLB` / `DL`; map them to Knowball codes (`S` / `ILB` / `DT`). Long snappers (`LS`) stay unmapped (no catalog).
- `load_player_stats` uses `passing_interceptions` / `sacks_suffered`; the spine renames to `interceptions` / `sacks_taken`.
- Snap and PFR joins are null when `pfr_player_id` is missing from `ff_playerids` — coverage will look low for fringe players, not a join bug.
- FTN, participation, and PBP are cached in `data/raw/` but not fully exploded into the spine yet (play grain).
- Cached parquet under `data/` is gitignored; `--force` / `--force-fetch` redownloads.
- Stage C YTD rates use ratio-of-sums; passer rating is recomputed from YTD components.
- CPOE prefers NGS volume-weighted `completion_percentage_above_expectation` (pp). Averaging weekly box `passing_cpoe` can cancel toward 0 and disagree with season cmp% − xcmp%.
- NGS aggressiveness / expected completion arrive as 0–100; Stage C scales them to 0–1 for Knowball `percent` format.
- Stage C covers all position groups via `catalog/registry.py`. Play-grain stats stay `missing_source` until PBP/participation are on the spine.
- OL spine rows currently lack `pfr_player_id`, so snap joins are null and snap % is `missing_source` (penalties still compute from box).
- Returner `position_group` has no weekly rows (nflverse roster positions are rarely KR/PR).
- Stage G batch (`publish-all`) writes **scalar** JSON under `data/pages/`, shared curves under `data/league/{season}/w{week}/{group}.json`, and rebuilds `data/index/{players,current}.json`. Knowball must not import page JSON into the Next bundle — only the small index.
- Pre-2018 spines omit many PFR advanced columns (ingest empty stubs). Stage C null-fills expected enrichment cols before aggregate so older seasons publish with `missing_source` instead of crashing.
- Storage uploads (`upload-storage`) go to public bucket `knowball-public` under `index/`, `pages/`, `league/`, and `highlights/`. Free plan is ~1 GB — skip `--also-current` (duplicates the season). Dashboard drag-drops often land at bucket root; re-upload with `--index` to get `index/*.json`.
- Service-role key lives in ballnet `.env` only; Knowball fetches public object URLs (no Supabase client).
- Never re-embed league `curve` onto player pages; league shapes live under `data/league/` only.
- League JSON is KDE `curve[]` for every catalog id. Catalog `kind` is metadata; do not emit `bins` or `samples`.
