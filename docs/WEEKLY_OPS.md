# Weekly / post-game ops plan

Goal: after each NFL week (or after a slate finishes), refresh Knowball viz for the **current season** through the latest completed REG week, then upload Storage.

## Canonical one-command shape (target)

```bash
uv run ballnet refresh --season YEAR --as-of-week W \
  --sync-knowball ../knowball/web \
  --upload
```

Intended stages inside `refresh` (not a new math path — wrap what already exists):

1. **A** `fetch --season YEAR` (incremental nflverse cache)
2. **B** `spine --season YEAR`
3. **C–E** for every publishable group at `as-of-week W` (ytd → densities → percentiles)
4. **G** `publish-all --season YEAR --as-of-week W` (scalar pages + league KDE + index/current)
5. **H** `highlights --season YEAR --week W` (weekly z-score board + `league_weekly` KDEs → `data/highlights/` + `data/dists/league_weekly/`)
6. Optional: `--sync-knowball` for the search index copy into Knowball
7. Optional: `--upload` → `upload-storage --index --season YEAR --highlights` (pages + league + highlights + league_weekly)

Until `refresh` exists, run that sequence manually (see below).

## Manual recipe (works today)

```bash
cd ballnet
uv sync

YEAR=2025
W=18   # latest completed REG week

uv run ballnet fetch --season $YEAR
uv run ballnet spine --season $YEAR
uv run ballnet publish-all --season $YEAR --as-of-week $W \
  --sync-knowball ../knowball/web
uv run ballnet highlights --season $YEAR --week $W
uv run ballnet upload-storage --index --season $YEAR --highlights
```

`publish-all` without `--skip-pipeline` already runs C–E then G for all groups.

## Automation options (pick later)

| Option | Pros | Cons |
|---|---|---|
| **Manual** after MNF / Tuesday AM | Simple, secrets stay local | Easy to forget |
| **cron / launchd** on a laptop or small VM | Cheap; same `uv run` | Machine must be on; secrets on disk |
| **GitHub Actions** (scheduled or `workflow_dispatch`) | Auditable; no local machine | Needs self-hosted or paid runner for long jobs + Storage secrets; nflverse download size |

Recommendation for v1: **manual or local cron** calling the recipe above. Move to GitHub Actions only after `refresh` is one command and runtime/secrets are measured.

### Secrets for upload

- ballnet `.env`: `supabase_url`, `supabase_service_role_key`
- Never put the service role in Knowball or the Next bundle
- Knowball only needs public `VIZ_STORAGE_BASE_URL` / `supabase_url` (anon is unused for Storage public URLs)

### Scheduling tip

nflverse weeklies often lag end-of-slate by hours. Prefer **Tuesday ~10:00 America/New_York** (or after you confirm week `W` is complete in raw box scores) over “Monday midnight.”

## What *not* to re-run every week

- Full `publish-range --start 2016 --end …` — historical season-end slices are static unless you intentionally rebuild
- `publish-league-range` alone — prefer full `publish-all` so pages + percentiles stay aligned with league curves
- `--also-current` Storage upload — doubles quota; Knowball uses `index/current.json`

## Failure modes to watch

- Empty/partial nflverse week → sparse YTD / wrong as-of-week — verify spine week max before publish
- Storage free-tier burst drops — retry `upload-storage --season YEAR` (upsert)
- Index out of sync with Knowball search — always `--sync-knowball` or re-copy `data/index/*.json`

## Follow-ups before automating

1. Implement CLI `refresh` (wrapper; no new Stage math).
2. Optional: detect latest REG week from spine / nflverse schedule instead of passing `--as-of-week`.
3. Log a small JSON report (season, week, player count, upload bytes, duration) under `data/logs/`.
4. Only then wire cron or Actions.

## Recommendation
Keep the manual/local-cron until a thin ballnet refresh wrapper exists; only then consider GitHub Actions. Prefer Tuesday AM (America/New_York) so nflverse weeklies have landed.