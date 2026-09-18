# Weekly / post-game ops plan

Goal: after each NFL week (or after a slate finishes), refresh Knowball viz for the **current season** through the latest completed REG week, then upload Storage.

## Canonical one-command shape (target)

```bash
uv run ballnet refresh --season YEAR --as-of-week W --upload
```

Intended stages inside `refresh` (not a new math path — wrap what already exists):

1. **A** `fetch --season YEAR` (incremental nflverse cache)
2. **B** `spine --season YEAR`
3. **C–E** for every publishable group at `as-of-week W` (ytd → densities → percentiles)
4. **G** `publish-all --season YEAR --as-of-week W` (scalar pages + league KDE + search leaderboards + **merged** index/current/seasons)
5. **H** `highlights --season YEAR --week W` (weekly z-score board + `league_weekly` KDEs → `data/highlights/` + `data/dists/league_weekly/`)
6. Optional: `--upload` → `upload-storage --index --season YEAR --highlights` (pages + league + leaderboards + highlights + league_weekly)

Until `refresh` exists, run that sequence manually (see below).

## Manual recipe (works today)

```bash
cd ballnet
uv sync

YEAR=2026
W=1   # latest completed REG week

uv run ballnet fetch --season $YEAR --force
uv run ballnet spine --season $YEAR --force-fetch
uv run ballnet publish-all --season $YEAR --as-of-week $W
uv run ballnet highlights --season $YEAR --week $W
uv run ballnet upload-storage --index --season $YEAR --as-of-week $W --highlights
```

`publish-all` without `--skip-pipeline` already runs C–E then G for all groups. It **merges** into the multi-season players/seasons index by default (use `--replace-index` only when intentionally wiping history).

Knowball reads index/pages from **Supabase Storage** (or sibling `ballnet/data` locally). Do not commit Ballnet JSON into the Knowball repo.

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

## Fantasy position ranks

Weekly `publish-all` for the **current** season refreshes FantasyPros weekly ECR (`fantasyPosRankKind: "consensus"`) on pages, search leaderboards, and highlight rows (QB/WR/RB/TE only).

Closed seasons use PPR finish among that position (`kind: "finish"`). After a season ends and `index/current.json` advances to year `Y+1`, **republish year `Y` at its final REG week** so last year’s pages switch from consensus to finish:

```bash
# Current pointer is already Y+1
uv run ballnet publish-all --season $Y --as-of-week 18 --no-current --skip-pipeline
uv run ballnet highlights --season $Y --week 18
uv run ballnet upload-storage --season $Y --as-of-week 18 --highlights
```

(`--no-current` keeps `pages/current/` and `index/current.json` on the new year. Kind resolution then treats `$Y` as finish.)

One-time historical backfill (finish ranks on 2016–last closed season):

```bash
uv run ballnet publish-range --start 2016 --end $Y --skip-pipeline --no-current
# then upload each season slice + index
```

## What *not* to re-run every week

- Full `publish-range --start 2016 --end …` — historical season-end slices are static unless you intentionally rebuild
- `publish-league-range` alone — prefer full `publish-all` so pages + percentiles stay aligned with league curves
- `--also-current` Storage upload — doubles quota; Knowball uses `index/current.json`

## Failure modes to watch

- Empty/partial nflverse week → sparse YTD / wrong as-of-week — verify spine week max before publish
- Storage free-tier burst drops — retry `upload-storage --season YEAR` (upsert)
- Index out of sync with Knowball search — re-run `upload-storage --index` (Knowball reads Storage, not a git-synced copy)

## Follow-ups before automating

1. Implement CLI `refresh` (wrapper; no new Stage math).
2. Optional: detect latest REG week from spine / nflverse schedule instead of passing `--as-of-week`.
3. Log a small JSON report (season, week, player count, upload bytes, duration) under `data/logs/`.
4. Only then wire cron or Actions.

## Recommendation
Keep the manual/local-cron until a thin ballnet refresh wrapper exists; only then consider GitHub Actions. Prefer Tuesday AM (America/New_York) so nflverse weeklies have landed.