# Ballnet

Football data pipeline for [Knowball](https://github.com/ehfurgeson/knowball).

Ballnet ingests [nflverse](https://github.com/nflverse) data, qualifies players, estimates league distributions with kernel density estimation, computes oriented percentiles, and publishes JSON that Knowball renders. Coverage spans the NGS era (2016–present).

**Stack:** Python · uv · Polars/pandas · scikit-learn · Supabase Storage

## Setup

```bash
uv sync
```

## Pipeline

```bash
# Backfill raw + weekly spine
uv run ballnet backfill --start 2016 --end 2025

# Publish one season (players, index, densities, percentiles)
uv run ballnet publish-all --season 2025 --as-of-week 18

# Multi-season publish + merged search index
uv run ballnet publish-range --start 2016 --end 2025

# Upload to Supabase Storage
uv run ballnet upload-storage --index --season 2025
```

Artifacts land under gitignored `data/` (`raw/`, `spine/`, `ytd/`, `dists/`, `pages/`, `league/`, `index/`).

Weekly refresh notes: [`docs/WEEKLY_OPS.md`](docs/WEEKLY_OPS.md).

## Related

- [knowball](https://github.com/ehfurgeson/knowball) — Next.js visualization site
