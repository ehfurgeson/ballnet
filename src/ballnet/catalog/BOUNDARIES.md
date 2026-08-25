# Boundaries: `src/ballnet/catalog`

## Always

- Keep `stat.id` strings identical to knowball `web/src/lib/catalog/*.ts` (and the ETL brief).
- Put machine denoms on `StatDefinition.denom` (e.g. `ngs_rush_weeks`, `targets_allowed`) — Stage C maps those keys on the wide panel.
- Register every group in `registry.STATS_BY_GROUP` / `POSITION_GROUPS`.

## Ask First

- Adding or renaming a `stat.id` (must start as a knowball catalog change).
- Changing domain / `higherIsBetter` without mirroring knowball. Catalog `kind` / `binWidth` are formatting metadata; they do not select a density shape.

## Never

- Invent slider rows that knowball does not list.
- Impute 0 for missing NGS/PFR/snap sources via catalog defaults.

## Silent Failures & Gotchas

- OL pass-protection ids stay `always_unavailable` and are omitted from Stage C output; Knowball still grays them from its catalog.
- Play-grain ids (`red_zone_*`, `route_pct`) exist in catalogs but Stage C always marks `missing_source` until PBP/participation land on the spine.
- Returner group has no spine rows today (nflverse positions are WR/RB primary, not KR/PR).
