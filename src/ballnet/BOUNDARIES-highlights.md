# Boundaries: `ballnet/highlights` (Stage H)

## Always

- Read weekly values from Stage B spine only — never Stage C YTD or Stage G page JSON.
- Score only the curated allowlist in `_allowlist_for_group` (volumes + clear single-game rates).
- Orient z-scores with catalog `higherIsBetter` via `ballnet.scoring` against the **season-of-games** sample (`week <= W`), not same-week peers alone.
- Require `peerN >= MIN_PEER_N` (16) before emitting a row; apply volume floors for KDE + z; apply `min_value` only when selecting board rows (rare-event spam), not when building the KDE sample.
- Write `data/highlights/{season}/w{week}.json` and allowlist curves under `data/dists/league_weekly/{season}/w{week}/{group}.json` (`scope: "league_weekly"`). Do not embed `curve[]` on board rows.

## Ask First

- Expanding the allowlist to noisy weekly rates (snap %, cushion, separation).
- Changing global-top balancing across position groups.
- Adding season / all-time boards (separate publish paths).

## Never

- Fold highlight math into Stage G `publish.py` or reuse `league/` YTD curves for home expand charts.
- Invent Knowball catalog ids here.
- Emit boards for `returner` / empty OL / punter without spine-backed columns.

## Silent Failures & Gotchas

- Missing spine file raises; empty week after filters yields `top: []` (Knowball shows pending).
- `peerN` is the count of qualified player-weeks 1..W, not same-week peers — week 1 matches the old same-week z; later weeks diverge.
- `oneInN` is Gaussian-tail of oriented z — for later copy, not home UI yet.
- Discrete rare events still need `min_value` floors on the board; zeros/low counts stay in the KDE sample so the shape is honest.
- `--highlights` upload must include both the board and `dists/league_weekly/` or home expand charts stay pending.
