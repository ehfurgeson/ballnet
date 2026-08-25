# Boundaries: `ballnet/highlights` (Stage H)

## Always

- Read weekly values from Stage B spine only — never Stage C YTD or Stage G page JSON.
- Score only the curated allowlist in `_allowlist_for_group` (volumes + clear single-game rates).
- Orient z-scores with catalog `higherIsBetter` via `ballnet.scoring`.
- Require `peerN >= MIN_PEER_N` (16) before emitting a row; apply volume / `min_value` floors so rare flukes do not dominate.
- Write `data/highlights/{season}/w{week}.json` only (Storage prefix `highlights/`).

## Ask First

- Expanding the allowlist to noisy weekly rates (snap %, cushion, separation).
- Changing global-top balancing across position groups.
- Adding season / all-time boards (separate publish paths).

## Never

- Fold highlight math into Stage G `publish.py`.
- Invent Knowball catalog ids here.
- Emit boards for `returner` / empty OL / punter without spine-backed columns.

## Silent Failures & Gotchas

- Missing spine file raises; empty week after filters yields `top: []` (Knowball shows pending).
- `oneInN` is Gaussian-tail of oriented z — for later copy, not home UI yet.
- Discrete rare events still need `min_value` floors; peers include zeros so z stays meaningful.
