# Boundaries: `src/ballnet/fantasy_rank`

## Always

- Emit ranks only for **QB / WR / RB / TE**. Omit `fantasyPosRank` / `fantasyPosRankKind` for every other position (FB, OL, defense, K, P).
- Live season (`updating_current` or `index/current.json` season) uses FantasyPros weekly ECR (`kind: "consensus"`). Every other season uses REG PPR finish (`kind: "finish"`).
- Join weekly ECR `id` / `fantasypros_id` → GSIS via `ff_playerids.fantasypros_id`. Null GSIS is a skip, not a guessed rank.
- Finish rank is competition rank within modal weekly box position (`map_position`): ties share the min (1, 2, 2, 4). `fantasy_points_ppr == 0` gets no rank.
- Build the table **once** per `(season, as_of_week, kind)` and pass it into pages, leaderboards, and highlights.

## Ask First

- Adding dynasty / overall / superflex ECR, kicker ranks, or depth-chart ranks.
- Changing kind resolution so a closed season stays on consensus.

## Never

- Invent a rank when ECR, GSIS, or PPR is missing.
- Put these fields on `index/players.json` (latest-only bios).
- Overload highlight board `rank` (that is z-score order).
- Compute ranks in Knowball.

## Silent Failures & Gotchas

- Weekly `load_ff_rankings("week")` often has **no** `ecr_type`. Filter `ecr_type == "rp"` only when that column has values; otherwise keep `pos` in {QB,RB,WR,TE} and drop dynasty/overall/superflex page strings.
- `round(ecr)` is the integer subscript; Python/Polars bankers rounding applies at `.5`.
- Finish uses already-fetched `player_stats_{season}` (REG). Modal position can disagree with the player's current roster code (e.g. WR/RB flex weeks).
- Consensus fetch failures print and return `{}` — pages publish without the fields rather than aborting Stage G.
- Season rollover: when `index/current.json` advances to `Y+1`, year `Y` must be republished at its final REG week or last year’s pages keep `kind: consensus`.
