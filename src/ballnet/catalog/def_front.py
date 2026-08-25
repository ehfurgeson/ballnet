"""Defensive-front catalog ids — must match knowball `web/src/lib/catalog/def-front.ts`."""

from __future__ import annotations

from ballnet.catalog.types import StatDefinition

DEF_FRONT_STATS: list[StatDefinition] = [
    StatDefinition("tackles_solo", "discrete", True, "count", 0, 16, 1, "games", 1999, bin_width=1),
    StatDefinition("tackles_ast", "discrete", True, "count", 0, 12, 1, "games", 1999, bin_width=1),
    StatDefinition(
        "tackles_combined", "discrete", True, "count", 0, 24, 1, "games", 1999, bin_width=1
    ),
    StatDefinition(
        "missed_tackles", "discrete", False, "count", 0, 7, 1, "games", 2018, bin_width=1
    ),
    StatDefinition("sacks", "discrete", True, "one_decimal", 0, 7, 1, "games", 1999, bin_width=0.5),
    StatDefinition(
        "tackles_for_loss", "discrete", True, "count", 0, 6, 1, "games", 1999, bin_width=1
    ),
    StatDefinition("qb_hits", "discrete", True, "count", 0, 12, 1, "games", 1999, bin_width=1),
    StatDefinition("pressures", "discrete", True, "count", 0, 15, 1, "games", 2018, bin_width=1),
    StatDefinition("hurries", "discrete", True, "count", 0, 10, 1, "games", 2018, bin_width=1),
    StatDefinition(
        "forced_fumbles", "discrete", True, "count", 0, 4, 1, "games", 1999, bin_width=1
    ),
    StatDefinition("interceptions", "discrete", True, "count", 0, 3, 1, "games", 1999, bin_width=1),
    StatDefinition(
        "defensive_snap_pct",
        "continuous",
        True,
        "percent",
        0,
        1,
        1,
        "snap_weeks",
        2012,
        lower_bound=0,
        upper_bound=1,
    ),
]

DEF_FRONT_STAT_IDS = [s.id for s in DEF_FRONT_STATS]
