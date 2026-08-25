"""Special-teams catalog ids — must match knowball `web/src/lib/catalog/special-teams.ts`."""

from __future__ import annotations

from ballnet.catalog.types import StatDefinition

KICKER_STATS: list[StatDefinition] = [
    StatDefinition("fg_attempts", "discrete", True, "count", 0, 8, 1, "fg_att", 1999, bin_width=1),
    StatDefinition("fg_made", "discrete", True, "count", 0, 8, 1, "fg_att", 1999, bin_width=1),
    StatDefinition(
        "fg_pct",
        "continuous",
        True,
        "percent",
        0,
        1,
        1,
        "fg_att",
        1999,
        lower_bound=0,
        upper_bound=1,
    ),
    StatDefinition("fg_long", "discrete", True, "yards", 0, 70, 1, "fg_att", 1999, bin_width=5),
    StatDefinition("fg_40_49", "discrete", True, "count", 0, 4, 1, "fg_att", 1999, bin_width=1),
    StatDefinition("xp_attempts", "discrete", True, "count", 0, 10, 1, "pat_att", 1999, bin_width=1),
    StatDefinition("xp_made", "discrete", True, "count", 0, 10, 1, "pat_att", 1999, bin_width=1),
]

PUNTER_STATS: list[StatDefinition] = [
    StatDefinition("punts", "discrete", True, "count", 0, 16, 2, "punts", 1999, bin_width=1),
    StatDefinition(
        "gross_punt_yards", "continuous", True, "yards", 0, 685, 2, "punts", 1999, lower_bound=0
    ),
    StatDefinition(
        "net_punt_yards", "continuous", True, "yards", 0, 600, 2, "punts", 1999, lower_bound=0
    ),
    StatDefinition("inside_20", "discrete", True, "count", 0, 8, 2, "punts", 1999, bin_width=1),
    StatDefinition("touchbacks", "discrete", False, "count", 0, 6, 2, "punts", 1999, bin_width=1),
    StatDefinition("fair_catches", "discrete", True, "count", 0, 8, 2, "punts", 1999, bin_width=1),
]

RETURNER_STATS: list[StatDefinition] = [
    StatDefinition(
        "kick_returns", "discrete", True, "count", 0, 11, 1, "kick_returns", 1999, bin_width=1
    ),
    StatDefinition(
        "punt_returns", "discrete", True, "count", 0, 11, 1, "punt_returns", 1999, bin_width=1
    ),
    StatDefinition(
        "kick_return_yards",
        "continuous",
        True,
        "yards",
        -10,
        305,
        1,
        "kick_returns",
        1999,
        lower_bound=0,
    ),
    StatDefinition(
        "punt_return_yards",
        "continuous",
        True,
        "yards",
        -10,
        200,
        1,
        "punt_returns",
        1999,
        lower_bound=0,
    ),
    StatDefinition("return_tds", "discrete", True, "count", 0, 2, 1, "returns", 1999, bin_width=1),
]

KICKER_STAT_IDS = [s.id for s in KICKER_STATS]
PUNTER_STAT_IDS = [s.id for s in PUNTER_STATS]
RETURNER_STAT_IDS = [s.id for s in RETURNER_STATS]
