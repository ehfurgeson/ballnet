"""Special-teams catalog ids — must match knowball `web/src/lib/catalog/special-teams.ts`."""

from __future__ import annotations

from ballnet.catalog.types import StatDefinition, with_min_n_base, with_volume

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
    StatDefinition(
        "net_punt_yards", "continuous", True, "yards", 0, 600, 2, "punts", 1999, lower_bound=0
    ),
    StatDefinition(
        "inside_20_rate",
        "continuous",
        True,
        "percent",
        0,
        1,
        2,
        "punts",
        1999,
        lower_bound=0,
        upper_bound=1,
    ),
    StatDefinition(
        "touchback_rate",
        "continuous",
        False,
        "percent",
        0,
        1,
        2,
        "punts",
        1999,
        lower_bound=0,
        upper_bound=1,
    ),
    StatDefinition(
        "fair_catch_rate",
        "continuous",
        True,
        "percent",
        0,
        1,
        2,
        "punts",
        1999,
        lower_bound=0,
        upper_bound=1,
    ),
    StatDefinition(
        "gross_to_net_loss",
        "continuous",
        False,
        "one_decimal",
        0,
        15,
        2,
        "punts",
        1999,
        lower_bound=0,
    ),
    StatDefinition(
        "gross_punt_yards", "continuous", True, "yards", 0, 685, 2, "punts", 1999, lower_bound=0
    ),
    StatDefinition("inside_20", "discrete", True, "count", 0, 8, 2, "punts", 1999, bin_width=1),
    StatDefinition("fair_catches", "discrete", True, "count", 0, 8, 2, "punts", 1999, bin_width=1),
    StatDefinition("touchbacks", "discrete", False, "count", 0, 6, 2, "punts", 1999, bin_width=1),
    StatDefinition("punts", "discrete", True, "count", 0, 16, 2, "punts", 1999, bin_width=1),
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


KICKER_STATS = with_volume(
    KICKER_STATS,
    {
        "fg_attempts": "fg_attempts",
        "fg_long": "fg_attempts",
        "fg_made": "fg_attempts",
        "fg_pct": "fg_attempts",
        "xp_attempts": "xp_attempts",
        "xp_made": "xp_attempts",
    },
)

PUNTER_STATS = with_volume(
    PUNTER_STATS,
    {
        "fair_catch_rate": "punts",
        "fair_catches": "punts",
        "gross_punt_yards": "punts",
        "gross_to_net_loss": "punts",
        "inside_20": "punts",
        "inside_20_rate": "punts",
        "net_punt_yards": "punts",
        "punts": "punts",
        "touchback_rate": "punts",
        "touchbacks": "punts",
    },
)

RETURNER_STATS = with_volume(
    RETURNER_STATS,
    {
        "kick_return_yards": "kick_returns",
        "kick_returns": "kick_returns",
        "punt_return_yards": "punt_returns",
        "punt_returns": "punt_returns",
    },
)


KICKER_STATS = with_min_n_base(KICKER_STATS, {
    "fg_attempts": 2,
    "fg_long": 2,
    "fg_made": 2,
    "fg_pct": 2,
    "xp_attempts": 2,
    "xp_made": 2,
})

PUNTER_STATS = with_min_n_base(PUNTER_STATS, {
    "fair_catch_rate": 4,
    "fair_catches": 4,
    "gross_punt_yards": 4,
    "gross_to_net_loss": 4,
    "inside_20": 4,
    "inside_20_rate": 4,
    "net_punt_yards": 4,
    "punts": 4,
    "touchback_rate": 4,
    "touchbacks": 4,
})

KICKER_STAT_IDS = [s.id for s in KICKER_STATS]
PUNTER_STAT_IDS = [s.id for s in PUNTER_STATS]
RETURNER_STAT_IDS = [s.id for s in RETURNER_STATS]
