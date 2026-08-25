"""QB catalog ids — must match knowball `web/src/lib/catalog/qb.ts`."""

from __future__ import annotations

from ballnet.catalog.types import StatDefinition

QB_STATS: list[StatDefinition] = [
    StatDefinition("pass_attempts", "discrete", True, "count", 0, 70, 8, "attempts", 1999, bin_width=1),
    StatDefinition("completions", "discrete", True, "count", 0, 45, 8, "attempts", 1999, bin_width=1),
    StatDefinition(
        "passing_yards", "continuous", True, "yards", -10, 554, 8, "attempts", 1999, lower_bound=0
    ),
    StatDefinition("passing_tds", "discrete", True, "count", 0, 7, 1, "games_with_attempts", 1999, bin_width=1),
    StatDefinition(
        "interceptions", "discrete", False, "count", 0, 7, 1, "games_with_attempts", 1999, bin_width=1
    ),
    StatDefinition(
        "sacks_taken", "discrete", False, "count", 0, 12, 1, "games_with_attempts", 1999, bin_width=1
    ),
    StatDefinition(
        "completion_pct",
        "continuous",
        True,
        "percent",
        0,
        1,
        10,
        "attempts",
        1999,
        lower_bound=0,
        upper_bound=1,
    ),
    StatDefinition(
        "passer_rating",
        "continuous",
        True,
        "rating",
        0,
        158.3,
        10,
        "attempts",
        1999,
        lower_bound=0,
        upper_bound=158.3,
    ),
    StatDefinition("cpoe", "continuous", True, "percent_pts", -25, 25, 10, "attempts", 1999),
    StatDefinition("passing_epa", "continuous", True, "one_decimal", -40, 40, 8, "attempts", 1999),
    StatDefinition("pacr", "continuous", True, "ratio", 0, 3, 8, "passing_air_yards", 1999, lower_bound=0),
    StatDefinition(
        "time_to_throw",
        "continuous",
        False,
        "seconds",
        1.5,
        6,
        1,
        "ngs_weeks",
        2016,
        lower_bound=1.5,
    ),
    StatDefinition("iay", "continuous", True, "one_decimal", 0, 25, 1, "ngs_weeks", 2016, lower_bound=0),
    StatDefinition("cay", "continuous", True, "one_decimal", 0, 20, 1, "ngs_weeks", 2016, lower_bound=0),
    StatDefinition(
        "aggressiveness",
        "continuous",
        True,
        "percent",
        0,
        1,
        1,
        "ngs_weeks",
        2016,
        lower_bound=0,
        upper_bound=1,
    ),
    StatDefinition(
        "expected_completion_pct",
        "continuous",
        True,
        "percent",
        0,
        1,
        1,
        "ngs_weeks",
        2016,
        lower_bound=0,
        upper_bound=1,
    ),
    StatDefinition("air_yards_diff", "continuous", True, "one_decimal", -10, 8, 1, "ngs_weeks", 2016),
    StatDefinition(
        "air_yards_to_sticks", "continuous", True, "one_decimal", -8, 8, 1, "ngs_weeks", 2016
    ),
    StatDefinition("deep_attempts", "discrete", True, "count", 0, 20, 8, "attempts", 1999, bin_width=1),
    StatDefinition("rush_attempts", "discrete", True, "count", 0, 20, 3, "carries", 1999, bin_width=1),
    StatDefinition(
        "rushing_yards", "continuous", True, "yards", -15, 150, 3, "carries", 1999, lower_bound=0
    ),
    StatDefinition("rushing_tds", "discrete", True, "count", 0, 4, 3, "carries", 1999, bin_width=1),
    StatDefinition(
        "offensive_snap_pct",
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

QB_STAT_IDS = [s.id for s in QB_STATS]
