"""Backfield catalog ids — must match knowball `web/src/lib/catalog/backfield.ts`."""

from __future__ import annotations

from ballnet.catalog.types import StatDefinition

BACKFIELD_STATS: list[StatDefinition] = [
    StatDefinition("rush_attempts", "discrete", True, "count", 0, 45, 5, "carries", 1999, bin_width=1),
    StatDefinition(
        "rushing_yards", "continuous", True, "yards", -15, 296, 5, "carries", 1999, lower_bound=0
    ),
    StatDefinition(
        "rushing_tds", "discrete", True, "count", 0, 6, 1, "games_with_carries", 1999, bin_width=1
    ),
    StatDefinition("yards_per_carry", "continuous", True, "two_decimal", -5, 20, 5, "carries", 1999),
    StatDefinition("rushing_epa", "continuous", True, "one_decimal", -20, 20, 5, "carries", 1999),
    StatDefinition("fumbles", "discrete", False, "count", 0, 7, 1, "touches", 1999, bin_width=1),
    StatDefinition(
        "broken_tackles", "discrete", True, "count", 0, 16, 5, "carries", 2018, bin_width=1
    ),
    StatDefinition(
        "yards_after_contact", "continuous", True, "yards", 0, 175, 5, "carries", 2018, lower_bound=0
    ),
    StatDefinition(
        "time_to_los",
        "continuous",
        False,
        "seconds",
        1.5,
        5,
        1,
        "ngs_rush_weeks",
        2016,
        lower_bound=1.5,
    ),
    StatDefinition("ryoe", "continuous", True, "one_decimal", -10, 15, 1, "ngs_rush_weeks", 2016),
    StatDefinition(
        "eight_plus_defenders_pct",
        "continuous",
        True,
        "percent",
        0,
        1,
        1,
        "ngs_rush_weeks",
        2016,
        lower_bound=0,
        upper_bound=1,
    ),
    StatDefinition(
        "ngs_efficiency", "continuous", True, "ratio", 0, 8, 1, "ngs_rush_weeks", 2016, lower_bound=0
    ),
    StatDefinition(
        "expected_rush_yards",
        "continuous",
        True,
        "yards",
        0,
        200,
        1,
        "ngs_rush_weeks",
        2016,
        lower_bound=0,
    ),
    StatDefinition("targets", "discrete", True, "count", 0, 15, 2, "targets", 1999, bin_width=1),
    StatDefinition("receptions", "discrete", True, "count", 0, 12, 2, "targets", 1999, bin_width=1),
    StatDefinition(
        "receiving_yards", "continuous", True, "yards", -10, 150, 2, "targets", 1999, lower_bound=0
    ),
    StatDefinition(
        "target_share",
        "continuous",
        True,
        "percent",
        0,
        0.4,
        2,
        "targets",
        1999,
        lower_bound=0,
        upper_bound=0.4,
    ),
    StatDefinition("wopr", "continuous", True, "ratio", 0, 0.5, 2, "targets", 1999, lower_bound=0),
    # Play-grain: not on spine yet
    StatDefinition(
        "red_zone_touches", "discrete", True, "count", 0, 12, 1, "games", 1999, bin_width=1
    ),
    StatDefinition(
        "expected_fantasy_points",
        "continuous",
        True,
        "one_decimal",
        0,
        40,
        1,
        "games",
        2006,
        lower_bound=0,
    ),
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

BACKFIELD_STAT_IDS = [s.id for s in BACKFIELD_STATS]
