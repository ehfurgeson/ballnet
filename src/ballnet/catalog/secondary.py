"""Secondary catalog ids — must match knowball `web/src/lib/catalog/secondary.ts`."""

from __future__ import annotations

from ballnet.catalog.types import StatDefinition, with_volume

SECONDARY_STATS: list[StatDefinition] = [
    StatDefinition(
        "rating_allowed",
        "continuous",
        False,
        "rating",
        0,
        158.3,
        4,
        "targets_allowed",
        2018,
        lower_bound=0,
        upper_bound=158.3,
    ),
    StatDefinition(
        "yards_per_target_allowed",
        "continuous",
        False,
        "one_decimal",
        0,
        20,
        4,
        "targets_allowed",
        2018,
        lower_bound=0,
    ),
    StatDefinition(
        "completion_pct_allowed",
        "continuous",
        False,
        "percent",
        0,
        1,
        4,
        "targets_allowed",
        2018,
        lower_bound=0,
        upper_bound=1,
    ),
    StatDefinition(
        "td_rate_allowed",
        "continuous",
        False,
        "percent",
        0,
        1,
        4,
        "targets_allowed",
        2018,
        lower_bound=0,
        upper_bound=1,
    ),
    StatDefinition(
        "adot_allowed",
        "continuous",
        False,
        "one_decimal",
        0,
        30,
        4,
        "targets_allowed",
        2018,
        lower_bound=0,
    ),
    StatDefinition(
        "tds_allowed", "discrete", False, "count", 0, 3, 2, "targets_allowed", 2018, bin_width=1
    ),
    StatDefinition(
        "receiving_yards_allowed",
        "continuous",
        False,
        "yards",
        0,
        250,
        2,
        "targets_allowed",
        2018,
        lower_bound=0,
    ),
    StatDefinition(
        "completions_allowed",
        "discrete",
        False,
        "count",
        0,
        15,
        2,
        "targets_allowed",
        2018,
        bin_width=1,
    ),
    StatDefinition(
        "targets_allowed", "discrete", False, "count", 0, 18, 2, "targets_allowed", 2018, bin_width=1
    ),
    StatDefinition("interceptions", "discrete", True, "count", 0, 4, 1, "games", 1999, bin_width=1),
    StatDefinition(
        "passes_defended", "discrete", True, "count", 0, 6, 1, "games", 1999, bin_width=1
    ),
    StatDefinition(
        "missed_tackle_rate",
        "continuous",
        False,
        "percent",
        0,
        1,
        1,
        "tackle_chances",
        2018,
        lower_bound=0,
        upper_bound=1,
    ),
    StatDefinition(
        "missed_tackles", "discrete", False, "count", 0, 7, 1, "games", 2018, bin_width=1
    ),
    StatDefinition(
        "tackles_combined", "discrete", True, "count", 0, 16, 1, "games", 1999, bin_width=1
    ),
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


SECONDARY_STATS = with_volume(SECONDARY_STATS, {
    "adot_allowed": "targets_allowed",
    "completion_pct_allowed": "targets_allowed",
    "completions_allowed": "targets_allowed",
    "rating_allowed": "targets_allowed",
    "receiving_yards_allowed": "targets_allowed",
    "targets_allowed": "targets_allowed",
    "td_rate_allowed": "targets_allowed",
    "tds_allowed": "targets_allowed",
    "yards_per_target_allowed": "targets_allowed",
})

SECONDARY_STAT_IDS = [s.id for s in SECONDARY_STATS]
