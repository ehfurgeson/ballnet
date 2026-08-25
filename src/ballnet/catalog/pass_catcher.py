"""Pass-catcher catalog ids — must match knowball `web/src/lib/catalog/pass-catcher.ts`."""

from __future__ import annotations

from ballnet.catalog.types import StatDefinition

PASS_CATCHER_STATS: list[StatDefinition] = [
    StatDefinition("targets", "discrete", True, "count", 0, 21, 3, "targets", 1999, bin_width=1),
    StatDefinition("receptions", "discrete", True, "count", 0, 21, 3, "targets", 1999, bin_width=1),
    StatDefinition(
        "receiving_yards", "continuous", True, "yards", -10, 336, 3, "targets", 1999, lower_bound=0
    ),
    StatDefinition("receiving_tds", "discrete", True, "count", 0, 5, 1, "games", 1999, bin_width=1),
    StatDefinition("drops", "discrete", False, "count", 0, 5, 3, "targets", 2018, bin_width=1),
    StatDefinition(
        "yac", "continuous", True, "yards", 0, 175, 2, "receptions", 1999, lower_bound=0
    ),
    StatDefinition(
        "avg_yac", "continuous", True, "one_decimal", 0, 20, 1, "ngs_rec_weeks", 2016, lower_bound=0
    ),
    StatDefinition(
        "expected_yac",
        "continuous",
        True,
        "one_decimal",
        0,
        20,
        1,
        "ngs_rec_weeks",
        2016,
        lower_bound=0,
    ),
    StatDefinition("yac_oe", "continuous", True, "one_decimal", -8, 8, 1, "ngs_rec_weeks", 2016),
    StatDefinition(
        "separation",
        "continuous",
        True,
        "one_decimal",
        0,
        9,
        1,
        "ngs_rec_weeks",
        2016,
        lower_bound=0,
    ),
    StatDefinition(
        "cushion", "continuous", True, "one_decimal", 0, 15, 1, "ngs_rec_weeks", 2016, lower_bound=0
    ),
    StatDefinition("adot", "continuous", True, "one_decimal", 0, 35, 3, "targets", 1999, lower_bound=0),
    StatDefinition(
        "catch_pct",
        "continuous",
        True,
        "percent",
        0,
        1,
        3,
        "targets",
        1999,
        lower_bound=0,
        upper_bound=1,
    ),
    StatDefinition(
        "target_share",
        "continuous",
        True,
        "percent",
        0,
        0.5,
        3,
        "targets",
        1999,
        lower_bound=0,
        upper_bound=0.5,
    ),
    StatDefinition(
        "air_yards_share",
        "continuous",
        True,
        "percent",
        0,
        0.6,
        3,
        "targets",
        1999,
        lower_bound=0,
        upper_bound=0.6,
    ),
    StatDefinition("wopr", "continuous", True, "ratio", 0, 1, 3, "targets", 1999, lower_bound=0),
    StatDefinition(
        "racr", "continuous", True, "ratio", 0, 3, 3, "receiving_air_yards", 1999, lower_bound=0
    ),
    StatDefinition("receiving_epa", "continuous", True, "one_decimal", -15, 20, 3, "targets", 1999),
    StatDefinition(
        "red_zone_targets", "discrete", True, "count", 0, 10, 1, "games", 1999, bin_width=1
    ),
    # Participation / FTN not exploded onto spine yet
    StatDefinition(
        "route_pct",
        "continuous",
        True,
        "percent",
        0,
        1,
        1,
        "snap_weeks",
        2016,
        lower_bound=0,
        upper_bound=1,
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

PASS_CATCHER_STAT_IDS = [s.id for s in PASS_CATCHER_STATS]
