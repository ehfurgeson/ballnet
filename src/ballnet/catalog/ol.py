"""OL catalog ids — must match knowball `web/src/lib/catalog/ol.ts`."""

from __future__ import annotations

from ballnet.catalog.types import StatDefinition

OL_STATS: list[StatDefinition] = [
    StatDefinition("snaps_played", "discrete", True, "count", 0, 100, 1, "snap_weeks", 2012, bin_width=5),
    StatDefinition(
        "snap_pct",
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
    StatDefinition(
        "sacks_allowed",
        "discrete",
        False,
        "count",
        0,
        6,
        None,
        "pass_snaps",
        None,
        bin_width=1,
        always_unavailable=True,
    ),
    StatDefinition(
        "pressures_allowed",
        "discrete",
        False,
        "count",
        0,
        12,
        None,
        "pass_snaps",
        None,
        bin_width=1,
        always_unavailable=True,
    ),
    StatDefinition(
        "block_win_rate",
        "continuous",
        True,
        "percent",
        0,
        1,
        None,
        "pass_snaps",
        None,
        lower_bound=0,
        upper_bound=1,
        always_unavailable=True,
    ),
    # Denom is weeks present on the spine until OL PFR/snap joins land (currently null).
    StatDefinition("penalties", "discrete", False, "count", 0, 6, 1, "games", 1999, bin_width=1),
    StatDefinition(
        "penalty_yards", "discrete", False, "yards", 0, 50, 1, "games", 1999, bin_width=5
    ),
]

OL_STAT_IDS = [s.id for s in OL_STATS]
