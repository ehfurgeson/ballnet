"""Catalog registry — position_group → StatDefinition list."""

from __future__ import annotations

from ballnet.catalog.backfield import BACKFIELD_STAT_IDS, BACKFIELD_STATS
from ballnet.catalog.def_front import DEF_FRONT_STAT_IDS, DEF_FRONT_STATS
from ballnet.catalog.ol import OL_STAT_IDS, OL_STATS
from ballnet.catalog.pass_catcher import PASS_CATCHER_STAT_IDS, PASS_CATCHER_STATS
from ballnet.catalog.qb import QB_STAT_IDS, QB_STATS
from ballnet.catalog.secondary import SECONDARY_STAT_IDS, SECONDARY_STATS
from ballnet.catalog.special_teams import (
    KICKER_STAT_IDS,
    KICKER_STATS,
    PUNTER_STAT_IDS,
    PUNTER_STATS,
    RETURNER_STAT_IDS,
    RETURNER_STATS,
)
from ballnet.catalog.types import StatDefinition

POSITION_GROUPS: tuple[str, ...] = (
    "qb",
    "backfield",
    "pass_catcher",
    "ol",
    "def_front",
    "secondary",
    "kicker",
    "punter",
    "returner",
)

STATS_BY_GROUP: dict[str, list[StatDefinition]] = {
    "qb": QB_STATS,
    "backfield": BACKFIELD_STATS,
    "pass_catcher": PASS_CATCHER_STATS,
    "ol": OL_STATS,
    "def_front": DEF_FRONT_STATS,
    "secondary": SECONDARY_STATS,
    "kicker": KICKER_STATS,
    "punter": PUNTER_STATS,
    "returner": RETURNER_STATS,
}

STAT_IDS_BY_GROUP: dict[str, list[str]] = {
    "qb": QB_STAT_IDS,
    "backfield": BACKFIELD_STAT_IDS,
    "pass_catcher": PASS_CATCHER_STAT_IDS,
    "ol": OL_STAT_IDS,
    "def_front": DEF_FRONT_STAT_IDS,
    "secondary": SECONDARY_STAT_IDS,
    "kicker": KICKER_STAT_IDS,
    "punter": PUNTER_STAT_IDS,
    "returner": RETURNER_STAT_IDS,
}


def stats_for_group(position_group: str) -> list[StatDefinition]:
    try:
        return STATS_BY_GROUP[position_group]
    except KeyError as e:
        raise KeyError(f"unknown position_group {position_group!r}") from e
