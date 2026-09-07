"""Catalog types mirroring knowball `web/src/lib/catalog/types.ts`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["continuous", "discrete"]
Unavailable = Literal["insufficient_sample", "missing_source", "not_in_nflverse"]


@dataclass(frozen=True)
class StatDefinition:
    id: str
    kind: Kind
    higher_is_better: bool
    format: str
    x_min: float
    x_max: float
    min_n_base: int | None
    denom: str
    start_year: int | None
    lower_bound: float | None = None
    upper_bound: float | None = None
    bin_width: float | None = None
    always_unavailable: bool = False
    # Knowball search UX only — Stage C still qualifies on machine `denom`.
    volume_stat_id: str | None = None


def with_volume(
    stats: list[StatDefinition], mapping: dict[str, str]
) -> list[StatDefinition]:
    """Attach Knowball UX volume_stat_id without rewriting positional constructors."""
    from dataclasses import replace

    return [
        replace(s, volume_stat_id=mapping[s.id]) if s.id in mapping else s
        for s in stats
    ]


def with_min_n_base(
    stats: list[StatDefinition], mapping: dict[str, int]
) -> list[StatDefinition]:
    """Override ramp–hold n_base (keep in sync with knowball minNBase)."""
    from dataclasses import replace

    return [
        replace(s, min_n_base=mapping[s.id]) if s.id in mapping else s
        for s in stats
    ]
