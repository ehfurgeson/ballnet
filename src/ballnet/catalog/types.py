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
