"""Locked ramp–hold: min_n = n_base * min(w, 4)."""

from __future__ import annotations


def min_n(n_base: int | float, as_of_week: int) -> float:
    """as_of_week is the NFL REG week being viewed, not weeks since debut."""
    if as_of_week < 1:
        raise ValueError(f"as_of_week must be >= 1, got {as_of_week}")
    return float(n_base) * min(as_of_week, 4)
