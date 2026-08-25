"""NFL passer rating from box components (not average of weekly ratings)."""

from __future__ import annotations


def _component(value: float) -> float:
    return max(0.0, min(2.375, value))


def passer_rating(
    completions: float,
    attempts: float,
    yards: float,
    touchdowns: float,
    interceptions: float,
) -> float | None:
    if attempts <= 0:
        return None
    a = _component(((completions / attempts) - 0.3) * 5.0)
    b = _component(((yards / attempts) - 3.0) * 0.25)
    c = _component((touchdowns / attempts) * 20.0)
    d = _component(2.375 - ((interceptions / attempts) * 25.0))
    return ((a + b + c + d) / 6.0) * 100.0
