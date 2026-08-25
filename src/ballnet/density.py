"""Stage D — league_ytd densities (reflected Gaussian KDE for every catalog id)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from ballnet.catalog.registry import stats_for_group
from ballnet.catalog.types import StatDefinition
from ballnet.paths import DIST_DIR, YTD_DIR, ensure_data_dirs

GRID_N = 512


@dataclass(frozen=True)
class DensityResult:
    path: str
    rows: int
    seconds: float


def _kde_cdf(curve: list[dict[str, float]], value: float) -> float:
    """Inclusive CDF matching knowball `kdeCdf` (trapezoid / total mass)."""
    if len(curve) < 2:
        return 0.0
    mass = 0.0
    total = 0.0
    for i in range(1, len(curve)):
        left, right = curve[i - 1], curve[i]
        slice_ = ((left["y"] + right["y"]) / 2.0) * (right["x"] - left["x"])
        total += slice_
        if right["x"] <= value:
            mass += slice_
        elif left["x"] < value:
            t = (value - left["x"]) / (right["x"] - left["x"])
            y = left["y"] + t * (right["y"] - left["y"])
            mass += ((left["y"] + y) / 2.0) * (value - left["x"])
    if total <= 0:
        return 0.0
    return float(min(1.0, max(0.0, mass / total)))


def oriented_percentile(
    *,
    higher_is_better: bool,
    player_value: float,
    curve: list[dict[str, float]] | None,
) -> float:
    p = _kde_cdf(curve or [], player_value)
    pct = 100.0 * p if higher_is_better else 100.0 * (1.0 - p)
    return float(min(100.0, max(0.0, pct)))


def _expand_domain(
    x_min: float,
    x_max: float,
    sample: np.ndarray,
) -> tuple[float, float]:
    """Keep catalog domain unless the qualified sample truly exceeds it."""
    if sample.size == 0:
        return x_min, x_max
    lo = float(np.min(sample))
    hi = float(np.max(sample))
    return min(x_min, lo), max(x_max, hi)


def _reflected_kde_curve(
    sample: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    lower_bound: float | None,
    upper_bound: float | None,
    grid_n: int = GRID_N,
) -> list[dict[str, float]]:
    """Gaussian KDE with reflection at catalog walls; ∫y dx ≈ 1 on [x_min, x_max]."""
    xs = np.linspace(x_min, x_max, grid_n)
    if sample.size == 0:
        return [{"x": float(x), "y": 0.0} for x in xs]
    if sample.size == 1 or float(np.std(sample)) < 1e-12:
        # Degenerate: put a narrow bump at the single value (still integrate ~1).
        y = np.zeros_like(xs)
        idx = int(np.argmin(np.abs(xs - sample[0])))
        if grid_n > 1:
            dx = float(xs[1] - xs[0])
            y[idx] = 1.0 / dx
        else:
            y[0] = 1.0
        return [{"x": float(x), "y": float(yy)} for x, yy in zip(xs, y)]

    kde = stats.gaussian_kde(sample)

    def density_at(x: np.ndarray) -> np.ndarray:
        dens = kde(x)
        if lower_bound is not None:
            dens = dens + kde(2.0 * lower_bound - x)
        if upper_bound is not None:
            dens = dens + kde(2.0 * upper_bound - x)
        return dens

    y = density_at(xs)
    # Zero density outside reflection walls when walls sit inside the plot domain.
    if lower_bound is not None:
        y = np.where(xs < lower_bound, 0.0, y)
    if upper_bound is not None:
        y = np.where(xs > upper_bound, 0.0, y)

    # Normalize on the plotted grid (trapezoid), matching how the UI integrates.
    integral = float(np.trapezoid(y, xs))
    if integral > 0:
        y = y / integral
    return [{"x": float(x), "y": float(yy)} for x, yy in zip(xs, y)]


def build_league_density(
    sample: np.ndarray,
    stat: StatDefinition,
) -> dict[str, Any]:
    """Reflected KDE for every catalog id. `kind` is catalog metadata, not a shape switch."""
    x_min, x_max = _expand_domain(stat.x_min, stat.x_max, sample)
    n_sample = int(sample.size)
    curve = _reflected_kde_curve(
        sample,
        x_min=x_min,
        x_max=x_max,
        lower_bound=stat.lower_bound,
        upper_bound=stat.upper_bound,
    )
    y_max = max((p["y"] for p in curve), default=0.0)
    return {
        "stat_id": stat.id,
        "kind": stat.kind,
        "x_min": x_min,
        "x_max": x_max,
        "y_max": float(y_max),
        "lower_bound": stat.lower_bound,
        "upper_bound": stat.upper_bound,
        "curve": curve,
        "n_sample": n_sample,
    }


def build_densities(
    season: int,
    as_of_week: int,
    *,
    position_group: str = "qb",
) -> DensityResult:
    """Build league_ytd densities for a position group from Stage C parquet."""
    if position_group not in {
        "qb",
        "backfield",
        "pass_catcher",
        "ol",
        "def_front",
        "secondary",
        "kicker",
        "punter",
        "returner",
    }:
        raise NotImplementedError(f"densities unknown group {position_group}")
    ensure_data_dirs()
    t0 = time.perf_counter()

    ytd_path = YTD_DIR / f"ytd_{position_group}_{season}_w{as_of_week}.parquet"
    if not ytd_path.exists():
        raise FileNotFoundError(
            f"Missing Stage C panel {ytd_path}; run `ballnet ytd` first"
        )
    long = pl.read_parquet(ytd_path)
    stats = stats_for_group(position_group)

    records: list[dict[str, Any]] = []
    for stat in stats:
        if stat.always_unavailable:
            continue
        sample_df = long.filter(
            (pl.col("stat_id") == stat.id)
            & pl.col("qualified")
            & pl.col("player_value").is_not_null()
        )
        sample = sample_df["player_value"].to_numpy().astype(float)
        dens = build_league_density(sample, stat)
        dens.update(
            {
                "season": season,
                "as_of_week": as_of_week,
                "position_group": position_group,
            }
        )
        records.append(dens)

    out_dir = DIST_DIR / f"league_ytd_{position_group}_{season}_w{as_of_week}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "distributions.json"
    with json_path.open("w") as f:
        json.dump(records, f)

    slim = pl.DataFrame(
        [
            {
                "season": r["season"],
                "as_of_week": r["as_of_week"],
                "position_group": r["position_group"],
                "stat_id": r["stat_id"],
                "kind": r["kind"],
                "x_min": r["x_min"],
                "x_max": r["x_max"],
                "y_max": r["y_max"],
                "n_sample": r["n_sample"],
            }
            for r in records
        ]
    )
    slim.write_parquet(out_dir / "index.parquet")

    return DensityResult(
        path=str(json_path),
        rows=len(records),
        seconds=time.perf_counter() - t0,
    )


def load_densities(
    season: int,
    as_of_week: int,
    *,
    position_group: str = "qb",
) -> dict[str, dict[str, Any]]:
    path = DIST_DIR / f"league_ytd_{position_group}_{season}_w{as_of_week}" / "distributions.json"
    with path.open() as f:
        records = json.load(f)
    return {r["stat_id"]: r for r in records}
