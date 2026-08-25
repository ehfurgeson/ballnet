"""Stage E — oriented percentiles on Stage C rows using league densities."""

from __future__ import annotations

import time
from dataclasses import dataclass

import polars as pl

from ballnet.catalog.registry import stats_for_group
from ballnet.density import load_densities, oriented_percentile
from ballnet.paths import YTD_DIR, ensure_data_dirs


@dataclass(frozen=True)
class PercentileResult:
    path: str
    rows: int
    seconds: float


def attach_percentiles(
    season: int,
    as_of_week: int,
    *,
    position_group: str = "qb",
) -> PercentileResult:
    """Write Stage C panel with `percentile` filled from league_ytd densities."""
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
        raise NotImplementedError(f"percentiles unknown group {position_group}")
    ensure_data_dirs()
    t0 = time.perf_counter()

    ytd_path = YTD_DIR / f"ytd_{position_group}_{season}_w{as_of_week}.parquet"
    long = pl.read_parquet(ytd_path)
    dens_by_id = load_densities(season, as_of_week, position_group=position_group)
    higher = {s.id: s.higher_is_better for s in stats_for_group(position_group)}

    percentiles: list[float | None] = []
    for row in long.to_dicts():
        dens = dens_by_id.get(row["stat_id"])
        value = row.get("player_value")
        if (
            dens is None
            or value is None
            or not row.get("qualified")
            or dens["n_sample"] < 1
        ):
            percentiles.append(None)
            continue
        percentiles.append(
            oriented_percentile(
                higher_is_better=higher[row["stat_id"]],
                player_value=float(value),
                curve=dens.get("curve"),
            )
        )

    out = long.with_columns(pl.Series("percentile", percentiles, dtype=pl.Float64))
    out_path = YTD_DIR / f"ytd_{position_group}_{season}_w{as_of_week}_pct.parquet"
    out.write_parquet(out_path)
    return PercentileResult(
        path=str(out_path),
        rows=out.height,
        seconds=time.perf_counter() - t0,
    )
