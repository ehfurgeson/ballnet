"""Stage C — YTD as-of-week player-stat panel (catalog-driven, all position groups)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable

import polars as pl

from ballnet.catalog.registry import stats_for_group
from ballnet.catalog.types import StatDefinition, Unavailable
from ballnet.paths import SPINE_DIR, YTD_DIR, ensure_data_dirs
from ballnet.ramp_hold import min_n
from ballnet.rating import passer_rating

# NGS weekly averages often arrive 0–100; Knowball percent format is 0–1.
_NGS_PERCENT_COLS = (
    "ngs_pass_aggressiveness",
    "ngs_pass_expected_completion_percentage",
    "ngs_rush_percent_attempts_gte_eight_defenders",
)

# Enrichment columns Stage C aggregates. Older seasons (esp. pre-2018 PFR) may
# omit them from the spine — null-fill so aggregators don't ColumnNotFound.
_OPTIONAL_SPINE_FLOAT_COLS: tuple[str, ...] = (
    # snaps
    "snap_offense_snaps",
    "snap_offense_pct",
    "snap_defense_snaps",
    "snap_defense_pct",
    # PFR rush / rec / def (full advanced set arrives ~2018)
    "pfr_rush_rushing_broken_tackles",
    "pfr_rush_rushing_yards_after_contact",
    "pfr_rec_receiving_drop",
    "pfr_def_def_missed_tackles",
    "pfr_def_def_pressures",
    "pfr_def_def_times_hurried",
    "pfr_def_def_targets",
    "pfr_def_def_completions_allowed",
    "pfr_def_def_yards_allowed",
    "pfr_def_def_receiving_td_allowed",
    "pfr_def_def_passer_rating_allowed",
    "pfr_def_def_adot",
    # NGS (present from 2016 for most; still guard)
    "ngs_pass_avg_time_to_throw",
    "ngs_pass_avg_intended_air_yards",
    "ngs_pass_avg_completed_air_yards",
    "ngs_pass_aggressiveness",
    "ngs_pass_expected_completion_percentage",
    "ngs_pass_avg_air_yards_differential",
    "ngs_pass_avg_air_yards_to_sticks",
    "ngs_pass_completion_percentage_above_expectation",
    "ngs_pass_attempts",
    "ngs_rush_avg_time_to_los",
    "ngs_rush_rush_yards_over_expected_per_att",
    "ngs_rush_percent_attempts_gte_eight_defenders",
    "ngs_rush_efficiency",
    "ngs_rush_expected_rush_yards",
    "ngs_rush_rush_attempts",
    "ngs_rec_avg_yac",
    "ngs_rec_avg_expected_yac",
    "ngs_rec_avg_yac_above_expectation",
    "ngs_rec_avg_separation",
    "ngs_rec_avg_cushion",
    "ngs_rec_targets",
    # fantasy opportunity
    "ffopp_total_fantasy_points_exp",
)


def _ensure_optional_float_cols(panel: pl.DataFrame) -> pl.DataFrame:
    missing = [c for c in _OPTIONAL_SPINE_FLOAT_COLS if c not in panel.columns]
    if not missing:
        return panel
    return panel.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in missing])


@dataclass(frozen=True)
class YtdResult:
    path: str
    rows: int
    players: int
    seconds: float


def _ngs_weighted_mean(value_col: str, weight_col: str) -> pl.Expr:
    """Volume-weighted mean over weeks with non-null value; falls back to unweighted mean."""
    v = pl.col(value_col)
    w = pl.col(weight_col)
    num = pl.when(v.is_not_null() & w.is_not_null()).then(v * w).otherwise(None).sum()
    den = pl.when(v.is_not_null() & w.is_not_null() & (w > 0)).then(w).otherwise(None).sum()
    plain = v.mean()
    return (
        pl.when(den > 0)
        .then(num / den)
        .otherwise(plain)
        .alias(f"{value_col}_ytd")
    )


def _scale_ngs_percents(panel: pl.DataFrame) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for c in _NGS_PERCENT_COLS:
        if c in panel.columns:
            exprs.append(
                pl.when(pl.col(c).is_not_null() & (pl.col(c) > 1.0))
                .then(pl.col(c) / 100.0)
                .otherwise(pl.col(c))
                .alias(c)
            )
    return panel.with_columns(exprs) if exprs else panel


def _with_team_off_snaps(panel: pl.DataFrame) -> pl.DataFrame:
    if "snap_offense_snaps" not in panel.columns or "snap_offense_pct" not in panel.columns:
        return panel
    return panel.with_columns(
        pl.when(
            pl.col("snap_offense_snaps").is_not_null()
            & pl.col("snap_offense_pct").is_not_null()
            & (pl.col("snap_offense_pct") > 0)
        )
        .then(pl.col("snap_offense_snaps") / pl.col("snap_offense_pct"))
        .otherwise(None)
        .alias("_team_off_snaps")
    )


def _with_team_def_snaps(panel: pl.DataFrame) -> pl.DataFrame:
    if "snap_defense_snaps" not in panel.columns or "snap_defense_pct" not in panel.columns:
        return panel
    return panel.with_columns(
        pl.when(
            pl.col("snap_defense_snaps").is_not_null()
            & pl.col("snap_defense_pct").is_not_null()
            & (pl.col("snap_defense_pct") > 0)
        )
        .then(pl.col("snap_defense_snaps") / pl.col("snap_defense_pct"))
        .otherwise(None)
        .alias("_team_def_snaps")
    )


def _share_ytd(player_col: str, share_col: str, alias: str) -> pl.Expr:
    """Rebuild season share as sum(player) / sum(player/share) when weekly share exists."""
    p = pl.col(player_col)
    s = pl.col(share_col)
    team = pl.when(s.is_not_null() & (s > 0) & p.is_not_null()).then(p / s).otherwise(None)
    return (
        pl.when(team.sum() > 0)
        .then(p.sum() / team.sum())
        .otherwise(None)
        .alias(alias)
    )


def _base_keys() -> list[str]:
    return ["player_id", "season", "position_code", "position_group"]


def _meta_aggs() -> list[pl.Expr]:
    return [
        pl.col("player_display_name").last(),
        pl.col("team").last(),
        pl.len().alias("games"),
    ]


def _snap_off_aggs() -> list[pl.Expr]:
    return [
        pl.col("snap_offense_snaps").drop_nulls().len().alias("snap_weeks"),
        pl.col("snap_offense_snaps").sum().alias("offense_snaps"),
        pl.col("_team_off_snaps").sum().alias("team_offense_snaps"),
    ]


def _snap_def_aggs() -> list[pl.Expr]:
    return [
        pl.col("snap_defense_snaps").drop_nulls().len().alias("snap_weeks"),
        pl.col("snap_defense_snaps").sum().alias("defense_snaps"),
        pl.col("_team_def_snaps").sum().alias("team_defense_snaps"),
    ]


def _aggregate_qb_wide(panel: pl.DataFrame) -> pl.DataFrame:
    panel = _with_team_off_snaps(_scale_ngs_percents(panel))
    agg = panel.group_by(_base_keys()).agg(
        *_meta_aggs(),
        pl.col("attempts").sum().alias("attempts"),
        pl.col("completions").sum().alias("completions"),
        pl.col("passing_yards").sum().alias("passing_yards"),
        pl.col("passing_tds").sum().alias("passing_tds"),
        pl.col("interceptions").sum().alias("interceptions"),
        pl.col("sacks_taken").sum().alias("sacks_taken"),
        pl.col("passing_epa").sum().alias("passing_epa"),
        pl.col("passing_air_yards").sum().alias("passing_air_yards"),
        pl.col("passing_20").sum().alias("deep_attempts"),
        pl.col("carries").sum().alias("carries"),
        pl.col("rushing_yards").sum().alias("rushing_yards"),
        pl.col("rushing_tds").sum().alias("rushing_tds"),
        (pl.col("attempts") > 0).sum().alias("games_with_attempts"),
        _ngs_weighted_mean(
            "ngs_pass_completion_percentage_above_expectation", "ngs_pass_attempts"
        ).alias("cpoe_ngs"),
        (
            pl.when(pl.col("passing_cpoe").is_not_null() & (pl.col("attempts") > 0))
            .then(pl.col("passing_cpoe") * pl.col("attempts"))
            .otherwise(None)
            .sum()
            / pl.when(pl.col("passing_cpoe").is_not_null() & (pl.col("attempts") > 0))
            .then(pl.col("attempts"))
            .otherwise(None)
            .sum()
        ).alias("cpoe_box"),
        pl.col("passing_cpoe").drop_nulls().len().alias("cpoe_box_weeks"),
        pl.col("ngs_pass_completion_percentage_above_expectation")
        .drop_nulls()
        .len()
        .alias("cpoe_ngs_weeks"),
        pl.col("ngs_pass_avg_time_to_throw").drop_nulls().len().alias("ngs_weeks"),
        _ngs_weighted_mean("ngs_pass_avg_time_to_throw", "ngs_pass_attempts"),
        _ngs_weighted_mean("ngs_pass_avg_intended_air_yards", "ngs_pass_attempts"),
        _ngs_weighted_mean("ngs_pass_avg_completed_air_yards", "ngs_pass_attempts"),
        _ngs_weighted_mean("ngs_pass_aggressiveness", "ngs_pass_attempts"),
        _ngs_weighted_mean("ngs_pass_expected_completion_percentage", "ngs_pass_attempts"),
        _ngs_weighted_mean("ngs_pass_avg_air_yards_differential", "ngs_pass_attempts"),
        _ngs_weighted_mean("ngs_pass_avg_air_yards_to_sticks", "ngs_pass_attempts"),
        *_snap_off_aggs(),
    )
    return agg.with_columns(
        pl.when(pl.col("attempts") > 0)
        .then(pl.col("completions") / pl.col("attempts"))
        .otherwise(None)
        .alias("completion_pct"),
        pl.when(pl.col("passing_air_yards") > 0)
        .then(pl.col("passing_yards") / pl.col("passing_air_yards"))
        .otherwise(None)
        .alias("pacr"),
        pl.when(pl.col("team_offense_snaps") > 0)
        .then(pl.col("offense_snaps") / pl.col("team_offense_snaps"))
        .otherwise(None)
        .alias("offensive_snap_pct"),
        pl.struct(["completions", "attempts", "passing_yards", "passing_tds", "interceptions"])
        .map_elements(
            lambda r: passer_rating(
                r["completions"] or 0,
                r["attempts"] or 0,
                r["passing_yards"] or 0,
                r["passing_tds"] or 0,
                r["interceptions"] or 0,
            ),
            return_dtype=pl.Float64,
        )
        .alias("passer_rating"),
        pl.when(pl.col("cpoe_ngs_weeks") > 0)
        .then(pl.col("cpoe_ngs"))
        .when(pl.col("cpoe_box_weeks") > 0)
        .then(pl.col("cpoe_box"))
        .otherwise(None)
        .alias("cpoe"),
        pl.col("ngs_pass_avg_time_to_throw_ytd").alias("time_to_throw"),
        pl.col("ngs_pass_avg_intended_air_yards_ytd").alias("iay"),
        pl.col("ngs_pass_avg_completed_air_yards_ytd").alias("cay"),
        pl.col("ngs_pass_aggressiveness_ytd").alias("aggressiveness"),
        pl.col("ngs_pass_expected_completion_percentage_ytd").alias("expected_completion_pct"),
        pl.col("ngs_pass_avg_air_yards_differential_ytd").alias("air_yards_diff"),
        pl.col("ngs_pass_avg_air_yards_to_sticks_ytd").alias("air_yards_to_sticks"),
        pl.col("carries").alias("rush_attempts"),
        pl.col("attempts").alias("pass_attempts"),
    )


def _aggregate_backfield_wide(panel: pl.DataFrame) -> pl.DataFrame:
    panel = _with_team_off_snaps(_scale_ngs_percents(panel))
    agg = panel.group_by(_base_keys()).agg(
        *_meta_aggs(),
        pl.col("carries").sum().alias("carries"),
        pl.col("rushing_yards").sum().alias("rushing_yards"),
        pl.col("rushing_tds").sum().alias("rushing_tds"),
        pl.col("rushing_epa").sum().alias("rushing_epa"),
        (pl.col("rushing_fumbles").fill_null(0) + pl.col("receiving_fumbles").fill_null(0))
        .sum()
        .alias("fumbles"),
        (pl.col("carries").fill_null(0) + pl.col("targets").fill_null(0)).sum().alias("touches"),
        (pl.col("carries") > 0).sum().alias("games_with_carries"),
        pl.col("pfr_rush_rushing_broken_tackles").sum().alias("broken_tackles"),
        pl.col("pfr_rush_rushing_yards_after_contact").sum().alias("yards_after_contact"),
        pl.col("pfr_rush_rushing_broken_tackles").drop_nulls().len().alias("pfr_rush_weeks"),
        pl.col("targets").sum().alias("targets"),
        pl.col("receptions").sum().alias("receptions"),
        pl.col("receiving_yards").sum().alias("receiving_yards"),
        _share_ytd("targets", "target_share", "target_share"),
        # WOPR rebuilt from weekly when possible; fallback mean
        pl.col("wopr").mean().alias("wopr_mean"),
        pl.col("ngs_rush_avg_time_to_los").drop_nulls().len().alias("ngs_rush_weeks"),
        _ngs_weighted_mean("ngs_rush_avg_time_to_los", "ngs_rush_rush_attempts"),
        _ngs_weighted_mean(
            "ngs_rush_rush_yards_over_expected_per_att", "ngs_rush_rush_attempts"
        ),
        _ngs_weighted_mean(
            "ngs_rush_percent_attempts_gte_eight_defenders", "ngs_rush_rush_attempts"
        ),
        _ngs_weighted_mean("ngs_rush_efficiency", "ngs_rush_rush_attempts"),
        pl.col("ngs_rush_expected_rush_yards").sum().alias("expected_rush_yards"),
        pl.col("ffopp_total_fantasy_points_exp").sum().alias("expected_fantasy_points"),
        pl.col("ffopp_total_fantasy_points_exp").drop_nulls().len().alias("ffopp_weeks"),
        *_snap_off_aggs(),
    )
    return agg.with_columns(
        pl.col("carries").alias("rush_attempts"),
        pl.when(pl.col("carries") > 0)
        .then(pl.col("rushing_yards") / pl.col("carries"))
        .otherwise(None)
        .alias("yards_per_carry"),
        pl.col("ngs_rush_avg_time_to_los_ytd").alias("time_to_los"),
        pl.col("ngs_rush_rush_yards_over_expected_per_att_ytd").alias("ryoe"),
        pl.col("ngs_rush_percent_attempts_gte_eight_defenders_ytd").alias(
            "eight_plus_defenders_pct"
        ),
        pl.col("ngs_rush_efficiency_ytd").alias("ngs_efficiency"),
        pl.col("wopr_mean").alias("wopr"),
        pl.when(pl.col("team_offense_snaps") > 0)
        .then(pl.col("offense_snaps") / pl.col("team_offense_snaps"))
        .otherwise(None)
        .alias("offensive_snap_pct"),
        # Play-grain not on spine
        pl.lit(None).cast(pl.Float64).alias("red_zone_touches"),
    )


def _aggregate_pass_catcher_wide(panel: pl.DataFrame) -> pl.DataFrame:
    panel = _with_team_off_snaps(panel)
    agg = panel.group_by(_base_keys()).agg(
        *_meta_aggs(),
        pl.col("targets").sum().alias("targets"),
        pl.col("receptions").sum().alias("receptions"),
        pl.col("receiving_yards").sum().alias("receiving_yards"),
        pl.col("receiving_tds").sum().alias("receiving_tds"),
        pl.col("receiving_yards_after_catch").sum().alias("yac"),
        pl.col("receiving_air_yards").sum().alias("receiving_air_yards"),
        pl.col("receiving_epa").sum().alias("receiving_epa"),
        pl.col("pfr_rec_receiving_drop").sum().alias("drops"),
        pl.col("pfr_rec_receiving_drop").drop_nulls().len().alias("pfr_rec_weeks"),
        _share_ytd("targets", "target_share", "target_share"),
        _share_ytd("receiving_air_yards", "air_yards_share", "air_yards_share"),
        pl.col("wopr").mean().alias("wopr"),
        pl.col("ngs_rec_avg_yac").drop_nulls().len().alias("ngs_rec_weeks"),
        _ngs_weighted_mean("ngs_rec_avg_yac", "ngs_rec_targets"),
        _ngs_weighted_mean("ngs_rec_avg_expected_yac", "ngs_rec_targets"),
        _ngs_weighted_mean("ngs_rec_avg_yac_above_expectation", "ngs_rec_targets"),
        _ngs_weighted_mean("ngs_rec_avg_separation", "ngs_rec_targets"),
        _ngs_weighted_mean("ngs_rec_avg_cushion", "ngs_rec_targets"),
        *_snap_off_aggs(),
    )
    return agg.with_columns(
        pl.when(pl.col("targets") > 0)
        .then(pl.col("receptions") / pl.col("targets"))
        .otherwise(None)
        .alias("catch_pct"),
        pl.when(pl.col("targets") > 0)
        .then(pl.col("receiving_air_yards") / pl.col("targets"))
        .otherwise(None)
        .alias("adot"),
        pl.when(pl.col("receiving_air_yards") > 0)
        .then(pl.col("receiving_yards") / pl.col("receiving_air_yards"))
        .otherwise(None)
        .alias("racr"),
        pl.col("ngs_rec_avg_yac_ytd").alias("avg_yac"),
        pl.col("ngs_rec_avg_expected_yac_ytd").alias("expected_yac"),
        pl.col("ngs_rec_avg_yac_above_expectation_ytd").alias("yac_oe"),
        pl.col("ngs_rec_avg_separation_ytd").alias("separation"),
        pl.col("ngs_rec_avg_cushion_ytd").alias("cushion"),
        pl.when(pl.col("team_offense_snaps") > 0)
        .then(pl.col("offense_snaps") / pl.col("team_offense_snaps"))
        .otherwise(None)
        .alias("offensive_snap_pct"),
        pl.lit(None).cast(pl.Float64).alias("red_zone_targets"),
        pl.lit(None).cast(pl.Float64).alias("route_pct"),
    )


def _aggregate_ol_wide(panel: pl.DataFrame) -> pl.DataFrame:
    panel = _with_team_off_snaps(panel)
    agg = panel.group_by(_base_keys()).agg(
        *_meta_aggs(),
        pl.col("penalties").sum().alias("penalties"),
        pl.col("penalty_yards").sum().alias("penalty_yards"),
        *_snap_off_aggs(),
    )
    return agg.with_columns(
        pl.col("offense_snaps").alias("snaps_played"),
        pl.when(pl.col("team_offense_snaps") > 0)
        .then(pl.col("offense_snaps") / pl.col("team_offense_snaps"))
        .otherwise(None)
        .alias("snap_pct"),
    )


def _aggregate_def_front_wide(panel: pl.DataFrame) -> pl.DataFrame:
    panel = _with_team_def_snaps(panel)
    agg = panel.group_by(_base_keys()).agg(
        *_meta_aggs(),
        pl.col("def_tackles_solo").sum().alias("tackles_solo"),
        pl.col("def_tackle_assists").sum().alias("tackles_ast"),
        (
            pl.col("def_tackles_solo").fill_null(0) + pl.col("def_tackle_assists").fill_null(0)
        )
        .sum()
        .alias("tackles_combined"),
        pl.col("def_sacks").sum().alias("sacks"),
        pl.col("def_tackles_for_loss").sum().alias("tackles_for_loss"),
        pl.col("def_qb_hits").sum().alias("qb_hits"),
        pl.col("def_fumbles_forced").sum().alias("forced_fumbles"),
        pl.col("def_interceptions").sum().alias("interceptions"),
        pl.col("pfr_def_def_missed_tackles").sum().alias("missed_tackles"),
        pl.col("pfr_def_def_pressures").sum().alias("pressures"),
        pl.col("pfr_def_def_times_hurried").sum().alias("hurries"),
        pl.col("pfr_def_def_missed_tackles").drop_nulls().len().alias("pfr_def_weeks"),
        *_snap_def_aggs(),
    )
    return agg.with_columns(
        pl.when(pl.col("team_defense_snaps") > 0)
        .then(pl.col("defense_snaps") / pl.col("team_defense_snaps"))
        .otherwise(None)
        .alias("defensive_snap_pct"),
    )


def _aggregate_secondary_wide(panel: pl.DataFrame) -> pl.DataFrame:
    panel = _with_team_def_snaps(panel)
    agg = panel.group_by(_base_keys()).agg(
        *_meta_aggs(),
        pl.col("def_interceptions").sum().alias("interceptions"),
        pl.col("def_pass_defended").sum().alias("passes_defended"),
        (
            pl.col("def_tackles_solo").fill_null(0) + pl.col("def_tackle_assists").fill_null(0)
        )
        .sum()
        .alias("tackles_combined"),
        pl.col("pfr_def_def_targets").sum().alias("targets_allowed"),
        pl.col("pfr_def_def_completions_allowed").sum().alias("completions_allowed"),
        pl.col("pfr_def_def_yards_allowed").sum().alias("receiving_yards_allowed"),
        pl.col("pfr_def_def_receiving_td_allowed").sum().alias("tds_allowed"),
        pl.col("pfr_def_def_missed_tackles").sum().alias("missed_tackles"),
        pl.col("pfr_def_def_targets").drop_nulls().len().alias("pfr_def_weeks"),
        _ngs_weighted_mean("pfr_def_def_passer_rating_allowed", "pfr_def_def_targets").alias(
            "rating_allowed"
        ),
        _ngs_weighted_mean("pfr_def_def_adot", "pfr_def_def_targets").alias("adot_allowed"),
        *_snap_def_aggs(),
    )
    return agg.with_columns(
        pl.when(pl.col("targets_allowed") > 0)
        .then(pl.col("completions_allowed") / pl.col("targets_allowed"))
        .otherwise(None)
        .alias("completion_pct_allowed"),
        pl.when(pl.col("team_defense_snaps") > 0)
        .then(pl.col("defense_snaps") / pl.col("team_defense_snaps"))
        .otherwise(None)
        .alias("defensive_snap_pct"),
    )


def _aggregate_kicker_wide(panel: pl.DataFrame) -> pl.DataFrame:
    agg = panel.group_by(_base_keys()).agg(
        *_meta_aggs(),
        pl.col("fg_att").sum().alias("fg_att"),
        pl.col("fg_made").sum().alias("fg_made"),
        pl.col("fg_long").max().alias("fg_long"),
        pl.col("fg_made_40_49").sum().alias("fg_40_49"),
        pl.col("pat_att").sum().alias("pat_att"),
        pl.col("pat_made").sum().alias("pat_made"),
    )
    return agg.with_columns(
        pl.col("fg_att").alias("fg_attempts"),
        pl.col("pat_att").alias("xp_attempts"),
        pl.col("pat_made").alias("xp_made"),
        pl.when(pl.col("fg_att") > 0)
        .then(pl.col("fg_made") / pl.col("fg_att"))
        .otherwise(None)
        .alias("fg_pct"),
    )


def _aggregate_punter_wide(panel: pl.DataFrame) -> pl.DataFrame:
    agg = panel.group_by(_base_keys()).agg(
        *_meta_aggs(),
        pl.col("pt_att").sum().alias("punts"),
        pl.col("pt_yards").sum().alias("gross_punt_yards"),
        pl.col("pt_net_yards").sum().alias("net_punt_yards"),
        pl.col("pt_inside_20").sum().alias("inside_20"),
        pl.col("pt_touchback").sum().alias("touchbacks"),
        pl.col("pt_fair_caught").sum().alias("fair_catches"),
    )
    return agg


def _aggregate_returner_wide(panel: pl.DataFrame) -> pl.DataFrame:
    agg = panel.group_by(_base_keys()).agg(
        *_meta_aggs(),
        pl.col("kickoff_returns").sum().alias("kick_returns"),
        pl.col("punt_returns").sum().alias("punt_returns"),
        pl.col("kickoff_return_yards").sum().alias("kick_return_yards"),
        pl.col("punt_return_yards").sum().alias("punt_return_yards"),
        (
            pl.col("pt_return_tds").fill_null(0) + pl.col("special_teams_tds").fill_null(0)
        )
        .sum()
        .alias("return_tds"),
    )
    return agg.with_columns(
        (pl.col("kick_returns").fill_null(0) + pl.col("punt_returns").fill_null(0)).alias(
            "returns"
        ),
    )


_AGGREGATORS: dict[str, Callable[[pl.DataFrame], pl.DataFrame]] = {
    "qb": _aggregate_qb_wide,
    "backfield": _aggregate_backfield_wide,
    "pass_catcher": _aggregate_pass_catcher_wide,
    "ol": _aggregate_ol_wide,
    "def_front": _aggregate_def_front_wide,
    "secondary": _aggregate_secondary_wide,
    "kicker": _aggregate_kicker_wide,
    "punter": _aggregate_punter_wide,
    "returner": _aggregate_returner_wide,
}


def _denom_value(row: dict, denom: str) -> float | None:
    val = row.get(denom)
    if val is None:
        return None
    return float(val)


def _source_present(stat_id: str, row: dict, position_group: str) -> bool:
    """False → missing_source (do not impute 0)."""
    # Play-grain / participation not on spine yet
    if stat_id in {"red_zone_touches", "red_zone_targets", "route_pct"}:
        return False

    if stat_id == "cpoe":
        return (
            (row.get("cpoe_ngs_weeks") or 0) > 0 or (row.get("cpoe_box_weeks") or 0) > 0
        ) and row.get("cpoe") is not None

    ngs_pass = {
        "time_to_throw",
        "iay",
        "cay",
        "aggressiveness",
        "expected_completion_pct",
        "air_yards_diff",
        "air_yards_to_sticks",
    }
    if stat_id in ngs_pass:
        return (row.get("ngs_weeks") or 0) > 0 and row.get(stat_id) is not None

    ngs_rush = {"time_to_los", "ryoe", "eight_plus_defenders_pct", "ngs_efficiency", "expected_rush_yards"}
    if stat_id in ngs_rush:
        return (row.get("ngs_rush_weeks") or 0) > 0 and row.get(stat_id) is not None

    ngs_rec = {"avg_yac", "expected_yac", "yac_oe", "separation", "cushion"}
    if stat_id in ngs_rec:
        return (row.get("ngs_rec_weeks") or 0) > 0 and row.get(stat_id) is not None

    if stat_id in {"offensive_snap_pct", "snap_pct", "snaps_played", "defensive_snap_pct"}:
        return (row.get("snap_weeks") or 0) > 0 and row.get(stat_id) is not None

    if stat_id == "pacr":
        return (row.get("passing_air_yards") or 0) > 0 and row.get("pacr") is not None
    if stat_id == "racr":
        return (row.get("receiving_air_yards") or 0) > 0 and row.get("racr") is not None

    if stat_id in {"broken_tackles", "yards_after_contact"}:
        return (row.get("pfr_rush_weeks") or 0) > 0 and row.get(stat_id) is not None
    if stat_id == "drops":
        return (row.get("pfr_rec_weeks") or 0) > 0 and row.get(stat_id) is not None

    pfr_def = {
        "missed_tackles",
        "pressures",
        "hurries",
        "targets_allowed",
        "completions_allowed",
        "receiving_yards_allowed",
        "tds_allowed",
        "completion_pct_allowed",
        "rating_allowed",
        "adot_allowed",
    }
    if stat_id in pfr_def:
        return (row.get("pfr_def_weeks") or 0) > 0 and row.get(stat_id) is not None

    if stat_id == "expected_fantasy_points":
        return (row.get("ffopp_weeks") or 0) > 0 and row.get(stat_id) is not None

    return True


def _player_value(stat_id: str, row: dict) -> float | None:
    val = row.get(stat_id)
    if val is None:
        return None
    return float(val)


def _qualify(
    stat: StatDefinition,
    season: int,
    as_of_week: int,
    row: dict,
    position_group: str,
) -> tuple[bool, float | None, float | None, Unavailable | None]:
    if stat.always_unavailable or stat.min_n_base is None:
        return False, None, None, "not_in_nflverse"
    if stat.start_year is not None and season < stat.start_year:
        return False, None, None, "not_in_nflverse"
    if not _source_present(stat.id, row, position_group):
        return False, None, None, "missing_source"

    denom = _denom_value(row, stat.denom)
    threshold = min_n(stat.min_n_base, as_of_week)
    value = _player_value(stat.id, row)

    if denom is None or denom < threshold:
        return False, value, denom, "insufficient_sample"
    if value is None:
        return False, None, denom, "missing_source"
    return True, value, denom, None


def wide_to_long(
    wide: pl.DataFrame,
    *,
    season: int,
    as_of_week: int,
    position_group: str,
    stats: Iterable[StatDefinition] | None = None,
) -> pl.DataFrame:
    """Emit Stage C long rows matching viz.player_stat_values grain (no percentile yet)."""
    records: list[dict] = []
    stat_list = list(stats) if stats is not None else stats_for_group(position_group)
    for row in wide.to_dicts():
        for stat in stat_list:
            if stat.always_unavailable:
                # Omit always-unavailable ids from the panel (Knowball grays via catalog).
                continue
            qualified, value, denom, reason = _qualify(
                stat, season, as_of_week, row, position_group
            )
            threshold = (
                min_n(stat.min_n_base, as_of_week) if stat.min_n_base is not None else None
            )
            records.append(
                {
                    "player_id": row["player_id"],
                    "player_display_name": row.get("player_display_name"),
                    "team": row.get("team"),
                    "position_code": row.get("position_code"),
                    "position_group": row.get("position_group"),
                    "season": season,
                    "as_of_week": as_of_week,
                    "stat_id": stat.id,
                    "kind": stat.kind,
                    "higher_is_better": stat.higher_is_better,
                    "format": stat.format,
                    "player_value": value,
                    "denom_ytd": denom,
                    "min_n": threshold,
                    "qualified": qualified,
                    "unavailable_reason": reason,
                }
            )
    schema = {
        "player_id": pl.Utf8,
        "player_display_name": pl.Utf8,
        "team": pl.Utf8,
        "position_code": pl.Utf8,
        "position_group": pl.Utf8,
        "season": pl.Int64,
        "as_of_week": pl.Int64,
        "stat_id": pl.Utf8,
        "kind": pl.Utf8,
        "higher_is_better": pl.Boolean,
        "format": pl.Utf8,
        "player_value": pl.Float64,
        "denom_ytd": pl.Float64,
        "min_n": pl.Float64,
        "qualified": pl.Boolean,
        "unavailable_reason": pl.Utf8,
    }
    # Explicit schema: early rows often have null reasons and Polars would
    # infer Null then fail when a string reason appears.
    return pl.DataFrame(records, schema=schema)


def build_ytd(
    season: int,
    as_of_week: int,
    *,
    position_group: str = "qb",
) -> YtdResult:
    """Build catalog YTD long panel for a position group and persist parquet."""
    if position_group not in _AGGREGATORS:
        raise NotImplementedError(f"Stage C unknown group {position_group}")
    ensure_data_dirs()
    t0 = time.perf_counter()

    spine = pl.read_parquet(SPINE_DIR / f"player_week_{season}.parquet")
    panel = spine.filter(
        (pl.col("week") >= 1)
        & (pl.col("week") <= as_of_week)
        & (pl.col("position_group") == position_group)
    )
    panel = _ensure_optional_float_cols(panel)
    wide = _AGGREGATORS[position_group](panel)
    long = wide_to_long(wide, season=season, as_of_week=as_of_week, position_group=position_group)

    out = YTD_DIR / f"ytd_{position_group}_{season}_w{as_of_week}.parquet"
    long.write_parquet(out)
    return YtdResult(
        path=str(out),
        rows=long.height,
        players=wide.height,
        seconds=time.perf_counter() - t0,
    )


def player_stat_table(long: pl.DataFrame, player_id: str) -> pl.DataFrame:
    return (
        long.filter(pl.col("player_id") == player_id)
        .select(
            [
                "stat_id",
                "player_value",
                "denom_ytd",
                "min_n",
                "qualified",
                "unavailable_reason",
                "format",
            ]
        )
        .sort("stat_id")
    )
