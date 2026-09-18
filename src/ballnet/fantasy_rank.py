"""Fantasy position ranks for Knowball subscripts (QB/WR/RB/TE only)."""

from __future__ import annotations

import json
from typing import Any, Literal, Mapping, TypedDict

import polars as pl

from ballnet.paths import INDEX_DIR, RAW_DIR
from ballnet.positions import POSITION_TO_CODE

RankKind = Literal["consensus", "finish"]

SKILL_POSITIONS: frozenset[str] = frozenset({"QB", "WR", "RB", "TE"})

# Modal-position lookup for finish ranks — same mapping as Stage B, as a frame.
_POS_MAP = pl.DataFrame(
    {
        "raw": list(POSITION_TO_CODE.keys()),
        "code": list(POSITION_TO_CODE.values()),
    }
)


class FantasyPosRank(TypedDict):
    rank: int
    kind: RankKind


# One table per (season, as_of_week, kind). Publish-all / leaderboards / highlights
# share this so we do not hit FantasyPros or re-sum PPR once per group.
_RANKS_CACHE: dict[tuple[int, int, RankKind], dict[str, FantasyPosRank]] = {}


def resolve_rank_kind(
    season: int,
    *,
    updating_current: bool = False,
) -> RankKind:
    """Live pointer → consensus; every other season → PPR finish."""
    if updating_current:
        return "consensus"
    live = _live_season_from_index()
    if live is not None and season == live:
        return "consensus"
    return "finish"


def fantasy_pos_ranks(
    season: int,
    as_of_week: int,
    *,
    kind: RankKind | None = None,
    updating_current: bool = False,
) -> dict[str, FantasyPosRank]:
    """`(player_id, season)` ranks for this publish slice. Cached per kind."""
    resolved = kind or resolve_rank_kind(season, updating_current=updating_current)
    key = (int(season), int(as_of_week), resolved)
    cached = _RANKS_CACHE.get(key)
    if cached is not None:
        return cached
    built = (
        _consensus_ranks(resolved)
        if resolved == "consensus"
        else _finish_ranks(season, resolved)
    )
    _RANKS_CACHE[key] = built
    return built


def attach_fantasy_pos_rank(
    obj: dict[str, Any],
    player_id: str,
    ranks: Mapping[str, FantasyPosRank] | None,
) -> None:
    """Set optional JSON fields. Missing / non-skill players stay omitted."""
    if not ranks:
        return
    rec = ranks.get(player_id)
    if rec is None:
        return
    obj["fantasyPosRank"] = int(rec["rank"])
    obj["fantasyPosRankKind"] = rec["kind"]


def _live_season_from_index() -> int | None:
    path = INDEX_DIR / "current.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return int(payload["season"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _fp_id_map() -> pl.DataFrame:
    path = RAW_DIR / "ff_playerids.parquet"
    if not path.is_file():
        return pl.DataFrame(schema={"fantasypros_id": pl.Utf8, "player_id": pl.Utf8})
    ids = pl.read_parquet(path)
    if "fantasypros_id" not in ids.columns or "gsis_id" not in ids.columns:
        return pl.DataFrame(schema={"fantasypros_id": pl.Utf8, "player_id": pl.Utf8})
    return (
        ids.select(
            pl.col("fantasypros_id").cast(pl.Utf8),
            pl.col("gsis_id").cast(pl.Utf8).alias("player_id"),
        )
        .filter(
            pl.col("fantasypros_id").is_not_null()
            & (pl.col("fantasypros_id") != "")
            & pl.col("player_id").is_not_null()
            & (pl.col("player_id") != "")
        )
        .unique(subset=["fantasypros_id"], keep="first")
    )


def _filter_redraft_positional(df: pl.DataFrame) -> pl.DataFrame:
    if "pos" not in df.columns:
        return df.filter(pl.lit(False))
    out = df.with_columns(pl.col("pos").cast(pl.Utf8).str.to_uppercase().alias("_pos"))
    ecr_vals = (
        out["ecr_type"].drop_nulls() if "ecr_type" in out.columns else None
    )
    if ecr_vals is not None and ecr_vals.len() > 0:
        out = out.filter(pl.col("ecr_type").cast(pl.Utf8).str.to_lowercase() == "rp")
    else:
        for col in ("page_type", "fp_page"):
            if col not in out.columns:
                continue
            lowered = pl.col(col).cast(pl.Utf8).fill_null("").str.to_lowercase()
            out = out.filter(
                ~lowered.str.contains("dynasty")
                & ~lowered.str.contains("overall")
                & ~lowered.str.contains("superflex")
                & ~lowered.str.contains("sflex")
            )
    return out.filter(pl.col("_pos").is_in(sorted(SKILL_POSITIONS)))


def _consensus_ranks(kind: RankKind) -> dict[str, FantasyPosRank]:
    try:
        import nflreadpy as nfl

        weekly = nfl.load_ff_rankings("week")
    except Exception as e:
        print(f"fantasy_rank consensus skipped: {type(e).__name__}: {e}")
        return {}
    if weekly is None or weekly.height == 0:
        return {}

    weekly = _filter_redraft_positional(weekly)
    if weekly.height == 0 or "ecr" not in weekly.columns:
        return {}

    fp_col = (
        "fantasypros_id"
        if "fantasypros_id" in weekly.columns
        else "id"
        if "id" in weekly.columns
        else None
    )
    if fp_col is None:
        return {}

    ranked = (
        weekly.select(
            pl.col(fp_col).cast(pl.Utf8).alias("fantasypros_id"),
            pl.col("ecr").cast(pl.Float64),
        )
        .filter(
            pl.col("fantasypros_id").is_not_null()
            & (pl.col("fantasypros_id") != "")
            & pl.col("ecr").is_not_null()
            & pl.col("ecr").is_finite()
        )
        .with_columns(pl.col("ecr").round(0).cast(pl.Int32).alias("rank"))
        .filter(pl.col("rank") >= 1)
        .sort("rank")
        .unique(subset=["fantasypros_id"], keep="first")
    )
    joined = ranked.join(_fp_id_map(), on="fantasypros_id", how="inner")
    joined = joined.filter(pl.col("player_id").is_not_null()).unique(
        subset=["player_id"], keep="first"
    )
    # Null GSIS is a silent skip (inner join). Do not invent a rank.
    return {
        rec["player_id"]: FantasyPosRank(rank=int(rec["rank"]), kind=kind)
        for rec in joined.select("player_id", "rank").to_dicts()
    }


def _finish_ranks(season: int, kind: RankKind) -> dict[str, FantasyPosRank]:
    path = RAW_DIR / f"player_stats_{season}.parquet"
    if not path.is_file():
        return {}
    box = pl.read_parquet(path)
    if "player_id" not in box.columns or "fantasy_points_ppr" not in box.columns:
        return {}
    if "season_type" in box.columns:
        box = box.filter(pl.col("season_type") == "REG")
    elif "game_type" in box.columns:
        box = box.filter(pl.col("game_type") == "REG")
    if box.height == 0:
        return {}

    pos_col = "position" if "position" in box.columns else None
    if pos_col is None:
        return {}

    mapped = (
        box.with_columns(
            pl.col(pos_col).cast(pl.Utf8).str.to_uppercase().str.strip_chars().alias("raw")
        )
        .join(_POS_MAP, on="raw", how="left")
        .with_columns(pl.col("player_id").cast(pl.Utf8))
    )
    # Modal weekly box position, then keep QB/WR/RB/TE only (FB/OL/K/… omit).
    modal = (
        mapped.filter(pl.col("code").is_not_null())
        .group_by("player_id", "code")
        .len()
        .sort(["len", "code"], descending=[True, False])
        .unique(subset=["player_id"], keep="first")
        .filter(pl.col("code").is_in(sorted(SKILL_POSITIONS)))
        .select("player_id", "code")
    )
    ppr = (
        mapped.group_by("player_id")
        .agg(pl.col("fantasy_points_ppr").cast(pl.Float64).sum().alias("ppr"))
        .filter(pl.col("ppr").is_not_null() & (pl.col("ppr") > 0))
    )
    if modal.height == 0 or ppr.height == 0:
        return {}
    joined = modal.join(ppr, on="player_id", how="inner")
    # Competition rank: ties share the min (1, 2, 2, 4).
    ranked = joined.with_columns(
        pl.col("ppr").rank(method="min", descending=True).over("code").cast(pl.Int32).alias("rank")
    )
    return {
        rec["player_id"]: FantasyPosRank(rank=int(rec["rank"]), kind=kind)
        for rec in ranked.select("player_id", "rank").to_dicts()
    }
