"""Local data layout (gitignored `data/`)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPINE_DIR = DATA_DIR / "spine"
YTD_DIR = DATA_DIR / "ytd"
DIST_DIR = DATA_DIR / "dists"
PAGES_DIR = DATA_DIR / "pages"
INDEX_DIR = DATA_DIR / "index"
# Knowball-facing league shapes (camelCase). Distinct from Stage D `dists/league_ytd_*`.
LEAGUE_DIR = DATA_DIR / "league"


def ensure_data_dirs() -> None:
    for d in (RAW_DIR, SPINE_DIR, YTD_DIR, DIST_DIR, PAGES_DIR, INDEX_DIR, LEAGUE_DIR):
        d.mkdir(parents=True, exist_ok=True)
