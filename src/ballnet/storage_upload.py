"""Upload local Stage G JSON to Supabase Storage (`knowball-public`)."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from supabase import Client, create_client

from ballnet.paths import INDEX_DIR, LEAGUE_DIR, PAGES_DIR, REPO_ROOT
from ballnet.publish import PUBLISHABLE_GROUPS, default_as_of_week

DEFAULT_BUCKET = "knowball-public"
# Free plan is 1 GB; skip pages/current duplicates by default (~2× season size).
# Keep concurrency modest — bursty uploads get RemoteProtocolError from Storage.
DEFAULT_WORKERS = 4


@dataclass(frozen=True)
class UploadResult:
    bucket: str
    uploaded: int
    failed: int
    bytes: int
    seconds: float
    errors: list[str]


def _load_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def supabase_client(*, env_file: Path | None = None) -> Client:
    """Build a service-role client from env / ballnet `.env`."""
    file_env = _load_dotenv(env_file or (REPO_ROOT / ".env"))
    url = os.environ.get("SUPABASE_URL") or os.environ.get("supabase_url") or file_env.get(
        "supabase_url"
    ) or file_env.get("SUPABASE_URL")
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("supabase_service_role_key")
        or file_env.get("supabase_service_role_key")
        or file_env.get("SUPABASE_SERVICE_ROLE_KEY")
    )
    if not url or not key:
        raise SystemExit(
            "Missing supabase_url / supabase_service_role_key "
            "(set in ballnet/.env or the environment)"
        )
    return create_client(url, key)


def _upload_one(
    client: Client,
    bucket: str,
    object_path: str,
    local: Path,
    *,
    upsert: bool,
    retries: int = 4,
) -> tuple[str, int]:
    data = local.read_bytes()
    opts: dict[str, str] = {"content-type": "application/json"}
    if upsert:
        opts["upsert"] = "true"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            client.storage.from_(bucket).upload(object_path, data, file_options=opts)
            return object_path, len(data)
        except Exception as e:  # noqa: BLE001
            last_err = e
            # Brief backoff; free-tier Storage often drops bursty concurrent uploads.
            time.sleep(0.4 * (2**attempt))
    assert last_err is not None
    raise last_err


def upload_files(
    pairs: Iterable[tuple[str, Path]],
    *,
    bucket: str = DEFAULT_BUCKET,
    upsert: bool = True,
    workers: int = DEFAULT_WORKERS,
    client: Client | None = None,
) -> UploadResult:
    """Upload (object_path, local_path) pairs concurrently."""
    client = client or supabase_client()
    items = [(obj, path) for obj, path in pairs if path.is_file()]
    t0 = time.perf_counter()
    uploaded = 0
    failed = 0
    total_bytes = 0
    errors: list[str] = []

    def work(item: tuple[str, Path]) -> tuple[str, int]:
        obj, path = item
        return _upload_one(client, bucket, obj, path, upsert=upsert)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(work, item): item for item in items}
        done = 0
        for fut in as_completed(futures):
            obj, path = futures[fut]
            done += 1
            try:
                _, n = fut.result()
                uploaded += 1
                total_bytes += n
            except Exception as e:  # noqa: BLE001 — collect and continue
                failed += 1
                errors.append(f"{obj}: {type(e).__name__}: {e}")
            if done % 100 == 0 or done == len(futures):
                print(
                    f"  progress {done}/{len(futures)} "
                    f"ok={uploaded} fail={failed}",
                    flush=True,
                )

    return UploadResult(
        bucket=bucket,
        uploaded=uploaded,
        failed=failed,
        bytes=total_bytes,
        seconds=time.perf_counter() - t0,
        errors=errors[:20],
    )


def upload_index(*, bucket: str = DEFAULT_BUCKET, **kwargs) -> UploadResult:
    """Upload `index/{players,current,seasons}.json` from local data/index."""
    names = ("players.json", "current.json", "seasons.json")
    pairs = [(f"index/{name}", INDEX_DIR / name) for name in names]
    missing = [str(p) for _, p in pairs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing index files: {missing}")
    return upload_files(pairs, bucket=bucket, **kwargs)


def upload_season_pages(
    season: int,
    as_of_week: int | None = None,
    *,
    bucket: str = DEFAULT_BUCKET,
    also_current: bool = False,
    **kwargs,
) -> UploadResult:
    """Upload `pages/{season}/w{week}/*.json` (optional `pages/current/` copies)."""
    week = as_of_week if as_of_week is not None else default_as_of_week(season)
    week_dir = PAGES_DIR / str(season) / f"w{week}"
    if not week_dir.is_dir():
        raise FileNotFoundError(f"missing local pages dir {week_dir}")

    pairs: list[tuple[str, Path]] = []
    for path in sorted(week_dir.glob("*.json")):
        pairs.append((f"pages/{season}/w{week}/{path.name}", path))
        if also_current:
            pairs.append((f"pages/current/{path.name}", path))

    return upload_files(pairs, bucket=bucket, **kwargs)


def upload_season_league(
    season: int,
    as_of_week: int | None = None,
    *,
    bucket: str = DEFAULT_BUCKET,
    groups: Iterable[str] | None = None,
    **kwargs,
) -> UploadResult:
    """Upload `league/{season}/w{week}/{group}.json` shared curve files."""
    week = as_of_week if as_of_week is not None else default_as_of_week(season)
    selected = list(groups) if groups is not None else list(PUBLISHABLE_GROUPS)
    pairs: list[tuple[str, Path]] = []
    missing: list[str] = []
    for group in selected:
        local = LEAGUE_DIR / str(season) / f"w{week}" / f"{group}.json"
        if not local.is_file():
            missing.append(str(local))
            continue
        pairs.append((f"league/{season}/w{week}/{group}.json", local))
    if missing and not pairs:
        raise FileNotFoundError(f"missing league files: {missing}")
    return upload_files(pairs, bucket=bucket, **kwargs)
