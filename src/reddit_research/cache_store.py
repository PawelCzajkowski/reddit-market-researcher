"""Read/write the frozen query-keyed cache files on disk.

TTL and purge helpers (used by the cache lifecycle) key off `metadata.fetched_at`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .models.cache import RawCache


def write_cache(cache: RawCache, path: str | Path) -> Path:
    """Write a RawCache to `path` (overwriting), creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(cache.model_dump_json(indent=2), encoding="utf-8")
    return p


def read_cache(path: str | Path) -> RawCache:
    """Read and validate a RawCache from disk."""
    return RawCache.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _parse_iso(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def cache_age_days(cache: RawCache, *, now: datetime | None = None) -> float:
    """Age of the cache in days, from `metadata.fetched_at`."""
    now = now or datetime.now(timezone.utc)
    return (now - _parse_iso(cache.metadata.fetched_at)).total_seconds() / 86400.0


def purge_cache(
    cache_dir: str | Path, *, older_than_days: float, now: datetime | None = None
) -> list[Path]:
    """Delete cache files older than `older_than_days` (by `fetched_at`).

    Returns the deleted paths. Files that don't parse as a RawCache are left
    untouched (we only delete what we can positively date).
    """
    now = now or datetime.now(timezone.utc)
    directory = Path(cache_dir)
    if not directory.exists():
        return []
    deleted: list[Path] = []
    for path in sorted(directory.glob("*.json")):
        try:
            cache = read_cache(path)
        except Exception:
            continue
        if cache_age_days(cache, now=now) > older_than_days:
            path.unlink()
            deleted.append(path)
    return deleted
