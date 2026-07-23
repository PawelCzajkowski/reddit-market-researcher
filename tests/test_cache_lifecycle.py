"""Cache lifecycle: age from fetched_at, TTL boundary, purge by age."""

from __future__ import annotations

from datetime import datetime, timezone

from reddit_research.cache_store import cache_age_days, purge_cache, write_cache
from reddit_research.defaults import CACHE_TTL_DAYS

from .cache_fixtures import comment, make_cache, post

NOW = datetime(2026, 7, 23, tzinfo=timezone.utc)


def _cache_fetched(days_ago: float):
    fetched = datetime(2026, 7, 23, tzinfo=timezone.utc)
    fetched = fetched.fromtimestamp(NOW.timestamp() - days_ago * 86400, tz=timezone.utc)
    return make_cache(
        [post("p1", [comment("c1")])],
        fetched_at=fetched.isoformat().replace("+00:00", "Z"),
    )


def test_cache_age_days_from_fetched_at() -> None:
    assert cache_age_days(_cache_fetched(10), now=NOW) == 10.0


def test_ttl_boundary_default_is_30_days() -> None:
    assert CACHE_TTL_DAYS == 30
    assert cache_age_days(_cache_fetched(31), now=NOW) > CACHE_TTL_DAYS
    assert cache_age_days(_cache_fetched(29), now=NOW) < CACHE_TTL_DAYS


def test_purge_deletes_only_older_than_threshold(tmp_path) -> None:
    fresh = tmp_path / "fresh__aaaa.json"
    stale = tmp_path / "stale__bbbb.json"
    write_cache(_cache_fetched(5), fresh)
    write_cache(_cache_fetched(45), stale)

    deleted = purge_cache(tmp_path, older_than_days=30, now=NOW)

    assert deleted == [stale]
    assert not stale.exists()
    assert fresh.exists()


def test_purge_ignores_unparseable_files(tmp_path) -> None:
    (tmp_path / "junk__cccc.json").write_text("not json", encoding="utf-8")
    write_cache(_cache_fetched(100), tmp_path / "old__dddd.json")

    deleted = purge_cache(tmp_path, older_than_days=30, now=NOW)

    assert [p.name for p in deleted] == ["old__dddd.json"]
    assert (tmp_path / "junk__cccc.json").exists()  # left untouched


def test_purge_missing_dir_returns_empty(tmp_path) -> None:
    assert purge_cache(tmp_path / "nope", older_than_days=30, now=NOW) == []
