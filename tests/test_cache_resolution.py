"""analyze cache resolution: fetch-fresh default, --use-cached reuse, TTL warning."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from reddit_research import commands
from reddit_research.cache_store import write_cache
from reddit_research.fetch.query import cache_path, query_hash
from reddit_research.models.cache import FetchParams

from .cache_fixtures import comment, make_cache, post


def _write_query_cache(cache_dir, topic, subs, *, days_ago: float):
    fetched = datetime.now(timezone.utc) - timedelta(days=days_ago)
    cache = make_cache(
        [post("p1", [comment("c1")])],
        topic=topic,
        subreddits=subs,
        fetched_at=fetched.isoformat().replace("+00:00", "Z"),
    )
    path = cache_path(topic, query_hash(topic, subs, FetchParams()), cache_dir)
    write_cache(cache, path)
    return path


def test_use_cached_reuses_existing_file_without_fetching(tmp_path, monkeypatch) -> None:
    subs = ["r/Notion"]
    _write_query_cache(tmp_path, "Notion", subs, days_ago=1)

    def _boom(**kwargs):  # fetch must NOT be called
        raise AssertionError("should not fetch when a fresh cache exists")

    monkeypatch.setattr(commands, "_fetch_to_cache", _boom)
    cache = commands._resolve_cache(
        topic="Notion", subreddits=subs, use_cached=True, cache_dir=str(tmp_path)
    )
    assert cache.metadata.topic == "Notion"


def test_no_use_cached_fetches_fresh(tmp_path, monkeypatch) -> None:
    subs = ["r/Notion"]
    _write_query_cache(tmp_path, "Notion", subs, days_ago=1)  # exists but should be ignored
    sentinel = make_cache([post("p9", [comment("z")])], topic="Notion", subreddits=subs)

    called = {}

    def _fake_fetch(**kwargs):
        called["yes"] = True
        return sentinel, tmp_path / "fresh.json"

    monkeypatch.setattr(commands, "_fetch_to_cache", _fake_fetch)
    cache = commands._resolve_cache(
        topic="Notion", subreddits=subs, use_cached=False, cache_dir=str(tmp_path)
    )
    assert called.get("yes") is True
    assert cache.posts[0].id == "p9"


def test_use_cached_missing_file_falls_back_to_fetch(tmp_path, monkeypatch) -> None:
    subs = ["r/Notion"]
    sentinel = make_cache([post("p9", [comment("z")])], topic="Notion", subreddits=subs)
    monkeypatch.setattr(
        commands, "_fetch_to_cache", lambda **k: (sentinel, tmp_path / "f.json")
    )
    cache = commands._resolve_cache(
        topic="Notion", subreddits=subs, use_cached=True, cache_dir=str(tmp_path)
    )
    assert cache.posts[0].id == "p9"


def test_stale_cache_warns_but_proceeds(tmp_path, monkeypatch, capsys) -> None:
    subs = ["r/Notion"]
    _write_query_cache(tmp_path, "Notion", subs, days_ago=45)
    monkeypatch.setattr(
        commands, "_fetch_to_cache", lambda **k: pytest.fail("should not fetch")
    )
    cache = commands._resolve_cache(
        topic="Notion", subreddits=subs, use_cached=True, cache_dir=str(tmp_path)
    )
    assert cache.metadata.topic == "Notion"  # proceeded
    err = capsys.readouterr().err
    assert "TTL" in err and "45" in err
