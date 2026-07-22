"""Query normalization, slugging, and stable hashing (the cache key)."""

from __future__ import annotations

from reddit_research.fetch.query import (
    cache_filename,
    normalized_subreddits,
    query_hash,
    slugify,
    subreddit_name,
)
from reddit_research.models.cache import FetchParams


def test_slugify_lowercases_and_dashes() -> None:
    assert slugify("Notion") == "notion"
    assert slugify("Project Management!") == "project-management"
    assert slugify("  C++  Tips ") == "c-tips"


def test_subreddit_name_strips_prefix_case_insensitively() -> None:
    assert subreddit_name("r/Notion") == "Notion"
    assert subreddit_name("R/Notion") == "Notion"
    assert subreddit_name("Notion") == "Notion"


def test_normalized_subreddits_sorts_dedupes_and_prefixes() -> None:
    assert normalized_subreddits(["r/Notion", "productivity", "r/notion"]) == [
        "r/Notion",
        "r/productivity",
    ]


PARAMS = FetchParams(posts_per_subreddit=25, time_window_days=90, sort="top")


def test_query_hash_is_stable_for_same_normalized_query() -> None:
    a = query_hash("Notion", ["r/productivity", "r/Notion"], PARAMS)
    b = query_hash("notion", ["r/Notion", "r/productivity"], PARAMS)  # order + case differ
    assert a == b


def test_query_hash_changes_with_topic_subreddits_and_fetch_params() -> None:
    base = query_hash("Notion", ["r/Notion"], PARAMS)
    assert base != query_hash("Obsidian", ["r/Notion"], PARAMS)
    assert base != query_hash("Notion", ["r/productivity"], PARAMS)
    assert base != query_hash(
        "Notion", ["r/Notion"], FetchParams(posts_per_subreddit=50, time_window_days=90, sort="top")
    )
    assert base != query_hash(
        "Notion", ["r/Notion"], FetchParams(posts_per_subreddit=25, time_window_days=30, sort="top")
    )
    assert base != query_hash(
        "Notion", ["r/Notion"], FetchParams(posts_per_subreddit=25, time_window_days=90, sort="hot")
    )


def test_cache_filename_shape() -> None:
    h = query_hash("Notion", ["r/Notion"], PARAMS)
    name = cache_filename("Notion", h)
    assert name == f"notion__{h}.json"
    assert len(h) == 8
