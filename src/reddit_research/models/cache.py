"""D3 cache schema — the frozen raw-data artifact the analyze stage consumes.

One JSON file per unique query: `metadata` + `posts[]`, each post with a flat
comment list. Comments carry `parent_id` + `depth` (threading recoverable) and
denormalized post context so any comment becomes a D1 `Quote` with no re-fetch.
See docs/design/cache-schema.md.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

CACHE_SCHEMA_VERSION = "1.0"


class FetchParams(BaseModel):
    posts_per_subreddit: int = 25
    time_window_days: int = 90
    sort: str = "top"  # top | hot | new | relevance | comments


class CachedComment(BaseModel):
    id: str
    parent_id: str  # parent comment id, or the post id for top-level
    body: str
    author: Optional[str]  # null if deleted/unavailable
    score: int
    created_utc: str  # ISO-8601 UTC
    permalink: str  # full https URL to the comment
    depth: int
    # denormalized post context → D1 Quote needs no re-fetch:
    post_id: str
    post_title: str
    subreddit: str


class CachedPost(BaseModel):
    id: str
    subreddit: str
    title: str
    selftext: str  # "" for link posts
    author: Optional[str]
    score: int
    num_comments: int  # Reddit's count (may exceed stored)
    created_utc: str
    permalink: str  # full https URL to the post
    url: str  # external link, or self permalink
    comments: list[CachedComment]  # ALL loaded comments (replace_more(limit=0))


class SubredditCount(BaseModel):
    subreddit: str
    post_count: int
    comment_count: int


class CacheMetadata(BaseModel):
    schema_version: str = CACHE_SCHEMA_VERSION
    topic: str
    subreddits: list[str]
    query_hash: str
    params: FetchParams
    fetched_at: str  # ISO-8601 UTC — drives TTL
    praw_version: str
    post_count: int
    comment_count: int
    by_subreddit: list[SubredditCount]


class RawCache(BaseModel):
    metadata: CacheMetadata
    posts: list[CachedPost]
