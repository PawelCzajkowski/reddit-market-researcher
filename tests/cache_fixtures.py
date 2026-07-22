"""Builders for RawCache fixtures used across analyze-stage tests."""

from __future__ import annotations

from reddit_research.models.cache import (
    CachedComment,
    CachedPost,
    CacheMetadata,
    FetchParams,
    RawCache,
    SubredditCount,
)


def comment(
    cid: str,
    *,
    body: str = "notion comment",
    score: int = 10,
    author: str | None = "u/user",
    subreddit: str = "r/Notion",
    post_id: str = "p1",
    post_title: str = "A post",
    created: str = "2026-06-01T00:00:00Z",
) -> CachedComment:
    return CachedComment(
        id=cid,
        parent_id=f"t3_{post_id}",
        body=body,
        author=author,
        score=score,
        created_utc=created,
        permalink=f"https://www.reddit.com/r/Notion/comments/{post_id}/x/{cid}/",
        depth=0,
        post_id=post_id,
        post_title=post_title,
        subreddit=subreddit,
    )


def post(
    pid: str,
    comments: list[CachedComment],
    *,
    subreddit: str = "r/Notion",
    title: str = "A post",
) -> CachedPost:
    return CachedPost(
        id=pid,
        subreddit=subreddit,
        title=title,
        selftext="",
        author="u/op",
        score=100,
        num_comments=len(comments),
        created_utc="2026-06-01T00:00:00Z",
        permalink=f"https://www.reddit.com/r/{subreddit[2:]}/comments/{pid}/",
        url="https://example.com",
        comments=comments,
    )


def make_cache(
    posts: list[CachedPost],
    *,
    topic: str = "Notion",
    subreddits: list[str] | None = None,
    fetched_at: str = "2026-07-20T00:00:00Z",
) -> RawCache:
    subreddits = subreddits or ["r/Notion"]
    by_sub: dict[str, SubredditCount] = {}
    for p in posts:
        sc = by_sub.setdefault(
            p.subreddit, SubredditCount(subreddit=p.subreddit, post_count=0, comment_count=0)
        )
        by_sub[p.subreddit] = SubredditCount(
            subreddit=p.subreddit,
            post_count=sc.post_count + 1,
            comment_count=sc.comment_count + len(p.comments),
        )
    meta = CacheMetadata(
        topic=topic,
        subreddits=subreddits,
        query_hash="deadbeef",
        params=FetchParams(),
        fetched_at=fetched_at,
        praw_version="7.7.1",
        post_count=len(posts),
        comment_count=sum(len(p.comments) for p in posts),
        by_subreddit=list(by_sub.values()),
    )
    return RawCache(metadata=meta, posts=posts)
