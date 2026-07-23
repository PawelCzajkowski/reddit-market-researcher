"""Fetch stage: query Reddit read-only (PRAW) into a frozen RawCache.

The Reddit client is injected into `fetch_corpus`, so the network boundary is a
single seam — tests drive it with a fake client and never touch live Reddit.
Post selection (time-window client filter + top-N) happens here because it
defines the API calls; **all** comments are stored regardless of score, because
the score cutoff and relevance filter are analyze-time knobs (D3 §3).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from ..models.cache import (
    CachedComment,
    CachedPost,
    CacheMetadata,
    FetchParams,
    RawCache,
    SubredditCount,
)
from .query import normalized_subreddits, query_hash, subreddit_name

_REDDIT_BASE = "https://www.reddit.com"


def build_reddit(creds: Any) -> Any:  # pragma: no cover - thin PRAW constructor
    """Construct a read-only PRAW client from Reddit credentials."""
    import praw

    reddit = praw.Reddit(
        client_id=creds.client_id,
        client_secret=creds.client_secret,
        user_agent=creds.user_agent,
    )
    reddit.read_only = True
    return reddit


def _iso(epoch: float) -> str:
    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _author_name(author: Any) -> Optional[str]:
    if author is None:
        return None
    name = getattr(author, "name", None)
    return f"u/{name}" if name else None


def _full_url(permalink: str) -> str:
    if permalink.startswith("http"):
        return permalink
    return f"{_REDDIT_BASE}{permalink}"


def _build_comment(comment: Any, *, post: Any, subreddit: str) -> CachedComment:
    return CachedComment(
        id=str(comment.id),
        parent_id=str(comment.parent_id),
        body=comment.body or "",
        author=_author_name(comment.author),
        score=int(comment.score),
        created_utc=_iso(comment.created_utc),
        permalink=_full_url(comment.permalink),
        depth=int(getattr(comment, "depth", 0)),
        post_id=str(post.id),
        post_title=post.title,
        subreddit=subreddit,
    )


def _build_post(submission: Any, *, subreddit: str) -> CachedPost:
    submission.comments.replace_more(limit=0)
    comments = [
        _build_comment(c, post=submission, subreddit=subreddit)
        for c in submission.comments.list()
    ]
    return CachedPost(
        id=str(submission.id),
        subreddit=subreddit,
        title=submission.title,
        selftext=getattr(submission, "selftext", "") or "",
        author=_author_name(submission.author),
        score=int(submission.score),
        num_comments=int(submission.num_comments),
        created_utc=_iso(submission.created_utc),
        permalink=_full_url(submission.permalink),
        url=getattr(submission, "url", "") or _full_url(submission.permalink),
        comments=comments,
    )


def fetch_corpus(
    reddit: Any,
    *,
    topic: str,
    subreddits: list[str],
    params: FetchParams,
    now: datetime | None = None,
    praw_version: str | None = None,
) -> RawCache:
    """Query each subreddit, client-filter to the window, keep top-N posts."""
    now = now or datetime.now(timezone.utc)
    cutoff = now.timestamp() - params.time_window_days * 86400
    subs = normalized_subreddits(subreddits)

    posts: list[CachedPost] = []
    counts: list[SubredditCount] = []
    for display in subs:  # normalized "r/Name" forms
        name = subreddit_name(display)  # bare name for the PRAW call
        hits = reddit.subreddit(name).search(
            topic, sort=params.sort, time_filter="year", limit=None
        )
        recent = [p for p in hits if p.created_utc >= cutoff]
        recent.sort(key=lambda p: p.score, reverse=True)
        selected = recent[: params.posts_per_subreddit]
        built = [_build_post(p, subreddit=display) for p in selected]
        posts.extend(built)
        counts.append(
            SubredditCount(
                subreddit=display,
                post_count=len(built),
                comment_count=sum(len(p.comments) for p in built),
            )
        )

    if praw_version is None:  # pragma: no cover - trivial import
        try:
            import praw

            praw_version = praw.__version__
        except Exception:
            praw_version = "unknown"

    metadata = CacheMetadata(
        topic=topic,
        subreddits=subs,
        query_hash=query_hash(topic, subreddits, params),
        params=params,
        fetched_at=now.isoformat().replace("+00:00", "Z"),
        praw_version=praw_version,
        post_count=len(posts),
        comment_count=sum(len(p.comments) for p in posts),
        by_subreddit=counts,
    )
    return RawCache(metadata=metadata, posts=posts)
