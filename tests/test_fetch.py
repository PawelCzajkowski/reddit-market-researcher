"""Fetch stage: PRAW query -> frozen RawCache, driven by a fake Reddit client."""

from __future__ import annotations

from datetime import datetime, timezone

from reddit_research.cache_store import read_cache, write_cache
from reddit_research.fetch.reddit import fetch_corpus
from reddit_research.models.cache import FetchParams, RawCache

from .fakes import FakeAuthor, FakeComment, FakeReddit, FakeSubmission

NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)
DAY = 86400.0


def _comment(cid: str, score: int, *, depth: int = 0, author: str | None = "user") -> FakeComment:
    return FakeComment(
        id=cid,
        parent_id="t3_post1",
        body=f"comment {cid}",
        score=score,
        created_utc=NOW.timestamp() - DAY,
        permalink=f"/r/Notion/comments/post1/x/{cid}/",
        depth=depth,
        author=FakeAuthor(author) if author else None,
    )


def _submission(sid: str, score: int, *, age_days: float, ncomments: int = 2) -> FakeSubmission:
    comments = [_comment(f"{sid}c{i}", score=i, author=None if i == 0 else "u") for i in range(ncomments)]
    return FakeSubmission(
        id=sid,
        title=f"Post {sid}",
        score=score,
        created_utc=NOW.timestamp() - age_days * DAY,
        permalink=f"/r/Notion/comments/{sid}/",
        num_comments=99,
        selftext="body",
        url="https://example.com",
        author=FakeAuthor("op"),
        _comments=comments,
    )


def _reddit() -> FakeReddit:
    return FakeReddit(
        {
            "Notion": [
                _submission("in1", score=100, age_days=10),
                _submission("in2", score=50, age_days=80),
                _submission("old", score=999, age_days=200),  # outside 90-day window
            ],
            "productivity": [_submission("p1", score=20, age_days=5)],
        }
    )


PARAMS = FetchParams(posts_per_subreddit=25, time_window_days=90, sort="top")


def test_fetch_produces_valid_rawcache() -> None:
    cache = fetch_corpus(
        _reddit(),
        topic="Notion",
        subreddits=["r/Notion", "r/productivity"],
        params=PARAMS,
        now=NOW,
        praw_version="7.7.1",
    )
    assert isinstance(cache, RawCache)
    RawCache.model_validate(cache.model_dump())  # round-trips through the schema


def test_time_window_filters_old_posts() -> None:
    cache = fetch_corpus(
        _reddit(), topic="Notion", subreddits=["r/Notion"], params=PARAMS, now=NOW, praw_version="x"
    )
    ids = {p.id for p in cache.posts}
    assert ids == {"in1", "in2"}  # "old" (200 days) dropped


def test_all_comments_stored_regardless_of_score() -> None:
    cache = fetch_corpus(
        _reddit(), topic="Notion", subreddits=["r/Notion"], params=PARAMS, now=NOW, praw_version="x"
    )
    # score=0 comments are kept — cutoff is an analyze-time knob, not applied here
    scores = [c.score for p in cache.posts for c in p.comments]
    assert 0 in scores


def test_top_n_selection_per_subreddit() -> None:
    params = FetchParams(posts_per_subreddit=1, time_window_days=90, sort="top")
    cache = fetch_corpus(
        _reddit(), topic="Notion", subreddits=["r/Notion"], params=params, now=NOW, praw_version="x"
    )
    assert [p.id for p in cache.posts] == ["in1"]  # highest score in-window


def test_comment_carries_denormalized_post_context() -> None:
    cache = fetch_corpus(
        _reddit(), topic="Notion", subreddits=["r/Notion"], params=PARAMS, now=NOW, praw_version="x"
    )
    c = cache.posts[0].comments[0]
    assert c.post_id == "in1"
    assert c.post_title == "Post in1"
    assert c.subreddit == "r/Notion"
    assert c.permalink.startswith("https://www.reddit.com/")
    assert c.parent_id == "t3_post1"
    assert c.author is None  # deleted/unavailable author -> null


def test_metadata_counts_and_fields() -> None:
    cache = fetch_corpus(
        _reddit(),
        topic="Notion",
        subreddits=["r/Notion", "r/productivity"],
        params=PARAMS,
        now=NOW,
        praw_version="7.7.1",
    )
    m = cache.metadata
    assert m.post_count == len(cache.posts) == 3  # in1, in2, p1
    assert m.comment_count == sum(len(p.comments) for p in cache.posts)
    assert m.praw_version == "7.7.1"
    assert m.fetched_at == "2026-07-22T00:00:00Z"
    by = {s.subreddit: s.post_count for s in m.by_subreddit}
    assert by == {"r/Notion": 2, "r/productivity": 1}


def test_query_hash_stable_and_roundtrips_to_disk(tmp_path) -> None:
    cache = fetch_corpus(
        _reddit(), topic="Notion", subreddits=["r/Notion"], params=PARAMS, now=NOW, praw_version="x"
    )
    path = tmp_path / f"notion__{cache.metadata.query_hash}.json"
    write_cache(cache, path)
    reloaded = read_cache(path)
    assert reloaded.metadata.query_hash == cache.metadata.query_hash
    assert reloaded.model_dump() == cache.model_dump()
