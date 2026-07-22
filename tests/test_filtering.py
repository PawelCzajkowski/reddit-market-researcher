"""Code filter: score cutoff AND topic-keyword relevance (0 tokens)."""

from __future__ import annotations

from reddit_research.analyze.filtering import filter_comments, is_relevant
from reddit_research.models.cache import CachedComment, CachedPost, RawCache

from .cache_fixtures import make_cache


def _comment(cid: str, *, score: int, body: str) -> CachedComment:
    return CachedComment(
        id=cid,
        parent_id="t3_p",
        body=body,
        author="u/x",
        score=score,
        created_utc="2026-06-01T00:00:00Z",
        permalink=f"https://www.reddit.com/r/Notion/comments/p/x/{cid}/",
        depth=0,
        post_id="p",
        post_title="Post",
        subreddit="r/Notion",
    )


def test_is_relevant_matches_topic_token_case_insensitively() -> None:
    assert is_relevant("I love NOTION so much", "Notion")
    assert not is_relevant("unrelated chatter here", "Notion")


def test_is_relevant_multiword_topic_any_token() -> None:
    assert is_relevant("great project tool", "project management")
    assert not is_relevant("nothing to see", "project management")


def test_filter_drops_below_min_score_and_irrelevant() -> None:
    post = CachedPost(
        id="p", subreddit="r/Notion", title="Post", selftext="", author="u/op",
        score=10, num_comments=3, created_utc="2026-06-01T00:00:00Z",
        permalink="https://www.reddit.com/r/Notion/comments/p/", url="x",
        comments=[
            _comment("a", score=10, body="Notion is slow"),      # keep
            _comment("b", score=2, body="Notion is great"),       # drop: score
            _comment("c", score=50, body="totally unrelated"),    # drop: relevance
            _comment("d", score=5, body="switching from notion"), # keep: boundary
        ],
    )
    cache = RawCache(metadata=make_cache([]).metadata, posts=[post])
    kept = filter_comments(cache, min_score=5, topic="Notion")
    assert {c.id for c in kept} == {"a", "d"}
