"""Step 1 — code filter (0 tokens): score cutoff AND topic-keyword relevance.

Keep comments with `score >= min_score` AND a topic-keyword match in the body.
Semantic relevance is a future upgrade (SPEC §6); this is a deterministic keyword
match, so the surviving corpus is fully reproducible.
"""

from __future__ import annotations

import re

from ..models.cache import CachedComment, RawCache

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _topic_tokens(topic: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(topic.lower()) if len(t) >= 2]


def is_relevant(body: str, topic: str) -> bool:
    """True when any significant topic token appears in the comment body."""
    text = body.lower()
    tokens = _topic_tokens(topic)
    if not tokens:
        return True  # empty/degenerate topic -> don't over-filter
    return any(token in text for token in tokens)


def filter_comments(
    cache: RawCache, *, min_score: int, topic: str
) -> list[CachedComment]:
    """All comments across all posts passing the score cutoff and relevance match."""
    return [
        c
        for p in cache.posts
        for c in p.comments
        if c.score >= min_score and is_relevant(c.body, topic)
    ]
