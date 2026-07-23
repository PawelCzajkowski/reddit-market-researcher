"""Query identity: normalization, slugging, and the stable cache key.

The cache key (`query_hash`) is a short stable hash of the *normalized* query —
topic + sorted subreddits + fetch params (posts_per_subreddit, time_window_days,
sort). The comment score cutoff and relevance filter are deliberately excluded:
they are analyze-time knobs re-tunable against the same frozen corpus (D3 §3).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..models.cache import FetchParams

_HASH_LEN = 8
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(topic: str) -> str:
    """A filesystem-safe lowercase slug from a topic string."""
    slug = _SLUG_RE.sub("-", topic.strip().lower()).strip("-")
    return slug or "topic"


def subreddit_name(sub: str) -> str:
    """Bare subreddit name with any leading `r/` (case-insensitive) removed."""
    return re.sub(r"^r/", "", sub.strip(), flags=re.IGNORECASE)


def normalized_subreddits(subreddits: list[str]) -> list[str]:
    """Canonical display list: `r/`-prefixed, deduped, case-insensitively sorted.

    Dedup is case-insensitive (Reddit names are case-insensitive); the first
    casing seen wins for display.
    """
    seen: dict[str, str] = {}
    for sub in subreddits:
        name = subreddit_name(sub)
        if not name:
            continue
        seen.setdefault(name.lower(), f"r/{name}")
    return [seen[k] for k in sorted(seen)]


def query_hash(topic: str, subreddits: list[str], params: FetchParams) -> str:
    """Short stable hash of the normalized query (order/case-insensitive)."""
    names = [n for n in (subreddit_name(s).lower() for s in subreddits) if n]
    canonical = {
        "topic": topic.strip().lower(),
        "subreddits": sorted(names),
        "posts_per_subreddit": params.posts_per_subreddit,
        "time_window_days": params.time_window_days,
        "sort": params.sort,
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:_HASH_LEN]


def cache_filename(topic: str, hash_: str) -> str:
    return f"{slugify(topic)}__{hash_}.json"


def cache_path(topic: str, hash_: str, cache_dir: str | Path) -> Path:
    return Path(cache_dir) / cache_filename(topic, hash_)
