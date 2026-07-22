# Fetch-and-Cache Stage Design (D3, #7) — LOCKED

How the fetch stage turns `(topic, subreddits, params)` into a frozen raw-data artifact the analyze stage consumes without re-fetching. Builds on [R1](../research/reddit-acquisition.md) (acquisition mechanics) and [D1](output-schema.md) (the `Quote` fields the cache must supply).

## Locked decisions

1. **One JSON file per unique query**, query-keyed (no timestamp — see §naming). Structure: `metadata` + `posts[]`, each post with a **flat comment list**.
2. **Comments are flat per post**, each carrying `parent_id` + `depth` (threading recoverable) and **denormalized post context** (`post_id`, `post_title`, `subreddit`, `permalink`) so any comment becomes a D1 `Quote` with no re-fetch.
3. **Filter split = C (store raw, filter at analyze):** the *post selection* (90-day window + top-N) runs at fetch because it defines the API calls; **all comments are stored** regardless of score. The **comment score cutoff and relevance filter are analyze-time knobs** — re-tunable against the frozen corpus without re-fetching or re-paying Reddit. Consequently these are **not** part of the cache key.
4. **Rate limits:** handled by PRAW's automatic `X-Ratelimit` throttling (R1). A default run is ~75–100 requests, far under 100 QPM — no custom backoff needed beyond a descriptive `user_agent` and PRAW's default `ratelimit_seconds`.
5. **Cache is an ephemeral working cache:** internal `fetched_at`; `--use-cached` warns (but proceeds) past a **30-day TTL**; a `cache purge --older-than <days>` command exists. Satisfies R1's "honor deletions" in spirit without forcing aggressive re-fetch.

## Query → Reddit API mapping (per R1)

PRAW read-only ("script" app). Per subreddit:

```python
posts = []
for sub in subreddits:
    hits = reddit.subreddit(sub.lstrip("r/")).search(
        topic, sort="top", time_filter="year", limit=None
    )
    recent = [p for p in hits if p.created_utc >= now - time_window_days*86400]
    for p in recent[:posts_per_subreddit]:          # default 25
        p.comments.replace_more(limit=0)            # drop "load more", keep loaded comments
        posts.append(build_cached_post(p, p.comments.list()))  # store ALL comments
```

- **90-day window has no direct `time_filter`** → query `time_filter="year"` (top by score), then filter client-side on `created_utc`, then take the top `posts_per_subreddit`. If fewer than N in-window, take what's there.
- `sort="top"` gives the highest-signal posts for market research; `sort` is a fetch param (default `top`).

## File naming & identity

- **Fetch cache (this stage):** `cache/<topic-slug>__<queryhash>.json` — e.g. `cache/notion__a1b2c3d4.json`.
  - `queryhash` = short stable hash of the **normalized query**: `topic` + **sorted** `subreddits` + **fetch params** (`posts_per_subreddit`, `time_window_days`, `sort`). Score cutoff/relevance are excluded (analyze-time).
  - **One file per query.** `--use-cached`: file for this query exists → **reuse**; else **fetch**. A fresh fetch **overwrites**. Optional `--use-cached <path>` pins an exact file.
  - Default (no flag) = fetch fresh.
- **Results file (analyze stage — recorded here, owned by D2/CLI):** `results/<topic-slug>__<YYYYMMDDTHHMMSSZ>.json` + matching `.md` report. **Timestamped, accumulating** — each analysis run yields a new pair, building a diffable history as prompts are tuned. (User preference: timestamp belongs on the *results*, not the fetch cache.)

## Schema (Pydantic)

```python
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel

CACHE_SCHEMA_VERSION = "1.0"

class FetchParams(BaseModel):
    posts_per_subreddit: int = 25
    time_window_days: int = 90
    sort: str = "top"                 # top | hot | new | relevance | comments

class CachedComment(BaseModel):
    id: str
    parent_id: str                    # parent comment id, or the post id for top-level
    body: str
    author: Optional[str]             # null if deleted/unavailable
    score: int
    created_utc: str                  # ISO-8601 UTC
    permalink: str                    # full https URL to the comment
    depth: int
    # denormalized post context → D1 Quote needs no re-fetch:
    post_id: str
    post_title: str
    subreddit: str

class CachedPost(BaseModel):
    id: str
    subreddit: str
    title: str
    selftext: str                     # "" for link posts
    author: Optional[str]
    score: int
    num_comments: int                 # Reddit's count (may exceed stored, per R1)
    created_utc: str
    permalink: str                    # full https URL to the post
    url: str                          # external link, or self permalink
    comments: list[CachedComment]     # ALL loaded comments (replace_more(limit=0))

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
    fetched_at: str                   # ISO-8601 UTC — drives TTL
    praw_version: str
    post_count: int
    comment_count: int
    by_subreddit: list[SubredditCount]

class RawCache(BaseModel):
    metadata: CacheMetadata
    posts: list[CachedPost]
```

## Quote-field coverage check (D1 constraint)

D1 `Quote` needs: `text`, `author`, `type`, `permalink`, `subreddit`, `post_title`, `score`, `created_at`.
- **Comment → Quote:** `body`→text, `author`, type=`comment`, `permalink`, `subreddit`, `post_title`, `score`, `created_utc`→created_at. ✅
- **Post → Quote:** `title`/`selftext`→text, `author`, type=`post`, `permalink`, `subreddit`, `title`→post_title, `score`, `created_utc`. ✅

## Handoffs

- **D2 (pipeline):** applies the comment **score cutoff** (default ≥5) and any **relevance filter** at analyze time over `RawCache`; owns the timestamped `results/` naming recorded above; reads `metadata` to populate D1 `run_metadata`.
- **S1 (spec):** ToS — use is personal/non-commercial (free tier); cache is ephemeral (TTL + purge); confirm OpenAI API no-training.
