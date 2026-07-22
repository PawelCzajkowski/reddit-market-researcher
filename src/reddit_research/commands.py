"""Command handlers — the glue between CLI flags and the domain stages.

Each handler is invoked by `cli.py` after flags are parsed and `.env` is loaded.
"""

from __future__ import annotations

from pathlib import Path

import typer

from . import defaults
from .cache_store import write_cache
from .config import RedditCredentials
from .fetch.query import cache_path
from .fetch.reddit import build_reddit, fetch_corpus
from .models.cache import FetchParams, RawCache


def _fetch_to_cache(
    *, topic: str, subreddits: list[str], params: FetchParams, cache_dir: str
) -> tuple[RawCache, Path]:
    """Fetch fresh from Reddit and write the query-keyed cache file (overwrite)."""
    creds = RedditCredentials.from_env()
    reddit = build_reddit(creds)
    cache = fetch_corpus(reddit, topic=topic, subreddits=subreddits, params=params)
    path = cache_path(topic, cache.metadata.query_hash, cache_dir)
    write_cache(cache, path)
    return cache, path


def fetch_command(
    *,
    topic: str,
    subreddits: list[str],
    posts_per_subreddit: int,
    time_window_days: int,
    sort: str,
) -> None:
    params = FetchParams(
        posts_per_subreddit=posts_per_subreddit,
        time_window_days=time_window_days,
        sort=sort,
    )
    cache, path = _fetch_to_cache(
        topic=topic, subreddits=subreddits, params=params, cache_dir=defaults.CACHE_DIR
    )
    m = cache.metadata
    typer.echo(
        f"Fetched {m.post_count} posts / {m.comment_count} comments "
        f"across {len(m.subreddits)} subreddit(s) -> {path}"
    )


def analyze_command(
    *,
    topic: str,
    subreddits: list[str],
    use_cached: bool,
    min_score: int,
    model: str,
    yes: bool,
) -> None:
    typer.echo(
        f"[stub] analyze topic={topic!r} subreddits={subreddits} "
        f"use_cached={use_cached} min_score={min_score} model={model!r} yes={yes}"
    )


def cache_purge_command(*, older_than_days: int) -> None:
    typer.echo(f"[stub] cache purge older_than_days={older_than_days}")
