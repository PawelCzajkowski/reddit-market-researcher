"""Command handlers — the glue between CLI flags and the domain stages.

Each handler is invoked by `cli.py` after flags are parsed and `.env` is loaded.
"""

from __future__ import annotations

import typer


def fetch_command(
    *,
    topic: str,
    subreddits: list[str],
    posts_per_subreddit: int,
    time_window_days: int,
    sort: str,
) -> None:
    typer.echo(
        f"[stub] fetch topic={topic!r} subreddits={subreddits} "
        f"posts_per_subreddit={posts_per_subreddit} "
        f"time_window_days={time_window_days} sort={sort}"
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
