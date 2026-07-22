"""Command-line interface (SPEC §4).

    reddit-research fetch   --topic ... --subreddits r/a r/b [--posts-per-subreddit N]
                            [--time-window-days N] [--sort top]
    reddit-research analyze --topic ... --subreddits r/a r/b [--use-cached]
                            [--min-score N] [--model gpt-5.4-mini] [--yes]
    reddit-research cache purge --older-than N

This module is a thin adapter: it parses flags and delegates to `commands`.
Argument shape follows the spec; `--subreddits` accepts either repeated flags
or space/comma-separated values within one flag.
"""

from __future__ import annotations

from typing import Optional

import typer

from . import defaults
from .config import ConfigError, load_env

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Personal, non-commercial market research on Reddit.",
)
cache_app = typer.Typer(no_args_is_help=True, help="Manage the fetch cache.")
app.add_typer(cache_app, name="cache")


def _normalize_subreddits(raw: list[str]) -> list[str]:
    """Flatten space/comma-separated values into a clean subreddit list."""
    out: list[str] = []
    for value in raw:
        for token in value.replace(",", " ").split():
            token = token.strip()
            if token and token not in out:
                out.append(token)
    return out


@app.command()
def fetch(
    topic: str = typer.Option(..., "--topic", help="Topic/keyword to search for."),
    subreddits: list[str] = typer.Option(
        ..., "--subreddits", help="Subreddits to search, e.g. r/productivity r/Notion."
    ),
    posts_per_subreddit: int = typer.Option(
        defaults.POSTS_PER_SUBREDDIT, "--posts-per-subreddit"
    ),
    time_window_days: int = typer.Option(defaults.TIME_WINDOW_DAYS, "--time-window-days"),
    sort: str = typer.Option(defaults.SORT, "--sort"),
) -> None:
    """Fetch matching discussion from Reddit into a frozen cache file."""
    from . import commands

    load_env()
    subs = _normalize_subreddits(subreddits)
    try:
        commands.fetch_command(
            topic=topic,
            subreddits=subs,
            posts_per_subreddit=posts_per_subreddit,
            time_window_days=time_window_days,
            sort=sort,
        )
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command()
def analyze(
    topic: str = typer.Option(..., "--topic", help="Topic/keyword analyzed."),
    subreddits: list[str] = typer.Option(..., "--subreddits"),
    use_cached: bool = typer.Option(
        False, "--use-cached", help="Reuse the query-keyed cache instead of fetching fresh."
    ),
    min_score: int = typer.Option(
        defaults.COMMENT_MIN_SCORE, "--min-score", help="Comment score cutoff."
    ),
    model: str = typer.Option(defaults.DEFAULT_MODEL, "--model", help="LLM model id."),
    yes: bool = typer.Option(
        False, "--yes", help="Skip the pre-flight cost confirmation."
    ),
) -> None:
    """Analyze the discussion and write a timestamped JSON + Markdown report."""
    from . import commands

    load_env()
    subs = _normalize_subreddits(subreddits)
    try:
        commands.analyze_command(
            topic=topic,
            subreddits=subs,
            use_cached=use_cached,
            min_score=min_score,
            model=model,
            yes=yes,
        )
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@cache_app.command("purge")
def cache_purge(
    older_than: int = typer.Option(
        ..., "--older-than", help="Delete cache files older than this many days."
    ),
) -> None:
    """Delete cache files older than the given age."""
    from . import commands

    commands.cache_purge_command(older_than_days=older_than)


if __name__ == "__main__":  # pragma: no cover
    app()
