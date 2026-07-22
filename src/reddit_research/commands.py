"""Command handlers — the glue between CLI flags and the domain stages.

Each handler is invoked by `cli.py` after flags are parsed and `.env` is loaded.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer

from . import defaults
from .analyze.pipeline import run_analysis
from .analyze.results import result_paths
from .cache_store import read_cache, write_cache
from .config import OpenAICredentials, RedditCredentials
from .fetch.query import cache_path, query_hash
from .fetch.reddit import build_reddit, fetch_corpus
from .models.cache import FetchParams, RawCache
from .report import render_report


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


def _resolve_cache(
    *, topic: str, subreddits: list[str], use_cached: bool, cache_dir: str
) -> RawCache:
    """Reuse the query-keyed cache when `--use-cached` and it exists; else fetch fresh.

    Analyze has no fetch-param flags (SPEC §4), so the query key is computed with
    default fetch params — the same key `fetch` writes with defaults.
    """
    params = FetchParams()
    path = cache_path(topic, query_hash(topic, subreddits, params), cache_dir)
    if use_cached and path.exists():
        typer.echo(f"Using cached corpus: {path}")
        return read_cache(path)
    cache, path = _fetch_to_cache(
        topic=topic, subreddits=subreddits, params=params, cache_dir=cache_dir
    )
    typer.echo(f"Fetched fresh corpus -> {path}")
    return cache


def analyze_command(
    *,
    topic: str,
    subreddits: list[str],
    use_cached: bool,
    min_score: int,
    model: str,
    yes: bool,
) -> None:
    from .analyze.canonicalize import run_canonicalize
    from .analyze.cost import estimate_cost
    from .analyze.llm import (
        build_canonicalize_model,
        build_map_model,
        build_summary_model,
    )
    from .analyze.summarize import SummaryContext, run_summary
    from .config import langsmith_enabled
    from .models.cache import CachedComment

    OpenAICredentials.from_env()  # fail early with a clear error if the key is missing
    cache = _resolve_cache(
        topic=topic,
        subreddits=subreddits,
        use_cached=use_cached,
        cache_dir=defaults.CACHE_DIR,
    )

    if langsmith_enabled():
        typer.secho(
            "LangSmith tracing is ON — per-run cost will be reported by LangSmith. "
            "Note: the Reddit comment text sent to the model is included in traces "
            "uploaded to LangSmith's cloud.",
            fg=typer.colors.YELLOW,
        )

    def preflight(filtered: list[CachedComment]) -> None:
        est = estimate_cost(filtered, topic=topic, model_id=model)
        note = "" if est.price_known else " (no local price for this model; conservative estimate)"
        typer.echo(
            f"Pre-flight estimate: ~${est.projected_usd:.2f} for {len(filtered)} comments "
            f"(~{est.input_tokens:,} in / ~{est.output_tokens:,} out tokens){note}"
        )
        if est.projected_usd > defaults.COST_SOFT_CEILING_USD and not yes:
            typer.secho(
                f"Projected cost ${est.projected_usd:.2f} exceeds the "
                f"${defaults.COST_SOFT_CEILING_USD:.2f} soft ceiling.",
                fg=typer.colors.YELLOW,
            )
            if not typer.confirm("Proceed with the analysis?"):
                raise typer.Abort()

    generated_at = datetime.now(timezone.utc)
    map_model = build_map_model(model)
    canonicalize_model = build_canonicalize_model(model)
    summary_model = build_summary_model(model)

    def canonicalizer(counts, competitors, topic_):  # noqa: ANN001, ANN202
        return run_canonicalize(canonicalize_model, counts, competitors, topic=topic_)

    def summarizer(ctx: SummaryContext) -> str:
        return run_summary(summary_model, ctx)

    result = run_analysis(
        cache,
        map_model=map_model,
        min_score=min_score,
        model_id=model,
        canonicalizer=canonicalizer,
        summarizer=summarizer,
        preflight=preflight,
        generated_at=generated_at,
    )

    json_path, md_path = result_paths(topic, generated_at, defaults.RESULTS_DIR)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_report(result), encoding="utf-8")
    typer.echo(
        f"Analyzed {result.run_metadata.corpus.comment_count} comments; "
        f"overall sentiment={result.overall.sentiment.label} "
        f"({result.overall.sentiment.score:+.2f})\n"
        f"  JSON  -> {json_path}\n"
        f"  report -> {md_path}"
    )


def cache_purge_command(*, older_than_days: int) -> None:
    typer.echo(f"[stub] cache purge older_than_days={older_than_days}")
