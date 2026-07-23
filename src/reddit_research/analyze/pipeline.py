"""Analyze-stage orchestration: RawCache -> AnalysisResult (D2 pipeline).

filter (code) -> map/extract (LLM, batched) -> canonicalize themes (LLM) ->
aggregate themes/feature-requests/competitors (code) -> summarize (LLM). The
canonicalizer and summarizer are optional injected steps: when omitted (or when a
preflight gate aborts) the run degrades gracefully to overall sentiment only. All
LLM steps are injected seams, so the whole pipeline runs in tests with fakes and
never calls OpenAI.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from ..models.cache import CachedComment, RawCache
from ..models.output import (
    AnalysisResult,
    Corpus,
    Models,
    Overall,
    RunMetadata,
    RunParams,
    SubredditCorpus,
)
from .aggregate import (
    aggregate_competitors,
    aggregate_feature_requests,
    aggregate_themes,
    pair,
)
from .canonicalize import ThemeTaxonomy
from .extract import CommentExtract, StructuredModel, run_map
from .filtering import filter_comments
from .sentiment import aggregate_sentiment
from .summarize import SummaryContext

SUMMARY_PLACEHOLDER = "Executive summary pending."

Summarizer = Callable[[SummaryContext], str]
# Invoked after the code filter and before any paid LLM call; may raise to abort.
Preflight = Callable[[list[CachedComment]], None]
# (label, count) pairs + raw competitor names + topic -> canonicalized taxonomy.
Canonicalizer = Callable[[list[tuple[str, int]], list[str], str], ThemeTaxonomy]


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_metadata(
    cache: RawCache,
    *,
    min_score: int,
    model_id: str,
    reduce_model_id: str,
    generated_at: datetime,
) -> RunMetadata:
    m = cache.metadata
    return RunMetadata(
        topic=m.topic,
        subreddits=m.subreddits,
        params=RunParams(
            posts_per_subreddit=m.params.posts_per_subreddit,
            time_window_days=m.params.time_window_days,
            comment_min_score=min_score,
            sort=m.params.sort,
        ),
        generated_at=_iso(generated_at),
        corpus=Corpus(
            post_count=m.post_count,
            comment_count=m.comment_count,
            by_subreddit=[
                SubredditCorpus(
                    subreddit=s.subreddit,
                    post_count=s.post_count,
                    comment_count=s.comment_count,
                )
                for s in m.by_subreddit
            ],
        ),
        models=Models(map=model_id, reduce=reduce_model_id),
    )


def _canonicalize(
    canonicalizer: Canonicalizer, extracts: list[CommentExtract], *, topic: str
) -> ThemeTaxonomy:
    from .canonicalize import label_counts

    labels = [lbl for e in extracts for lbl in e.candidate_theme_labels]
    competitors = [cm.name for e in extracts for cm in e.competitor_mentions]
    return canonicalizer(label_counts(labels), competitors, topic)


def run_analysis(
    cache: RawCache,
    *,
    map_model: StructuredModel,
    min_score: int,
    model_id: str,
    reduce_model_id: str | None = None,
    canonicalizer: Canonicalizer | None = None,
    summarizer: Summarizer | None = None,
    preflight: Preflight | None = None,
    generated_at: datetime | None = None,
) -> AnalysisResult:
    generated_at = generated_at or datetime.now(timezone.utc)
    reduce_model_id = reduce_model_id or model_id
    topic = cache.metadata.topic

    filtered = filter_comments(cache, min_score=min_score, topic=topic)
    if preflight is not None:
        preflight(filtered)  # cost gate — runs before any paid LLM call
    extracts = run_map(map_model, filtered, topic=topic)
    overall_sentiment = aggregate_sentiment(
        (e.sentiment.label, e.sentiment.score) for e in extracts
    )

    themes: list = []
    feature_requests: list = []
    competitor_mentions: list = []

    if canonicalizer is not None:
        analyzed = pair(filtered, extracts)
        taxonomy = _canonicalize(canonicalizer, extracts, topic=topic)
        themes = aggregate_themes(analyzed, taxonomy, total_analyzed=len(filtered))
        feature_requests = aggregate_feature_requests(
            analyzed, total_analyzed=len(filtered)
        )
        competitor_mentions = aggregate_competitors(analyzed, taxonomy)

    if summarizer is not None:
        context = SummaryContext(
            topic=topic,
            subreddits=cache.metadata.subreddits,
            overall=overall_sentiment,
            analyzed_comment_count=len(filtered),
            themes=themes,
            feature_requests=feature_requests,
            competitor_mentions=competitor_mentions,
        )
        executive_summary = summarizer(context)
    else:
        executive_summary = SUMMARY_PLACEHOLDER

    return AnalysisResult(
        run_metadata=_run_metadata(
            cache,
            min_score=min_score,
            model_id=model_id,
            reduce_model_id=reduce_model_id,
            generated_at=generated_at,
        ),
        overall=Overall(
            executive_summary=executive_summary, sentiment=overall_sentiment
        ),
        themes=themes,
        feature_requests=feature_requests,
        competitor_mentions=competitor_mentions,
    )
