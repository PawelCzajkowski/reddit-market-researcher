"""Analyze-stage orchestration: RawCache -> AnalysisResult (D2 pipeline).

For #11 this is the tracer bullet: filter (code) -> map/extract (LLM) -> overall
sentiment (code) -> AnalysisResult. Themes, feature requests, and competitor
mentions are empty here; later tickets layer canonicalize/aggregate/summarize in.
The structured map model is injected so the whole pipeline runs in tests with a
fake and never calls OpenAI.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..models.cache import RawCache
from ..models.output import (
    AnalysisResult,
    Corpus,
    Models,
    Overall,
    RunMetadata,
    RunParams,
    SubredditCorpus,
)
from .extract import StructuredModel, run_map
from .filtering import filter_comments
from .sentiment import aggregate_sentiment

SUMMARY_PLACEHOLDER = "Executive summary pending."


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_metadata(
    cache: RawCache, *, min_score: int, model_id: str, generated_at: datetime
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
        models=Models(map=model_id, reduce=model_id),
    )


def run_analysis(
    cache: RawCache,
    *,
    map_model: StructuredModel,
    min_score: int,
    model_id: str,
    generated_at: datetime | None = None,
) -> AnalysisResult:
    generated_at = generated_at or datetime.now(timezone.utc)
    topic = cache.metadata.topic

    filtered = filter_comments(cache, min_score=min_score, topic=topic)
    extracts = run_map(map_model, filtered, topic=topic)
    overall_sentiment = aggregate_sentiment(
        (e.sentiment.label, e.sentiment.score) for e in extracts
    )

    return AnalysisResult(
        run_metadata=_run_metadata(
            cache, min_score=min_score, model_id=model_id, generated_at=generated_at
        ),
        overall=Overall(
            executive_summary=SUMMARY_PLACEHOLDER, sentiment=overall_sentiment
        ),
        themes=[],
        feature_requests=[],
        competitor_mentions=[],
    )
