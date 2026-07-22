"""Executive summary (fake LLM) + pure Markdown rendering."""

from __future__ import annotations

from datetime import datetime, timezone

from reddit_research.analyze.pipeline import run_analysis
from reddit_research.analyze.summarize import (
    SummaryContext,
    SummaryOutput,
    build_summary_input,
    run_summary,
)
from reddit_research.models.output import (
    AnalysisResult,
    Corpus,
    Models,
    Overall,
    RunMetadata,
    RunParams,
    SentimentBlock,
    SentimentDistribution,
    SubredditCorpus,
)
from reddit_research.report import render_report

from .cache_fixtures import comment, make_cache, post
from .test_pipeline import FakeMapModel

GEN = datetime(2026, 7, 22, 14, 30, 0, tzinfo=timezone.utc)


class FakeSummaryModel:
    """Echoes the aggregate numbers back, proving the summary is grounded in them."""

    def __init__(self) -> None:
        self.seen_input = None

    def invoke(self, input) -> SummaryOutput:
        self.seen_input = input
        user = input[1]["content"]
        return SummaryOutput(executive_summary=f"SUMMARY based on: {user[:40]}")


def _result() -> AnalysisResult:
    return AnalysisResult(
        run_metadata=RunMetadata(
            topic="Notion",
            subreddits=["r/Notion", "r/productivity"],
            params=RunParams(
                posts_per_subreddit=25, time_window_days=90, comment_min_score=5, sort="top"
            ),
            generated_at="2026-07-22T14:30:00Z",
            corpus=Corpus(
                post_count=2,
                comment_count=40,
                by_subreddit=[
                    SubredditCorpus(subreddit="r/Notion", post_count=1, comment_count=25),
                    SubredditCorpus(subreddit="r/productivity", post_count=1, comment_count=15),
                ],
            ),
            models=Models(map="gpt-5.4-mini", reduce="gpt-5.4-mini"),
        ),
        overall=Overall(
            executive_summary="Users are broadly mixed on Notion.",
            sentiment=SentimentBlock(
                label="mixed",
                score=-0.12,
                distribution=SentimentDistribution(positive=10, negative=18, neutral=12),
            ),
        ),
        themes=[],
        feature_requests=[],
        competitor_mentions=[],
    )


def test_summary_input_contains_real_numbers() -> None:
    ctx = SummaryContext(
        topic="Notion",
        subreddits=["r/Notion"],
        overall=SentimentBlock(
            label="mixed",
            score=-0.12,
            distribution=SentimentDistribution(positive=10, negative=18, neutral=12),
        ),
        analyzed_comment_count=40,
        themes=[],
        feature_requests=[],
        competitor_mentions=[],
    )
    text = build_summary_input(ctx)[1]["content"]
    assert "40" in text
    assert "10 positive" in text
    assert "mixed" in text


def test_run_summary_returns_prose() -> None:
    ctx = SummaryContext(
        topic="Notion",
        subreddits=["r/Notion"],
        overall=SentimentBlock(
            label="mixed", score=0.0, distribution=SentimentDistribution(positive=1, negative=1, neutral=1)
        ),
        analyzed_comment_count=3,
        themes=[],
        feature_requests=[],
        competitor_mentions=[],
    )
    out = run_summary(FakeSummaryModel(), ctx)
    assert out.startswith("SUMMARY based on:")


def test_pipeline_uses_summarizer_output() -> None:
    cache = make_cache([post("p1", [comment("c1", body="I love notion", score=20)])])
    summary_model = FakeSummaryModel()
    result = run_analysis(
        cache,
        map_model=FakeMapModel(),
        min_score=5,
        model_id="gpt-5.4-mini",
        summarizer=lambda ctx: run_summary(summary_model, ctx),
        generated_at=GEN,
    )
    assert result.overall.executive_summary.startswith("SUMMARY based on:")


def test_report_is_pure_rendering_of_json() -> None:
    md = render_report(_result())
    assert "# Reddit market research: Notion" in md
    assert "Users are broadly mixed on Notion." in md
    assert "**mixed** (score -0.12)" in md
    assert "10 positive / 18 negative / 12 neutral" in md
    assert "r/Notion: 1 posts / 25 comments" in md
    assert "map `gpt-5.4-mini`" in md
