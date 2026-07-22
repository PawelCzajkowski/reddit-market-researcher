"""Analyze foundation: cache -> filter -> map (fake LLM) -> overall sentiment -> JSON."""

from __future__ import annotations

from datetime import datetime, timezone

from reddit_research.analyze.extract import (
    CommentExtract,
    CommentSentiment,
    MapBatchOutput,
)
from reddit_research.analyze.pipeline import run_analysis
from reddit_research.models.output import AnalysisResult

from .cache_fixtures import comment, make_cache, post

GEN = datetime(2026, 7, 22, 14, 30, 0, tzinfo=timezone.utc)


class FakeMapModel:
    """Deterministic map model: sentiment keyed off a marker in the comment body."""

    def __init__(self) -> None:
        self.calls: list = []

    def _label(self, body: str) -> tuple[str, float]:
        if "love" in body:
            return "positive", 0.8
        if "hate" in body or "slow" in body:
            return "negative", -0.7
        return "neutral", 0.0

    def batch(self, inputs: list) -> list[MapBatchOutput]:
        self.calls.append(inputs)
        outputs = []
        for messages in inputs:
            user = messages[1]["content"]
            extracts = []
            for line in user.splitlines():
                if line.startswith("- ["):
                    cid = line[line.index("[") + 1 : line.index("]")]
                    body = line[line.index("]") + 1 :]
                    label, score = self._label(body)
                    theme = "performance" if "slow" in body else "sentiment"
                    extracts.append(
                        CommentExtract(
                            comment_id=cid,
                            sentiment=CommentSentiment(label=label, score=score),
                            candidate_theme_labels=[theme],
                            quote_worthy=True,
                        )
                    )
            outputs.append(MapBatchOutput(extracts=extracts))
        return outputs


def _cache():
    comments = [
        comment("c1", body="I love notion", score=20),
        comment("c2", body="notion is slow", score=15),
        comment("c3", body="notion is fine", score=10),
        comment("c4", body="notion hate it", score=8),
        comment("lowscore", body="notion love", score=1),  # filtered by score
        comment("offtopic", body="totally unrelated", score=99),  # filtered by relevance
    ]
    return make_cache([post("p1", comments)], topic="Notion")


def test_analysis_result_validates_and_populates_overall() -> None:
    result = run_analysis(
        _cache(), map_model=FakeMapModel(), min_score=5, model_id="gpt-5.4-mini", generated_at=GEN
    )
    assert isinstance(result, AnalysisResult)
    AnalysisResult.model_validate(result.model_dump())
    assert result.themes == []
    assert result.feature_requests == []
    assert result.competitor_mentions == []


def test_overall_sentiment_computed_from_filtered_comments() -> None:
    result = run_analysis(
        _cache(), map_model=FakeMapModel(), min_score=5, model_id="gpt-5.4-mini", generated_at=GEN
    )
    dist = result.overall.sentiment.distribution
    # kept: c1(+) c2(-) c3(0) c4(-); dropped lowscore + offtopic
    assert dist.positive == 1
    assert dist.negative == 2
    assert dist.neutral == 1
    assert result.overall.sentiment.label == "mixed"


def test_run_metadata_reflects_model_params_and_corpus() -> None:
    result = run_analysis(
        _cache(), map_model=FakeMapModel(), min_score=7, model_id="gpt-5.6-terra", generated_at=GEN
    )
    m = result.run_metadata
    assert m.models.map == "gpt-5.6-terra"
    assert m.models.reduce == "gpt-5.6-terra"
    assert m.params.comment_min_score == 7
    assert m.topic == "Notion"
    assert m.generated_at == "2026-07-22T14:30:00Z"
    assert m.corpus.comment_count == 6  # full corpus counted, not the filtered subset


def test_default_model_is_gpt_54_mini() -> None:
    from reddit_research import defaults

    assert defaults.DEFAULT_MODEL == "gpt-5.4-mini"
