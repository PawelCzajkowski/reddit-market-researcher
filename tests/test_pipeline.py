"""Analyze foundation: cache -> filter -> map (fake LLM) -> overall sentiment -> JSON."""

from __future__ import annotations

from datetime import datetime, timezone

from reddit_research.analyze.extract import (
    CommentExtract,
    CommentSentiment,
    CompetitorMentionExtract,
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
                    is_fr = "wish" in body or "need" in body
                    competitors = (
                        [CompetitorMentionExtract(name="Obsidian", relationship="switching_to")]
                        if "obsidian" in body
                        else []
                    )
                    extracts.append(
                        CommentExtract(
                            comment_id=cid,
                            sentiment=CommentSentiment(label=label, score=score),
                            candidate_theme_labels=[theme],
                            quote_worthy=True,
                            is_feature_request=is_fr,
                            feature_request_text="offline mode" if is_fr else None,
                            feature_request_rationale="works on flights" if is_fr else None,
                            competitor_mentions=competitors,
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


def test_full_pipeline_populates_themes_features_competitors() -> None:
    from reddit_research.analyze.canonicalize import (
        CompetitorAssignment,
        LabelAssignment,
        TaxonomyTheme,
        ThemeTaxonomy,
    )

    comments = [
        comment("c1", body="notion is slow", score=20),
        comment("c2", body="notion is slow too", score=18),
        comment("c3", body="wish notion had offline mode", score=15),
        comment("c4", body="switching to obsidian from notion", score=12),
    ]
    cache = make_cache([post("p1", comments)], topic="Notion")

    def canonicalizer(counts, competitors, topic) -> ThemeTaxonomy:
        return ThemeTaxonomy(
            themes=[
                TaxonomyTheme(
                    id="performance",
                    label="Performance",
                    description="Speed complaints.",
                    is_pain_point=True,
                ),
                TaxonomyTheme(
                    id="sentiment",
                    label="General sentiment",
                    description="Everything else.",
                    is_pain_point=False,
                ),
            ],
            label_map=[
                LabelAssignment(raw_label="performance", theme_id="performance"),
                LabelAssignment(raw_label="sentiment", theme_id="sentiment"),
            ],
            competitor_map=[
                CompetitorAssignment(raw_name="Obsidian", canonical_name="Obsidian")
            ],
        )

    result = run_analysis(
        cache,
        map_model=FakeMapModel(),
        min_score=5,
        model_id="gpt-5.4-mini",
        canonicalizer=canonicalizer,
        summarizer=lambda ctx: f"Summary of {ctx.analyzed_comment_count} comments.",
        generated_at=GEN,
    )

    assert result.themes  # at least one theme populated
    assert any(fr.request.lower() == "offline mode" for fr in result.feature_requests)
    assert any(cm.name == "Obsidian" for cm in result.competitor_mentions)
    assert result.overall.executive_summary == "Summary of 4 comments."
