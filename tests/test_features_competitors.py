"""Feature-request grouping and competitor-mention aggregation (code)."""

from __future__ import annotations

from reddit_research.analyze.aggregate import (
    aggregate_competitors,
    aggregate_feature_requests,
    pair,
)
from reddit_research.analyze.canonicalize import CompetitorAssignment, ThemeTaxonomy
from reddit_research.analyze.extract import (
    CommentExtract,
    CommentSentiment,
    CompetitorMentionExtract,
)

from .cache_fixtures import comment


def _extract(
    cid: str,
    *,
    fr_text=None,
    fr_rationale=None,
    competitors=None,
    label="neutral",
    score=0.0,
) -> CommentExtract:
    return CommentExtract(
        comment_id=cid,
        sentiment=CommentSentiment(label=label, score=score),
        candidate_theme_labels=[],
        quote_worthy=True,
        is_feature_request=fr_text is not None,
        feature_request_text=fr_text,
        feature_request_rationale=fr_rationale,
        competitor_mentions=competitors or [],
    )


def _analyzed(extracts: list[CommentExtract]):
    comments = [comment(e.comment_id, score=10) for e in extracts]
    return pair(comments, extracts)


def test_feature_requests_grouped_by_canonical_text() -> None:
    extracts = [
        _extract("a", fr_text="Offline mode", fr_rationale="works on flights"),
        _extract("b", fr_text="offline mode!"),  # same, differs by case/punctuation
        _extract("c", fr_text="Dark theme"),
        _extract("d"),  # not a feature request
    ]
    frs = aggregate_feature_requests(_analyzed(extracts), total_analyzed=4)
    by_request = {fr.request.lower().rstrip("!"): fr for fr in frs}
    assert "offline mode" in by_request
    offline = by_request["offline mode"]
    assert offline.prevalence.mention_count == 2
    assert offline.prevalence.comment_percentage == 50.0
    assert offline.rationale == "works on flights"  # picked up from a member
    assert len(offline.representative_quotes) <= 3
    # ordered by prevalence desc
    assert frs[0].prevalence.mention_count == 2


def test_competitors_grouped_via_map_with_mode_relationship() -> None:
    taxonomy = ThemeTaxonomy(
        themes=[],
        label_map=[],
        competitor_map=[
            CompetitorAssignment(raw_name="obsidian.md", canonical_name="Obsidian"),
            CompetitorAssignment(raw_name="Obsidian", canonical_name="Obsidian"),
        ],
    )
    extracts = [
        _extract("a", competitors=[CompetitorMentionExtract(name="obsidian.md", relationship="switching_to")], label="positive", score=0.6),
        _extract("b", competitors=[CompetitorMentionExtract(name="Obsidian", relationship="switching_to")], label="positive", score=0.4),
        _extract("c", competitors=[CompetitorMentionExtract(name="Obsidian", relationship="comparison")], label="negative", score=-0.2),
    ]
    mentions = aggregate_competitors(_analyzed(extracts), taxonomy)
    assert len(mentions) == 1
    obsidian = mentions[0]
    assert obsidian.name == "Obsidian"
    assert obsidian.mention_count == 3
    assert obsidian.relationship == "switching_to"  # mode of the three
    assert obsidian.sentiment.distribution.positive == 2
    assert obsidian.sentiment.distribution.negative == 1


def test_unmapped_competitor_falls_back_to_raw_name() -> None:
    taxonomy = ThemeTaxonomy(themes=[], label_map=[], competitor_map=[])
    extracts = [
        _extract("a", competitors=[CompetitorMentionExtract(name="Roam", relationship=None)]),
    ]
    mentions = aggregate_competitors(_analyzed(extracts), taxonomy)
    assert mentions[0].name == "Roam"
    assert mentions[0].relationship is None
