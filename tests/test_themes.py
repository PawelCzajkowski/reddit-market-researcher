"""Theme canonicalization inputs + code aggregation (prevalence, quotes, ordering)."""

from __future__ import annotations

from reddit_research.analyze.aggregate import aggregate_themes, comment_to_quote, pair
from reddit_research.analyze.canonicalize import (
    LabelAssignment,
    TaxonomyTheme,
    ThemeTaxonomy,
    label_counts,
)
from reddit_research.analyze.extract import CommentExtract, CommentSentiment

from .cache_fixtures import comment


def _extract(cid: str, labels: list[str], *, label="negative", score=-0.5, worthy=True):
    return CommentExtract(
        comment_id=cid,
        sentiment=CommentSentiment(label=label, score=score),
        candidate_theme_labels=labels,
        quote_worthy=worthy,
        is_feature_request=False,
        feature_request_text=None,
        feature_request_rationale=None,
        competitor_mentions=[],
    )


def _taxonomy() -> ThemeTaxonomy:
    return ThemeTaxonomy(
        themes=[
            TaxonomyTheme(id="perf", label="Performance", description="Speed issues.", is_pain_point=True),
            TaxonomyTheme(id="ui", label="UI", description="Interface praise.", is_pain_point=False),
            TaxonomyTheme(id="empty", label="Unused", description="No comments.", is_pain_point=False),
        ],
        label_map=[
            LabelAssignment(raw_label="slow", theme_id="perf"),
            LabelAssignment(raw_label="laggy", theme_id="perf"),
            LabelAssignment(raw_label="pretty", theme_id="ui"),
        ],
        competitor_map=[],
    )


def test_label_counts_dedupes_case_insensitively_and_orders() -> None:
    counts = label_counts(["Slow", "slow", "laggy", "SLOW", "  "])
    assert counts[0] == ("Slow", 3)
    assert ("laggy", 1) in counts


def test_aggregate_assigns_prevalence_and_percentage() -> None:
    comments = [comment(f"c{i}", score=i * 10) for i in range(4)]
    extracts = [
        _extract("c0", ["slow"]),
        _extract("c1", ["laggy"]),
        _extract("c2", ["slow", "pretty"], label="positive", score=0.6),
        _extract("c3", ["pretty"], label="positive", score=0.6),
    ]
    themes = aggregate_themes(pair(comments, extracts), _taxonomy(), total_analyzed=4)
    by_id = {t.id: t for t in themes}
    assert by_id["perf"].prevalence.mention_count == 3  # c0, c1, c2
    assert by_id["perf"].prevalence.comment_percentage == 75.0
    assert by_id["ui"].prevalence.mention_count == 2  # c2, c3
    assert "empty" not in by_id  # no comments -> dropped


def test_themes_ordered_by_prevalence_desc() -> None:
    comments = [comment(f"c{i}") for i in range(3)]
    extracts = [_extract("c0", ["slow"]), _extract("c1", ["slow"]), _extract("c2", ["pretty"])]
    themes = aggregate_themes(pair(comments, extracts), _taxonomy(), total_analyzed=3)
    assert [t.id for t in themes] == ["perf", "ui"]


def test_quotes_capped_at_three_and_sorted_by_score() -> None:
    comments = [comment(f"c{i}", score=s) for i, s in enumerate([5, 100, 50, 1, 75])]
    extracts = [_extract(f"c{i}", ["slow"]) for i in range(5)]
    themes = aggregate_themes(pair(comments, extracts), _taxonomy(), total_analyzed=5)
    perf = next(t for t in themes if t.id == "perf")
    assert len(perf.representative_quotes) == 3
    assert [q.score for q in perf.representative_quotes] == [100, 75, 50]


def test_only_quote_worthy_comments_become_quotes() -> None:
    comments = [comment("c0", score=100), comment("c1", score=50)]
    extracts = [_extract("c0", ["slow"], worthy=False), _extract("c1", ["slow"], worthy=True)]
    themes = aggregate_themes(pair(comments, extracts), _taxonomy(), total_analyzed=2)
    perf = next(t for t in themes if t.id == "perf")
    assert [q.score for q in perf.representative_quotes] == [50]  # c0 excluded (not worthy)


def test_theme_sentiment_is_code_aggregated() -> None:
    comments = [comment(f"c{i}") for i in range(3)]
    extracts = [
        _extract("c0", ["slow"], label="negative", score=-0.8),
        _extract("c1", ["slow"], label="negative", score=-0.6),
        _extract("c2", ["slow"], label="negative", score=-0.7),
    ]
    themes = aggregate_themes(pair(comments, extracts), _taxonomy(), total_analyzed=3)
    perf = next(t for t in themes if t.id == "perf")
    assert perf.sentiment.label == "negative"
    assert perf.sentiment.distribution.negative == 3


def test_comment_to_quote_maps_all_d1_fields() -> None:
    c = comment("c1", body="notion is slow", score=42)
    q = comment_to_quote(c)
    assert q.text == "notion is slow"
    assert q.type == "comment"
    assert q.permalink.startswith("https://www.reddit.com/")
    assert q.subreddit == "r/Notion"
    assert q.score == 42
    assert q.post_title == "A post"
