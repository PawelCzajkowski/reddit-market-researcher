"""Themes flow through run_analysis and render in the Markdown report."""

from __future__ import annotations

from datetime import datetime, timezone

from reddit_research.analyze.canonicalize import (
    LabelAssignment,
    TaxonomyTheme,
    ThemeTaxonomy,
)
from reddit_research.analyze.pipeline import run_analysis
from reddit_research.report import render_report

from .cache_fixtures import comment, make_cache, post
from .test_pipeline import FakeMapModel

GEN = datetime(2026, 7, 22, 14, 30, 0, tzinfo=timezone.utc)


def _canonicalizer(counts, competitors, topic):  # noqa: ANN001
    # deterministic: the fake map emits "performance" (for slow) and "sentiment"
    return ThemeTaxonomy(
        themes=[
            TaxonomyTheme(id="perf", label="Performance", description="Speed.", is_pain_point=True),
            TaxonomyTheme(id="sent", label="General", description="Vibes.", is_pain_point=False),
        ],
        label_map=[
            LabelAssignment(raw_label="performance", theme_id="perf"),
            LabelAssignment(raw_label="sentiment", theme_id="sent"),
        ],
        competitor_map=[],
    )


def _cache():
    comments = [
        comment("c1", body="notion is slow", score=90),
        comment("c2", body="notion is slow too", score=40),
        comment("c3", body="I love notion", score=10),
    ]
    return make_cache([post("p1", comments)], topic="Notion")


def test_themes_populated_and_rendered() -> None:
    result = run_analysis(
        _cache(),
        map_model=FakeMapModel(),
        min_score=5,
        model_id="gpt-5.4-mini",
        canonicalizer=_canonicalizer,
        generated_at=GEN,
    )
    ids = {t.id for t in result.themes}
    assert ids == {"perf", "sent"}
    perf = next(t for t in result.themes if t.id == "perf")
    assert perf.is_pain_point is True
    assert perf.prevalence.mention_count == 2
    assert perf.representative_quotes[0].score == 90  # top by score

    md = render_report(result)
    assert "## Themes" in md
    assert "Performance" in md
    assert "pain point" in md
    assert "notion is slow" in md


def test_no_canonicalizer_keeps_themes_empty() -> None:
    result = run_analysis(
        _cache(), map_model=FakeMapModel(), min_score=5, model_id="gpt-5.4-mini", generated_at=GEN
    )
    assert result.themes == []
