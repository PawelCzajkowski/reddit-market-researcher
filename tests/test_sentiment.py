"""Deterministic sentiment aggregation: distribution, mean score, label rule."""

from __future__ import annotations

import pytest

from reddit_research.analyze.sentiment import aggregate_sentiment


def test_empty_is_neutral_zero() -> None:
    block = aggregate_sentiment([])
    assert block.label == "neutral"
    assert block.score == 0.0
    assert block.distribution.model_dump() == {"positive": 0, "negative": 0, "neutral": 0}


def test_distribution_counts_are_exact() -> None:
    items = (
        [("positive", 0.8)] * 3 + [("negative", -0.5)] * 2 + [("neutral", 0.0)] * 1
    )
    block = aggregate_sentiment(items)
    assert block.distribution.model_dump() == {"positive": 3, "negative": 2, "neutral": 1}


def test_positive_dominates_at_60_percent() -> None:
    block = aggregate_sentiment([("positive", 0.5)] * 6 + [("negative", -0.5)] * 4)
    assert block.label == "positive"


def test_negative_dominates() -> None:
    block = aggregate_sentiment([("negative", -0.5)] * 7 + [("positive", 0.5)] * 3)
    assert block.label == "negative"


def test_mixed_when_neither_dominates() -> None:
    block = aggregate_sentiment([("positive", 0.5)] * 5 + [("negative", -0.5)] * 5)
    assert block.label == "mixed"


def test_lopsided_positive_with_no_negative_is_not_mixed() -> None:
    # 55% positive / 45% neutral / 0% negative: all directional signal is positive,
    # so the label must be "positive", not "mixed" (D2 "both substantial" rule).
    block = aggregate_sentiment([("positive", 0.5)] * 55 + [("neutral", 0.0)] * 45)
    assert block.label == "positive"


def test_neutral_dominates() -> None:
    block = aggregate_sentiment(
        [("neutral", 0.0)] * 7 + [("positive", 0.5)] * 2 + [("negative", -0.5)] * 1
    )
    assert block.label == "neutral"


def test_reproduces_locked_d1_example_label() -> None:
    # D1 example: pos 421 / neg 508 / neutral 234 -> "mixed"
    items = (
        [("positive", 0.3)] * 421 + [("negative", -0.3)] * 508 + [("neutral", 0.0)] * 234
    )
    block = aggregate_sentiment(items)
    assert block.label == "mixed"
    assert block.distribution.model_dump() == {
        "positive": 421,
        "negative": 508,
        "neutral": 234,
    }


def test_mean_score_rounded() -> None:
    block = aggregate_sentiment([("positive", 1.0), ("negative", -1.0), ("neutral", 0.0)])
    assert block.score == pytest.approx(0.0)
    block2 = aggregate_sentiment([("positive", 0.5), ("positive", 0.7)])
    assert block2.score == pytest.approx(0.6)
