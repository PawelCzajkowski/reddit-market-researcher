"""Deterministic sentiment aggregation (code, 0 tokens).

Per-comment `label`+`score` (from the map step) roll up into a `SentimentBlock`:
`distribution` is an exact tally, `score` is the mean, and `label` derives by rule.
The label is `mixed` when no single bucket reaches the dominance threshold — which
is exactly "positive/negative both substantial" (D2 decision 3).
"""

from __future__ import annotations

from collections.abc import Iterable

from ..models.output import SentimentBlock, SentimentDistribution, SentimentLabel

PerComment = tuple[str, float]  # (label in {positive,negative,neutral}, score)

_DOMINANCE = 0.6  # a bucket at or above this share of all comments sets the label
_SCORE_DP = 2


def aggregate_sentiment(items: Iterable[PerComment]) -> SentimentBlock:
    items = list(items)
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    score_sum = 0.0
    for lbl, score in items:
        counts[lbl] += 1
        score_sum += score

    total = len(items)
    distribution = SentimentDistribution(**counts)
    if total == 0:
        return SentimentBlock(label="neutral", score=0.0, distribution=distribution)

    mean = round(score_sum / total, _SCORE_DP)
    shares = {k: v / total for k, v in counts.items()}
    label: SentimentLabel
    if shares["neutral"] >= _DOMINANCE:
        label = "neutral"
    elif shares["positive"] >= _DOMINANCE:
        label = "positive"
    elif shares["negative"] >= _DOMINANCE:
        label = "negative"
    else:
        label = "mixed"

    return SentimentBlock(label=label, score=mean, distribution=distribution)
