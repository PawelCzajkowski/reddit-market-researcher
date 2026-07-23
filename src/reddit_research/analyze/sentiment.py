"""Deterministic sentiment aggregation (code, 0 tokens).

Per-comment `label`+`score` (from the map step) roll up into a `SentimentBlock`:
`distribution` is an exact tally, `score` is the mean, and `label` derives by rule.

Label rule (D2 decision 3 — "'mixed' when positive/negative are both substantial,
neither ≳60%"):
- neutral dominates the whole corpus (≥60%)                       -> neutral
- otherwise, decide on the positive-vs-negative contest: whichever
  side holds ≳60% of the directional (pos+neg) signal wins; when
  neither does, both are substantial                              -> mixed
This keeps a lopsided case honest — 55% positive / 45% neutral / 0% negative reads
as positive (all directional signal is positive), not mixed.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..models.output import SentimentBlock, SentimentDistribution, SentimentLabel

PerComment = tuple[str, float]  # (label in {positive,negative,neutral}, score)

_DOMINANCE = 0.6  # share needed to declare a single dominant sentiment
_SCORE_DP = 2


def _label(counts: dict[str, int], total: int) -> SentimentLabel:
    if counts["neutral"] / total >= _DOMINANCE:
        return "neutral"
    directional = counts["positive"] + counts["negative"]
    if directional == 0:
        return "neutral"
    positive_share = counts["positive"] / directional
    if positive_share >= _DOMINANCE:
        return "positive"
    if positive_share <= 1 - _DOMINANCE:
        return "negative"
    return "mixed"


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
    return SentimentBlock(label=_label(counts, total), score=mean, distribution=distribution)
