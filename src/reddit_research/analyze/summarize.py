"""Step 5 — summarize (LLM, 1 call).

Input is the *code-computed* aggregate numbers (overall sentiment + counts + the
top themes/feature-requests/competitors already built in code). Output is prose
grounded in those numbers — the model writes narrative, it does not invent figures.
The structured model is injected, so the pipeline runs in tests with a fake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from ..models.output import (
    CompetitorMention,
    FeatureRequest,
    SentimentBlock,
    Theme,
)


@dataclass
class SummaryContext:
    topic: str
    subreddits: list[str]
    overall: SentimentBlock
    analyzed_comment_count: int
    themes: list[Theme]
    feature_requests: list[FeatureRequest]
    competitor_mentions: list[CompetitorMention]


class SummaryOutput(BaseModel):
    executive_summary: str


class SummaryModel(Protocol):
    def invoke(self, input: Any) -> SummaryOutput: ...


_SYSTEM_PROMPT = (
    "You are writing the executive summary of a Reddit market-research report. "
    "You are given already-computed statistics. Write 2-4 sentences of grounded "
    "prose that reflect these numbers. Do NOT invent counts, percentages, themes, "
    "or quotes beyond what is provided."
)


def _sentiment_line(sb: SentimentBlock) -> str:
    d = sb.distribution
    return (
        f"overall sentiment {sb.label} (score {sb.score:+.2f}; "
        f"{d.positive} positive / {d.negative} negative / {d.neutral} neutral)"
    )


def build_summary_input(ctx: SummaryContext) -> list[dict[str, str]]:
    lines = [
        f"Topic: {ctx.topic}",
        f"Subreddits: {', '.join(ctx.subreddits)}",
        f"Analyzed comments: {ctx.analyzed_comment_count}",
        f"Overall: {_sentiment_line(ctx.overall)}",
    ]
    if ctx.themes:
        lines.append("Top themes (by prevalence):")
        for t in ctx.themes:
            flag = " [pain point]" if t.is_pain_point else ""
            lines.append(
                f"  - {t.label}{flag}: {t.prevalence.mention_count} mentions "
                f"({t.prevalence.comment_percentage:.1f}%), sentiment {t.sentiment.label}"
            )
    if ctx.feature_requests:
        lines.append("Top feature requests:")
        for fr in ctx.feature_requests:
            lines.append(f"  - {fr.request} ({fr.prevalence.mention_count} mentions)")
    if ctx.competitor_mentions:
        lines.append("Competitors mentioned:")
        for cm in ctx.competitor_mentions:
            lines.append(f"  - {cm.name} ({cm.mention_count} mentions, {cm.sentiment.label})")
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def run_summary(model: SummaryModel, ctx: SummaryContext) -> str:
    return model.invoke(build_summary_input(ctx)).executive_summary
