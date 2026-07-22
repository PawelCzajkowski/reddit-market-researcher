"""D1 canonical output schema — the single JSON artifact one analysis run produces.

`run_metadata` is populated by our code; `overall`, `themes`, `feature_requests`,
`competitor_mentions` form the LLM-facing `AnalysisPayload`. The Markdown report is
a pure rendering of `AnalysisResult`. See docs/design/output-schema.md.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

SentimentLabel = Literal["positive", "negative", "neutral", "mixed"]
QuoteType = Literal["post", "comment"]
CompetitorRelationship = Literal[
    "alternative", "comparison", "switching_to", "switching_from", "complementary"
]


class SentimentDistribution(BaseModel):
    positive: int
    negative: int
    neutral: int


class SentimentBlock(BaseModel):
    label: SentimentLabel
    score: float = Field(ge=-1.0, le=1.0)
    distribution: SentimentDistribution


class Quote(BaseModel):
    text: str
    author: Optional[str]  # e.g. "u/example_user"; null if unavailable/deleted
    type: QuoteType
    permalink: str  # full https URL to the post or comment
    subreddit: str  # "r/Notion"
    post_title: str
    score: int
    created_at: str  # ISO-8601 UTC


class Prevalence(BaseModel):
    mention_count: int
    comment_percentage: float  # share of total corpus comments (0-100)


class Theme(BaseModel):
    id: str  # stable slug, e.g. "performance-large-workspaces"
    label: str
    description: str
    is_pain_point: bool
    sentiment: SentimentBlock
    prevalence: Prevalence
    representative_quotes: list[Quote]


class FeatureRequest(BaseModel):
    id: str
    request: str
    rationale: str  # why users want it
    prevalence: Prevalence
    representative_quotes: list[Quote]


class CompetitorMention(BaseModel):
    name: str
    relationship: Optional[CompetitorRelationship]
    sentiment: SentimentBlock
    mention_count: int
    representative_quotes: list[Quote]


class Overall(BaseModel):
    executive_summary: str
    sentiment: SentimentBlock


class AnalysisPayload(BaseModel):
    overall: Overall
    themes: list[Theme]
    feature_requests: list[FeatureRequest]
    competitor_mentions: list[CompetitorMention]


# --- code-populated metadata ---
class RunParams(BaseModel):
    posts_per_subreddit: int
    time_window_days: int
    comment_min_score: int
    sort: str


class SubredditCorpus(BaseModel):
    subreddit: str
    post_count: int
    comment_count: int


class Corpus(BaseModel):
    post_count: int
    comment_count: int
    by_subreddit: list[SubredditCorpus]


class Models(BaseModel):
    map: str
    reduce: str


class RunMetadata(BaseModel):
    topic: str
    subreddits: list[str]
    params: RunParams
    generated_at: str  # ISO-8601 UTC
    corpus: Corpus
    models: Models


class AnalysisResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_metadata: RunMetadata
    overall: Overall
    themes: list[Theme]
    feature_requests: list[FeatureRequest]
    competitor_mentions: list[CompetitorMention]
