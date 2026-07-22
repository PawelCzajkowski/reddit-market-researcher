# Canonical Output Schema (D1, #5) — LOCKED

The single JSON artifact one analysis run produces. The Markdown report is a pure rendering of this file, so it must carry everything the report needs.

## Locked decisions

1. **Sentiment** is `label` + `score` (−1…+1) + `distribution` (raw pos/neg/neutral counts), everywhere it appears. Distribution keeps it honest for market research.
2. **Feature requests** and **competitor mentions** are **top-level** arrays, not nested inside themes — they're acted on differently.
3. **Themes are general clusters** carrying `is_pain_point`; pain points are the subset where `is_pain_point == true`. One pipeline surfaces both complaints and praise.
4. **Prevalence** = `mention_count` + `comment_percentage`, where the denominator is **total comments in the corpus** (percentage-of-comments).
5. **Quotes keep `author`** (nullable) alongside the full `permalink` — the author lets a human find the original post/comment on Reddit manually. Cache is treated as ephemeral (R1 ToS: honor deletions).

**Code vs. LLM split:** `run_metadata` is populated by our code. `overall`, `themes`, `feature_requests`, `competitor_mentions` are the LLM-generated `AnalysisPayload` — the object handed to `with_structured_output(AnalysisPayload, method="json_schema", strict=True)` (per R2). Our code merges the payload with metadata into the final `AnalysisResult`.

**Strict-mode note:** OpenAI strict output cannot enforce array *caps*. "≤3 quotes per theme, ~10 themes max, themes ordered by prevalence desc" are **prompt-enforced conventions** (a D2 concern), not schema constraints. All fields are required; optionality is expressed as nullable.

## Pydantic models

```python
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
    author: Optional[str]           # e.g. "u/example_user"; null if unavailable/deleted
    type: QuoteType
    permalink: str                  # full https://reddit.com/... link to the post or comment
    subreddit: str                  # "r/Notion"
    post_title: str
    score: int
    created_at: str                 # ISO-8601 UTC

class Prevalence(BaseModel):
    mention_count: int
    comment_percentage: float       # share of total corpus comments (0–100)

class Theme(BaseModel):
    id: str                         # stable slug, e.g. "performance-large-workspaces"
    label: str
    description: str
    is_pain_point: bool
    sentiment: SentimentBlock
    prevalence: Prevalence
    representative_quotes: list[Quote]

class FeatureRequest(BaseModel):
    id: str
    request: str
    rationale: str                  # why users want it
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

# --- LLM-facing strict payload (the reduce-step structured output) ---
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
    generated_at: str               # ISO-8601 UTC
    corpus: Corpus
    models: Models

# --- the canonical artifact written to disk ---
class AnalysisResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_metadata: RunMetadata
    overall: Overall
    themes: list[Theme]
    feature_requests: list[FeatureRequest]
    competitor_mentions: list[CompetitorMention]
```

## Filled example

```jsonc
{
  "schema_version": "1.0",
  "run_metadata": {
    "topic": "Notion",
    "subreddits": ["r/productivity", "r/Notion"],
    "params": { "posts_per_subreddit": 25, "time_window_days": 90, "comment_min_score": 5, "sort": "top" },
    "generated_at": "2026-07-22T14:30:00Z",
    "corpus": {
      "post_count": 48, "comment_count": 1163,
      "by_subreddit": [
        { "subreddit": "r/productivity", "post_count": 25, "comment_count": 702 },
        { "subreddit": "r/Notion",       "post_count": 23, "comment_count": 461 }
      ]
    },
    "models": { "map": "gpt-5.4-mini", "reduce": "gpt-5.6-sol" }
  },
  "overall": {
    "executive_summary": "Users value Notion's flexibility but repeatedly cite performance on large workspaces and a steep onboarding curve as the top frustrations. Offline support is the single most-requested capability.",
    "sentiment": { "label": "mixed", "score": -0.12, "distribution": { "positive": 421, "negative": 508, "neutral": 234 } }
  },
  "themes": [
    {
      "id": "performance-large-workspaces",
      "label": "Sluggish on large workspaces",
      "description": "Load times and lag degrade sharply as pages/databases grow.",
      "is_pain_point": true,
      "sentiment": { "label": "negative", "score": -0.71, "distribution": { "positive": 12, "negative": 187, "neutral": 24 } },
      "prevalence": { "mention_count": 223, "comment_percentage": 19.2 },
      "representative_quotes": [
        {
          "text": "Once my workspace hit ~2000 pages it takes 6-7 seconds to open anything.",
          "author": "u/example_user", "type": "comment",
          "permalink": "https://reddit.com/r/Notion/comments/abc123/xyz/def456/",
          "subreddit": "r/Notion", "post_title": "Anyone else's Notion getting slow?",
          "score": 342, "created_at": "2026-06-14T09:12:00Z"
        }
      ]
    }
  ],
  "feature_requests": [
    {
      "id": "offline-mode",
      "request": "True offline editing with later sync",
      "rationale": "Users on commutes/flights lose access to their notes entirely.",
      "prevalence": { "mention_count": 141, "comment_percentage": 12.1 },
      "representative_quotes": []
    }
  ],
  "competitor_mentions": [
    {
      "name": "Obsidian",
      "relationship": "switching_to",
      "sentiment": { "label": "positive", "score": 0.44, "distribution": { "positive": 78, "negative": 19, "neutral": 22 } },
      "mention_count": 119,
      "representative_quotes": []
    }
  ]
}
```

## Handoffs

- **D2 (pipeline):** enforces the array-cap/ordering conventions via prompt; produces `AnalysisPayload` from the map-reduce; computes `prevalence` denominators and `distribution` counts.
- **D3 (fetch/cache):** the raw-data schema must supply every `Quote` field (permalink, author, subreddit, post_title, score, created_at, type) so the analyze stage can populate quotes without re-fetching.
