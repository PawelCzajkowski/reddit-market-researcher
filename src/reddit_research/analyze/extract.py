"""Step 2 — map/extract (LLM, batched).

Each surviving comment yields a `CommentExtract`. Comments are grouped into
token-budgeted chunks; `model.batch()` runs the chunks concurrently. The
structured model (a `with_structured_output(..., method="json_schema",
strict=True)` runnable) is injected, so the pipeline is exercised in tests with a
fake and never calls OpenAI.

For #11 the extract carries only per-comment sentiment; theme/feature/competitor
fields are layered on in later tickets.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel

from ..models.cache import CachedComment

CommentSentimentLabel = Literal["positive", "negative", "neutral"]

_DEFAULT_MAX_CHARS = 12_000  # ~3k tokens/chunk; keeps each map call well-bounded


class CommentSentiment(BaseModel):
    label: CommentSentimentLabel
    score: float


class CommentExtract(BaseModel):
    comment_id: str
    sentiment: CommentSentiment


class MapBatchOutput(BaseModel):
    """One structured output per chunk: an extract for each comment in the chunk."""

    extracts: list[CommentExtract]


class StructuredModel(Protocol):
    def batch(self, inputs: list[Any]) -> list[MapBatchOutput]: ...


_SYSTEM_PROMPT = (
    "You extract structured signal from Reddit comments for market research about "
    "a given topic. For EACH comment in the batch, return one extract preserving its "
    "comment_id. Classify sentiment toward the topic as 'positive', 'negative', or "
    "'neutral', with a score from -1.0 (very negative) to 1.0 (very positive). "
    "Do not invent comment_ids; return exactly one extract per input comment."
)


def chunk_comments(
    comments: list[CachedComment], *, max_chars: int = _DEFAULT_MAX_CHARS
) -> list[list[CachedComment]]:
    """Group comments into chunks whose combined body size stays under max_chars."""
    chunks: list[list[CachedComment]] = []
    current: list[CachedComment] = []
    size = 0
    for c in comments:
        clen = len(c.body) + len(c.id) + 32  # rough per-comment overhead
        if current and size + clen > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(c)
        size += clen
    if current:
        chunks.append(current)
    return chunks


def build_map_input(chunk: list[CachedComment], *, topic: str) -> list[dict[str, str]]:
    """Chat messages (system + human) for one chunk."""
    lines = [f"Topic: {topic}", "", "Comments:"]
    for c in chunk:
        body = c.body.replace("\n", " ").strip()
        lines.append(f"- [{c.id}] {body}")
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def run_map(
    model: StructuredModel,
    comments: list[CachedComment],
    *,
    topic: str,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> list[CommentExtract]:
    """Extract one CommentExtract per input comment (aligned to input order).

    Missing/spurious extracts from the model are reconciled by comment_id; any
    comment the model failed to score defaults to neutral/0.0 so aggregation is
    robust to omissions.
    """
    if not comments:
        return []
    chunks = chunk_comments(comments, max_chars=max_chars)
    inputs = [build_map_input(chunk, topic=topic) for chunk in chunks]
    outputs = model.batch(inputs)

    by_id: dict[str, CommentExtract] = {}
    for out in outputs:
        for extract in out.extracts:
            by_id.setdefault(extract.comment_id, extract)

    return [
        by_id.get(
            c.id,
            CommentExtract(
                comment_id=c.id, sentiment=CommentSentiment(label="neutral", score=0.0)
            ),
        )
        for c in comments
    ]
