"""Cheap pre-flight cost estimate (SPEC §8, D2 Cost).

`count_tokens_approximately` over the map inputs × a local price constant, before
any paid LLM call. This is a conservative *estimate*, not billing: LangSmith
tracing (when enabled) is the authoritative per-run cost. The estimate never hard
-aborts — it only drives the soft-ceiling warning + confirmation.

Prices are $/1M tokens from docs/research/openai-models.md (fetched 2026-07-22);
re-verify before relying on the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages.utils import count_tokens_approximately

from ..models.cache import CachedComment
from .extract import build_map_input, chunk_comments

# ($ per 1M input tokens, $ per 1M output tokens)
PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-5.6-terra": (2.50, 15.00),
    "gpt-5.6-luna": (1.00, 6.00),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
}
# Conservative fallback for an unrecognised model id (frontier-ish rate).
_FALLBACK_PRICE = (5.00, 30.00)

# Output-token heuristics: map emits a small extract per comment; the reduce
# steps (canonicalize + summarize) emit a bounded report.
_MAP_OUTPUT_TOKENS_PER_COMMENT = 30
_REDUCE_OUTPUT_TOKENS = 3_000


def price_for(model_id: str) -> tuple[tuple[float, float], bool]:
    """Return (price, known). Unknown ids fall back to a conservative rate."""
    if model_id in PRICES:
        return PRICES[model_id], True
    return _FALLBACK_PRICE, False


@dataclass(frozen=True)
class CostEstimate:
    model_id: str
    input_tokens: int
    output_tokens: int
    projected_usd: float
    price_known: bool


def estimate_cost(
    comments: list[CachedComment], *, topic: str, model_id: str
) -> CostEstimate:
    """Project the run's cost from the filtered corpus (all steps on one model)."""
    chunks = chunk_comments(comments)
    input_tokens = sum(
        count_tokens_approximately(build_map_input(chunk, topic=topic))
        for chunk in chunks
    )
    output_tokens = len(comments) * _MAP_OUTPUT_TOKENS_PER_COMMENT + (
        _REDUCE_OUTPUT_TOKENS if comments else 0
    )
    (price_in, price_out), known = price_for(model_id)
    projected = input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out
    return CostEstimate(
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        projected_usd=round(projected, 4),
        price_known=known,
    )
