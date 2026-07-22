"""Step 3 — canonicalize (LLM, 1 call).

The map step emits noisy free-text `candidate_theme_labels`. This step dedupes
them (with occurrence counts) and asks the model once for a coherent ~10-theme
taxonomy plus a `raw_label -> theme_id` map; code then assigns comments to themes
via that map (step 4). Maps are modelled as lists of pairs, not dicts, so the
output stays valid under OpenAI strict `json_schema` (which forbids open-ended
object keys). The structured model is injected for testing.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Protocol

from pydantic import BaseModel

from ..defaults import MAX_THEMES


class TaxonomyTheme(BaseModel):
    id: str  # stable slug, e.g. "performance-large-workspaces"
    label: str
    description: str
    is_pain_point: bool


class LabelAssignment(BaseModel):
    raw_label: str
    theme_id: str


class CompetitorAssignment(BaseModel):
    raw_name: str
    canonical_name: str


class ThemeTaxonomy(BaseModel):
    themes: list[TaxonomyTheme]
    label_map: list[LabelAssignment]  # raw candidate label -> theme id
    competitor_map: list[CompetitorAssignment]  # raw competitor name -> canonical


class CanonicalizeModel(Protocol):
    def invoke(self, input: Any) -> ThemeTaxonomy: ...


_SYSTEM_PROMPT = (
    "You are organizing raw, free-text labels harvested from Reddit comments into a "
    "coherent taxonomy for a market-research report about a topic.\n"
    f"- Produce at most {MAX_THEMES} themes, each a general cluster with a stable "
    "slug id, a short label, a one-sentence description, and is_pain_point=true when "
    "the theme is predominantly a complaint/frustration.\n"
    "- label_map MUST map every input raw label to exactly one theme id.\n"
    "- competitor_map maps each raw competitor/alternative name to a single canonical "
    "name (merge obvious spelling/casing variants).\n"
    "Keep themes distinct and non-overlapping."
)


def label_counts(labels: list[str]) -> list[tuple[str, int]]:
    """Deduped (label, count) pairs, most frequent first (case-insensitive dedup)."""
    counter: Counter[str] = Counter()
    display: dict[str, str] = {}
    for raw in labels:
        key = raw.strip().lower()
        if not key:
            continue
        counter[key] += 1
        display.setdefault(key, raw.strip())
    return [(display[k], counter[k]) for k, _ in counter.most_common()]


def build_canonicalize_input(
    counts: list[tuple[str, int]], competitors: list[str], *, topic: str
) -> list[dict[str, str]]:
    lines = [f"Topic: {topic}", "", "Candidate theme labels (with occurrence counts):"]
    for label, count in counts:
        lines.append(f"- {label} ({count})")
    if competitors:
        lines.append("")
        lines.append("Raw competitor/alternative names mentioned:")
        for name in sorted(set(competitors)):
            lines.append(f"- {name}")
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def run_canonicalize(
    model: CanonicalizeModel,
    counts: list[tuple[str, int]],
    competitors: list[str],
    *,
    topic: str,
) -> ThemeTaxonomy:
    return model.invoke(build_canonicalize_input(counts, competitors, topic=topic))
