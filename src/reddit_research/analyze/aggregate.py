"""Step 4 — aggregate (code, 0 tokens).

Everything countable is computed here, deterministically: theme assignment via the
canonicalized label map, prevalence, sentiment distributions, and representative
quote selection. Caps and ordering (<=3 quotes/theme, ~10 themes, prevalence-desc)
are enforced in code because OpenAI strict mode cannot cap arrays (D1/D2).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ..defaults import MAX_QUOTES_PER_THEME, MAX_THEMES
from ..fetch.query import slugify
from ..models.cache import CachedComment
from ..models.output import (
    CompetitorMention,
    CompetitorRelationship,
    FeatureRequest,
    Prevalence,
    Quote,
    Theme,
)
from .canonicalize import ThemeTaxonomy
from .extract import CommentExtract
from .sentiment import aggregate_sentiment


@dataclass
class AnalyzedComment:
    """A filtered comment paired with its LLM extract (aligned by index)."""

    comment: CachedComment
    extract: CommentExtract


def pair(comments: list[CachedComment], extracts: list[CommentExtract]) -> list[AnalyzedComment]:
    return [AnalyzedComment(comment=c, extract=e) for c, e in zip(comments, extracts)]


def comment_to_quote(comment: CachedComment) -> Quote:
    return Quote(
        text=comment.body,
        author=comment.author,
        type="comment",
        permalink=comment.permalink,
        subreddit=comment.subreddit,
        post_title=comment.post_title,
        score=comment.score,
        created_at=comment.created_utc,
    )


def _percentage(count: int, total: int) -> float:
    return round(count / total * 100, 1) if total else 0.0


def _representative_quotes(members: list[AnalyzedComment]) -> list[Quote]:
    """Top-K quote_worthy members by comment score, highest first."""
    worthy = [m for m in members if m.extract.quote_worthy]
    worthy.sort(key=lambda m: m.comment.score, reverse=True)
    return [comment_to_quote(m.comment) for m in worthy[:MAX_QUOTES_PER_THEME]]


def aggregate_themes(
    analyzed: list[AnalyzedComment],
    taxonomy: ThemeTaxonomy,
    *,
    total_analyzed: int,
) -> list[Theme]:
    """Build the ordered, capped theme list from the canonicalized taxonomy."""
    label_to_theme = {la.raw_label.strip().lower(): la.theme_id for la in taxonomy.label_map}

    members: dict[str, list[AnalyzedComment]] = {t.id: [] for t in taxonomy.themes}
    for item in analyzed:
        assigned: set[str] = set()
        for raw in item.extract.candidate_theme_labels:
            theme_id = label_to_theme.get(raw.strip().lower())
            if theme_id is not None and theme_id in members:
                assigned.add(theme_id)
        for theme_id in assigned:
            members[theme_id].append(item)

    themes: list[Theme] = []
    for tax in taxonomy.themes:
        group = members[tax.id]
        if not group:
            continue  # taxonomy theme with no assigned comments -> drop
        sentiment = aggregate_sentiment(
            (m.extract.sentiment.label, m.extract.sentiment.score) for m in group
        )
        themes.append(
            Theme(
                id=tax.id,
                label=tax.label,
                description=tax.description,
                is_pain_point=tax.is_pain_point,
                sentiment=sentiment,
                prevalence=Prevalence(
                    mention_count=len(group),
                    comment_percentage=_percentage(len(group), total_analyzed),
                ),
                representative_quotes=_representative_quotes(group),
            )
        )

    # order by prevalence desc (stable tie-break on id), keep top ~MAX_THEMES
    themes.sort(key=lambda t: (-t.prevalence.mention_count, t.id))
    return themes[:MAX_THEMES]


_WS_RE = re.compile(r"\s+")


def _canonical_text_key(text: str) -> str:
    """Normalize free-text for grouping: lowercase, strip punctuation/whitespace."""
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return _WS_RE.sub(" ", cleaned).strip()


def aggregate_feature_requests(
    analyzed: list[AnalyzedComment], *, total_analyzed: int
) -> list[FeatureRequest]:
    """Group feature requests by canonicalized text; prevalence + <=3 quotes each."""
    groups: dict[str, list[AnalyzedComment]] = {}
    for item in analyzed:
        if not item.extract.is_feature_request or not item.extract.feature_request_text:
            continue
        key = _canonical_text_key(item.extract.feature_request_text)
        if key:
            groups.setdefault(key, []).append(item)

    requests: list[FeatureRequest] = []
    for members in groups.values():
        texts = [m.extract.feature_request_text or "" for m in members]
        request = Counter(texts).most_common(1)[0][0]
        rationale = next(
            (m.extract.feature_request_rationale for m in members
             if m.extract.feature_request_rationale),
            "",
        )
        requests.append(
            FeatureRequest(
                id=slugify(request),
                request=request,
                rationale=rationale or "",
                prevalence=Prevalence(
                    mention_count=len(members),
                    comment_percentage=_percentage(len(members), total_analyzed),
                ),
                representative_quotes=_representative_quotes(members),
            )
        )

    requests.sort(key=lambda fr: (-fr.prevalence.mention_count, fr.id))
    return requests


def _mode_relationship(
    relationships: list[CompetitorRelationship | None],
) -> CompetitorRelationship | None:
    present = [r for r in relationships if r is not None]
    if not present:
        return None
    return Counter(present).most_common(1)[0][0]


def aggregate_competitors(
    analyzed: list[AnalyzedComment], taxonomy: ThemeTaxonomy
) -> list[CompetitorMention]:
    """Group competitor mentions by canonical name (via competitor_map)."""
    name_map = {ca.raw_name.strip().lower(): ca.canonical_name for ca in taxonomy.competitor_map}

    members: dict[str, list[AnalyzedComment]] = {}
    relationships: dict[str, list[CompetitorRelationship | None]] = {}
    for item in analyzed:
        seen: set[str] = set()
        for cm in item.extract.competitor_mentions:
            canonical = name_map.get(cm.name.strip().lower(), cm.name.strip())
            if not canonical or canonical in seen:
                continue
            # count each comment once per competitor — for both mention_count and
            # the relationship mode, so the two stay on the same per-comment basis
            seen.add(canonical)
            members.setdefault(canonical, []).append(item)
            relationships.setdefault(canonical, []).append(cm.relationship)

    mentions: list[CompetitorMention] = []
    for name, group in members.items():
        sentiment = aggregate_sentiment(
            (m.extract.sentiment.label, m.extract.sentiment.score) for m in group
        )
        mentions.append(
            CompetitorMention(
                name=name,
                relationship=_mode_relationship(relationships[name]),
                sentiment=sentiment,
                mention_count=len(group),
                representative_quotes=_representative_quotes(group),
            )
        )

    mentions.sort(key=lambda cm: (-cm.mention_count, cm.name))
    return mentions
