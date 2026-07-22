"""Markdown report — a *pure* rendering of an AnalysisResult.

No counting, selection, or recomputation happens here: every number shown is read
straight from the JSON artifact (SPEC §7). Theme / feature-request / competitor
sections are added by later tickets; this skeleton renders the metadata header,
executive summary, and overall sentiment.
"""

from __future__ import annotations

from .models.output import AnalysisResult, Quote, SentimentBlock, Theme


def _sentiment_md(sb: SentimentBlock) -> str:
    d = sb.distribution
    return (
        f"**{sb.label}** (score {sb.score:+.2f}) — "
        f"{d.positive} positive / {d.negative} negative / {d.neutral} neutral"
    )


def _quote_md(q: Quote) -> str:
    author = q.author or "unknown"
    return (
        f"> {q.text}\n>\n"
        f"> — {author}, [{q.subreddit}]({q.permalink}) "
        f"(score {q.score})"
    )


def _theme_md(theme: Theme) -> list[str]:
    tag = " 🔴 pain point" if theme.is_pain_point else ""
    lines = [
        f"### {theme.label}{tag}",
        "",
        theme.description,
        "",
        f"- **Prevalence:** {theme.prevalence.mention_count} mentions "
        f"({theme.prevalence.comment_percentage:.1f}% of analyzed comments)",
        f"- **Sentiment:** {_sentiment_md(theme.sentiment)}",
        "",
    ]
    for q in theme.representative_quotes:
        lines.append(_quote_md(q))
        lines.append("")
    return lines


def render_report(result: AnalysisResult) -> str:
    m = result.run_metadata
    lines: list[str] = []
    lines.append(f"# Reddit market research: {m.topic}")
    lines.append("")
    lines.append(f"- **Subreddits:** {', '.join(m.subreddits)}")
    lines.append(f"- **Generated:** {m.generated_at}")
    lines.append(f"- **Models:** map `{m.models.map}`, reduce `{m.models.reduce}`")
    lines.append(
        f"- **Corpus:** {m.corpus.post_count} posts / "
        f"{m.corpus.comment_count} comments"
    )
    for s in m.corpus.by_subreddit:
        lines.append(
            f"  - {s.subreddit}: {s.post_count} posts / {s.comment_count} comments"
        )
    lines.append(
        f"- **Params:** posts/subreddit {m.params.posts_per_subreddit}, "
        f"window {m.params.time_window_days}d, min score {m.params.comment_min_score}, "
        f"sort {m.params.sort}"
    )
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(result.overall.executive_summary)
    lines.append("")
    lines.append("## Overall sentiment")
    lines.append("")
    lines.append(_sentiment_md(result.overall.sentiment))
    lines.append("")

    if result.themes:
        lines.append("## Themes")
        lines.append("")
        for theme in result.themes:
            lines.extend(_theme_md(theme))

    return "\n".join(lines)
