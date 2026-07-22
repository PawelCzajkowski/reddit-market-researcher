"""Markdown report — a *pure* rendering of an AnalysisResult.

No counting, selection, or recomputation happens here: every number shown is read
straight from the JSON artifact (SPEC §7). Theme / feature-request / competitor
sections are added by later tickets; this skeleton renders the metadata header,
executive summary, and overall sentiment.
"""

from __future__ import annotations

from .models.output import AnalysisResult, SentimentBlock


def _sentiment_md(sb: SentimentBlock) -> str:
    d = sb.distribution
    return (
        f"**{sb.label}** (score {sb.score:+.2f}) — "
        f"{d.positive} positive / {d.negative} negative / {d.neutral} neutral"
    )


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
    return "\n".join(lines)
