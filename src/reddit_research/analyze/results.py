"""Results-file naming: results/<topic-slug>__<YYYYMMDDTHHMMSSZ>.{json,md}.

Timestamped and accumulating — each analysis run yields a new pair, building a
diffable history as prompts are tuned (D3 §naming).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..fetch.query import slugify


def timestamp_slug(generated_at: datetime) -> str:
    return generated_at.strftime("%Y%m%dT%H%M%SZ")


def result_paths(
    topic: str, generated_at: datetime, results_dir: str | Path
) -> tuple[Path, Path]:
    """(json_path, md_path) for a run — matching timestamped basenames."""
    base = f"{slugify(topic)}__{timestamp_slug(generated_at)}"
    d = Path(results_dir)
    return d / f"{base}.json", d / f"{base}.md"
