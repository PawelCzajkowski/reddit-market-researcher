"""Thin LangChain surface (R2): construct structured models via init_chat_model.

Only the thin slice is used — `init_chat_model` (provider/model swap via config)
and `with_structured_output(..., method="json_schema", strict=True)`. No agents,
no LangGraph; the map-reduce orchestration is hand-written in `pipeline.py`.
Imports are local so the package loads (and tests run) without the OpenAI stack
being exercised.
"""

from __future__ import annotations

from typing import Any

from .canonicalize import ThemeTaxonomy
from .extract import MapBatchOutput
from .summarize import SummaryOutput


def build_chat_model(model_id: str) -> Any:
    from langchain.chat_models import init_chat_model

    return init_chat_model(model_id, model_provider="openai")


def build_map_model(model_id: str) -> Any:
    """A structured runnable that returns MapBatchOutput per input (strict json_schema)."""
    return build_chat_model(model_id).with_structured_output(
        MapBatchOutput, method="json_schema", strict=True
    )


def build_canonicalize_model(model_id: str) -> Any:
    """A structured runnable that returns a ThemeTaxonomy (strict json_schema)."""
    return build_chat_model(model_id).with_structured_output(
        ThemeTaxonomy, method="json_schema", strict=True
    )


def build_summary_model(model_id: str) -> Any:
    """A structured runnable that returns a SummaryOutput (strict json_schema)."""
    return build_chat_model(model_id).with_structured_output(
        SummaryOutput, method="json_schema", strict=True
    )
