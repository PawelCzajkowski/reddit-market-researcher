"""Shared test fixtures."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _clean_secret_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure secrets from the developer's real environment never leak into tests."""
    for key in (
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USER_AGENT",
        "OPENAI_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_TRACING_V2",
    ):
        monkeypatch.delenv(key, raising=False)
