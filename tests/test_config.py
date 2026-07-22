"""Config/secret loading behaviour."""

from __future__ import annotations

import pytest

from reddit_research.config import (
    ConfigError,
    OpenAICredentials,
    RedditCredentials,
    langsmith_enabled,
)


def test_reddit_credentials_missing_secret_raises_clear_error() -> None:
    with pytest.raises(ConfigError) as exc:
        RedditCredentials.from_env()
    assert "REDDIT_CLIENT_ID" in str(exc.value)
    assert ".env" in str(exc.value)


def test_reddit_credentials_load_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "agent")
    creds = RedditCredentials.from_env()
    assert creds.client_id == "id"
    assert creds.user_agent == "agent"


def test_openai_credentials_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError):
        OpenAICredentials.from_env()


def test_blank_secret_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    with pytest.raises(ConfigError):
        OpenAICredentials.from_env()


@pytest.mark.parametrize(
    ("key", "flag", "expected"),
    [
        ("k", "true", True),
        ("k", "1", True),
        ("k", "false", False),
        ("", "true", False),
        ("k", "", False),
    ],
)
def test_langsmith_enabled(
    monkeypatch: pytest.MonkeyPatch, key: str, flag: str, expected: bool
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", key)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", flag)
    assert langsmith_enabled() is expected
