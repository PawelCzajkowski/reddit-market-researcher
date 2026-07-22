"""Configuration and secret loading from a git-ignored .env file.

Secrets never live in code or the repo; they load from the environment (populated
from .env via python-dotenv). Required secrets are validated lazily — only the
values a given command actually needs are demanded, so `--help` and stubs never
fail on a missing key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """A required configuration value is missing or malformed."""


def load_env() -> None:
    """Load .env into the process environment (idempotent, real env wins)."""
    load_dotenv(override=False)


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"Missing required config value {name!r}. "
            f"Set it in your .env file (see .env.example)."
        )
    return value


@dataclass(frozen=True)
class RedditCredentials:
    client_id: str
    client_secret: str
    user_agent: str

    @classmethod
    def from_env(cls) -> "RedditCredentials":
        return cls(
            client_id=_require("REDDIT_CLIENT_ID"),
            client_secret=_require("REDDIT_CLIENT_SECRET"),
            user_agent=_require("REDDIT_USER_AGENT"),
        )


@dataclass(frozen=True)
class OpenAICredentials:
    api_key: str

    @classmethod
    def from_env(cls) -> "OpenAICredentials":
        return cls(api_key=_require("OPENAI_API_KEY"))


def langsmith_enabled() -> bool:
    """True when LangSmith tracing is configured (key present + tracing flag on)."""
    key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    tracing = os.environ.get("LANGCHAIN_TRACING_V2", "").strip().lower()
    return bool(key) and tracing in {"1", "true", "yes"}
