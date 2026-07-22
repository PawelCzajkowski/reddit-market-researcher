"""CLI skeleton: help surfaces every command/flag; stubs run without crashing."""

from __future__ import annotations

from typer.testing import CliRunner

from reddit_research.cli import _normalize_subreddits, app


def test_root_help_lists_all_commands(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("fetch", "analyze", "cache"):
        assert cmd in result.output


def test_fetch_help_shows_spec_flags(runner: CliRunner) -> None:
    result = runner.invoke(app, ["fetch", "--help"])
    assert result.exit_code == 0
    for flag in ("--topic", "--subreddits", "--posts-per-subreddit", "--time-window-days", "--sort"):
        assert flag in result.output


def test_analyze_help_shows_spec_flags(runner: CliRunner) -> None:
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    for flag in ("--topic", "--subreddits", "--use-cached", "--min-score", "--model", "--yes"):
        assert flag in result.output


def test_cache_purge_help_shows_older_than(runner: CliRunner) -> None:
    result = runner.invoke(app, ["cache", "purge", "--help"])
    assert result.exit_code == 0
    assert "--older-than" in result.output


def test_normalize_subreddits_flattens_and_dedupes() -> None:
    assert _normalize_subreddits(["r/a r/b", "r/c,r/a"]) == ["r/a", "r/b", "r/c"]


def test_normalize_subreddits_repeated_flags() -> None:
    assert _normalize_subreddits(["r/a", "r/b"]) == ["r/a", "r/b"]
