"""Fake PRAW objects for driving the fetch stage without live Reddit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FakeAuthor:
    name: str


@dataclass
class FakeComment:
    id: str
    parent_id: str
    body: str
    score: int
    created_utc: float
    permalink: str
    depth: int
    author: Optional[FakeAuthor] = None


class FakeCommentForest:
    def __init__(self, comments: list[FakeComment]) -> None:
        self._comments = comments

    def replace_more(self, limit: int = 0) -> list:  # noqa: ARG002
        return []

    def list(self) -> list[FakeComment]:
        return self._comments


@dataclass
class FakeSubmission:
    id: str
    title: str
    score: int
    created_utc: float
    permalink: str
    num_comments: int
    selftext: str = ""
    url: str = ""
    author: Optional[FakeAuthor] = None
    _comments: list[FakeComment] = field(default_factory=list)

    @property
    def comments(self) -> FakeCommentForest:
        return FakeCommentForest(self._comments)


class FakeSubreddit:
    def __init__(self, submissions: list[FakeSubmission]) -> None:
        self._submissions = submissions

    def search(self, topic, sort="top", time_filter="year", limit=None):  # noqa: ANN001, ARG002
        return list(self._submissions)


class FakeReddit:
    """Maps bare subreddit name -> list of submissions returned by search()."""

    def __init__(self, by_subreddit: dict[str, list[FakeSubmission]]) -> None:
        self._by_subreddit = by_subreddit
        self.read_only = False

    def subreddit(self, name: str) -> FakeSubreddit:
        return FakeSubreddit(self._by_subreddit.get(name, []))
