"""Default parameters — the single source of truth (SPEC §10)."""

from __future__ import annotations

# Fetch stage
POSTS_PER_SUBREDDIT = 25
TIME_WINDOW_DAYS = 90
SORT = "top"

# Analyze stage
COMMENT_MIN_SCORE = 5
MAX_THEMES = 10
MAX_QUOTES_PER_THEME = 3

# Model — default gpt-5.4-mini for every step (SPEC §8, §10).
DEFAULT_MODEL = "gpt-5.4-mini"

# Cost guardrail
COST_SOFT_CEILING_USD = 2.00

# Cache
CACHE_TTL_DAYS = 30

# Directories (relative to cwd)
CACHE_DIR = "cache"
RESULTS_DIR = "results"
