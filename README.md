# reddit-research

A CLI tool for **personal, non-commercial market research** on Reddit. Give it a
topic and an explicit list of subreddits; it fetches the matching discussion, runs
LLM extraction over a frozen cache, and produces a structured **JSON artifact**
plus a **Markdown report** covering:

- **Theme-clustered pain points** (themes carry an `is_pain_point` flag)
- **Sentiment** — overall and per-theme (label + score + real distribution counts)
- **Representative quotes** with author + permalink back to Reddit
- **Feature requests** (unmet needs)
- **Competitor / alternative mentions**

See [`docs/SPEC.md`](docs/SPEC.md) for the locked specification and
[`docs/design/`](docs/design/) for the schemas.

## Architecture — staged, cache-in-the-middle

```
        fetch stage                         analyze stage
topic + subreddits ──► Reddit (PRAW) ──► cache/<slug>__<hash>.json ──► LLM pipeline ──► results/<slug>__<ts>.json
     + params                              (frozen raw corpus)          (extract→canon→          + .md report
                                                                         aggregate→summarize)
```

The **frozen cache is the seam**: fetch once, then re-analyze many times (tune
prompts, adjust the score cutoff, swap models) against the same corpus without
re-hitting Reddit or re-paying for the fetch.

The LLM does per-comment extraction, theme canonicalization, and the prose
summary; **all counting, sentiment distributions, and quote selection are
deterministic Python** — no fabricated numbers or quotes.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                       # install dependencies into a local venv
cp .env.example .env          # then fill in your secrets
```

### Configuration (`.env`)

| Variable | Required | Purpose |
|---|---|---|
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | yes (fetch) | Reddit read-only "script" app credentials |
| `REDDIT_USER_AGENT` | yes (fetch) | descriptive UA, e.g. `python:reddit-market-researcher:v0.1 (by u/you)` |
| `OPENAI_API_KEY` | yes (analyze) | LLM extraction / summary |
| `LANGSMITH_API_KEY` + `LANGCHAIN_TRACING_V2=true` | optional | authoritative per-run cost tracing |

Create a Reddit "script" app at <https://www.reddit.com/prefs/apps>. See
[`.env.example`](.env.example) for the full template.

## Usage

Run via `uv run` (or activate the venv and call `reddit-research` directly).

```bash
# Fetch the corpus into a query-keyed cache file
uv run reddit-research fetch --topic "Notion" --subreddits r/productivity r/Notion \
    [--posts-per-subreddit 25] [--time-window-days 90] [--sort top]

# Analyze — writes results/<slug>__<timestamp>.json and a matching .md report
uv run reddit-research analyze --topic "Notion" --subreddits r/productivity r/Notion \
    [--use-cached] [--min-score 5] [--model gpt-5.4-mini] [--reduce-model gpt-5.6-sol] [--yes]

# Delete cache files older than N days
uv run reddit-research cache purge --older-than 30
```

`--subreddits` accepts repeated flags or space/comma-separated values in one flag.

### `fetch`

Per subreddit, searches Reddit, client-filters to the time window, keeps the top
`--posts-per-subreddit` posts, and stores **all** their comments (the score cutoff
and relevance filter are analyze-time knobs). Writes one cache file per unique
query (`cache/<topic-slug>__<queryhash>.json`); re-fetching the same query
overwrites it.

### `analyze`

- **Without `--use-cached`** (default): fetches fresh, then analyzes.
- **With `--use-cached`**: reuses the cached corpus for this topic + subreddits
  (falling back to a fresh fetch if none exists). A cache older than the 30-day
  TTL warns but proceeds.
- `--min-score` sets the comment score cutoff (default 5); comments must also
  match the topic keyword.
- `--model` sets the model for all steps (default `gpt-5.4-mini`); `--reduce-model`
  optionally runs the reduce steps (canonicalize + summarize) on a stronger model
  for the mini(map)+sol(reduce) split.
- Before any paid call, a cheap **pre-flight cost estimate** runs; if projected
  cost exceeds the **$2.00** soft ceiling it warns and asks to confirm (`--yes`
  skips). The estimate never hard-aborts.

## Output

Each analyze run writes a timestamped, accumulating pair under `results/`:

- `<topic-slug>__<YYYYMMDDTHHMMSSZ>.json` — the canonical `AnalysisResult`
  ([schema](docs/design/output-schema.md)): code-populated `run_metadata` plus the
  LLM-generated `overall`, `themes`, `feature_requests`, `competitor_mentions`.
- `<topic-slug>__<YYYYMMDDTHHMMSSZ>.md` — a **pure rendering** of that JSON.

## Cost

Default `gpt-5.4-mini` for every step is ~$0.29/run at default volume. The
map(mini)+reduce(sol) split (`--reduce-model`) is ~$0.50/run. With LangSmith
enabled, actual per-run cost is reported from traces; otherwise the local
pre-flight estimate is the fallback. See [SPEC §8](docs/SPEC.md) and
[R3](docs/research/openai-models.md) for the full cost breakdown.

## Privacy & compliance

- Use is **personal / non-commercial** — Reddit's free API tier applies.
- The cache is an **ephemeral working cache** (30-day TTL + `cache purge`), not a
  permanent dataset; quotes keep `author` + `permalink` so deletions can be
  honored on Reddit.
- ⚠️ If LangSmith tracing is enabled, traces include the **Reddit comment text
  sent to the model** and are uploaded to LangSmith's cloud. Leave it off to keep
  everything local.

## Development

```bash
uv run pytest        # test suite (LLM + Reddit boundaries are mocked; no live calls)
uv run mypy          # type check
```
