# Reddit Market-Research Analyzer — Locked Spec

**Status:** locked (all shaping decisions resolved via [wayfinder map #1](https://github.com/PawelCzajkowski/reddit-market-researcher/issues/1)). This document is sharp enough to build from without further shaping decisions. Full schemas live in `docs/design/`; this spec references them rather than duplicating.

## 1. What it is

A **CLI tool for personal, non-commercial market research** on Reddit. You give it a **topic/keyword** and an **explicit list of subreddits**; it fetches the matching discussion, runs LLM extraction over it, and produces a structured **JSON artifact** plus a **Markdown report** covering:

- **Theme-clustered pain points** (themes carry an `is_pain_point` flag)
- **Sentiment** — overall and per-theme (label + score + real distribution counts)
- **Representative quotes** with author + permalink back to Reddit
- **Feature requests** (unmet needs)
- **Competitor / alternative mentions**

## 2. Architecture — staged, cache-in-the-middle

```
        fetch stage                         analyze stage
topic + subreddits ──► Reddit (PRAW) ──► cache/<slug>__<hash>.json ──► LLM pipeline ──► results/<slug>__<ts>.json
     + params                              (frozen raw corpus)          (extract→canon→          + .md report
                                                                         aggregate→summarize)
```

The **frozen cache** is the seam: fetch once, then re-analyze many times (tune prompts, adjust cutoffs) against the same corpus without re-hitting Reddit or re-paying. This is the core workflow the design optimizes for.

## 3. Stack & dependencies

- **Language:** Python 3.11+
- **Reddit:** `praw` (read-only script app)
- **LLM:** `langchain` + `langchain-openai` — **thin slice only** (`init_chat_model`, `with_structured_output`, `model.batch()`, `usage_metadata`); orchestration is hand-written. OpenAI models.
- **Validation:** `pydantic` v2 (all schemas)
- **Cost/observability:** `langsmith` (tracing → per-run cost)
- **Config:** `python-dotenv`
- **CLI:** `typer` (or `click`) — *build-agent's discretion*
- **Report rendering:** plain templating (f-strings/`jinja2`) — *build-agent's discretion*

## 4. CLI interface *(shape recommended; names at build-agent's discretion)*

```
reddit-research fetch   --topic "Notion" --subreddits r/productivity r/Notion
                        [--posts-per-subreddit 25] [--time-window-days 90] [--sort top]
reddit-research analyze --topic "Notion" --subreddits r/productivity r/Notion
                        [--use-cached] [--min-score 5] [--model gpt-5.4-mini]
                        [--yes]                     # skip the pre-flight cost confirm
reddit-research cache purge --older-than 30
```

- `fetch` writes/overwrites the query-keyed cache file. `analyze` with `--use-cached` reuses it (else fetches); without the flag, default behavior is fetch-fresh.
- `analyze` produces a timestamped `results/` JSON + Markdown pair.

## 5. Fetch stage — see [`docs/design/cache-schema.md`](design/cache-schema.md)

- **PRAW read-only** ("script" app). Per subreddit: `subreddit.search(topic, sort="top", time_filter="year")` → **client-side filter to 90 days** → take top `posts_per_subreddit`. `submission.comments.replace_more(limit=0)` then `.list()`; **store all comments** (filtering is deferred to analyze).
- **Rate limits:** PRAW auto-throttles via `X-Ratelimit`; a default run is ~75–100 requests, far under 100 QPM. Use a descriptive `user_agent`.
- **Cache file:** `cache/<topic-slug>__<queryhash>.json`, one per unique query (queryhash = topic + sorted subreddits + fetch params). `fetched_at` stored inside; `--use-cached` warns but proceeds past a 30-day TTL. Schema = `RawCache` (metadata + posts[] with flat, denormalized comments).

## 6. Analyze stage — see [`docs/design/analysis-pipeline.md`](design/analysis-pipeline.md)

`filter (code) → map/extract (LLM, batched) → canonicalize themes (LLM, 1 call) → aggregate (code) → summarize (LLM, 1 call)`.

- **LLM extracts, code aggregates:** the model does per-comment extraction, theme canonicalization, and the prose summary; **all counting, sentiment distributions, and quote selection are deterministic Python** (traceable, no fabricated numbers).
- **Filter (code, 0 tokens):** keep comments with `score ≥ 5` (default) AND topic-keyword relevance match. *(Semantic relevance = future.)*
- **Theme coherence:** dedupe the map step's candidate labels → one LLM call yields a coherent ~10-theme taxonomy + `label → theme` map → code assigns.
- **Sentiment:** per-comment `label`+`score` from the map step → code computes exact `distribution` tallies and mean `score`; `label` derives by rule with **"mixed" when positive/negative are both substantial**.
- **Caps/ordering:** ≤3 quotes/theme, ~10 themes, prevalence-desc — enforced in code (OpenAI strict mode can't cap arrays).

## 7. Output — see [`docs/design/output-schema.md`](design/output-schema.md)

Canonical artifact = `AnalysisResult` = code-populated `run_metadata` + the LLM-generated `AnalysisPayload` (`overall`, `themes`, `feature_requests`, `competitor_mentions`). The `AnalysisPayload` sub-objects are produced via `with_structured_output(..., method="json_schema", strict=True)`. The Markdown report is a **pure rendering** of this JSON — no independent logic.

## 8. Models & cost

- **Default: `gpt-5.4-mini` for every step** (~$0.29/run at default volume). Configurable up to the **mini(map)+sol(reduce) split** (~$0.50) or **single Terra** (~$0.96) via `--model` / config — switched manually.
- **Cost:** LangSmith tracing reports actual per-run cost. A cheap **pre-flight estimate** (`count_tokens_approximately` × a local price constant) warns + requires confirm if projected cost > **$2.00** (`--yes` skips).

## 9. Config & secrets

`.env` (git-ignored), loaded via `python-dotenv`:

```
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=python:reddit-market-researcher:v0.1 (by u/yourname)
OPENAI_API_KEY=...
LANGSMITH_API_KEY=...          # optional; enables per-run cost tracking
LANGCHAIN_TRACING_V2=true      # optional
```

## 10. Defaults (single source of truth)

| Parameter | Default |
|---|---|
| posts per subreddit | 25 |
| time window | 90 days |
| post sort | `top` |
| comment score cutoff | 5 |
| relevance filter | topic-keyword match |
| themes | ~10, ordered by prevalence desc |
| quotes per theme | ≤3, top by comment score |
| model (all steps) | **`gpt-5.4-mini`** |
| cost soft ceiling | $2.00 (warn + confirm) |
| cache TTL | 30 days (warn, proceed) |

## 11. Compliance & privacy notes (⚠️ read before first run)

- **Reddit ToS — personal/non-commercial use → free tier applies.** No paid tier needed for this effort. **Skim the [Reddit Data API Terms](https://www.redditinc.com/policies/data-api-terms) in a browser once** before building — R1 could not fetch them (Reddit blocks automated fetch), so the free/commercial classification and storage rules rest on secondary sources. Revisit only if this ever becomes commercial.
- **Honor deletions:** the cache is an **ephemeral working cache** (30-day TTL + `cache purge`), not a permanent dataset.
- **OpenAI no-training:** confirm the OpenAI API's default (API data is not used for training) still holds for your account/models.
- **LangSmith privacy:** if enabled, traces — **including the Reddit comment text sent to the model** — are sent to LangSmith's cloud. Acceptable for personal use; disable if undesired (cost then comes only from the local estimate).

## 12. Out of scope (this effort)

- Dashboard / web UI (the JSON makes it a pure rendering layer later)
- Whole-Reddit search (deliberately narrowed to chosen subreddits)
- Subreddit auto-discovery
- Semantic (embedding-based) relevance filtering and embed+cluster theming — future upgrades once the keyword/LLM-canonicalization baseline works

## Provenance

Decisions and their rationale: map [#1](https://github.com/PawelCzajkowski/reddit-market-researcher/issues/1). Research: [R1](https://github.com/PawelCzajkowski/reddit-market-researcher/issues/2)/[R2](https://github.com/PawelCzajkowski/reddit-market-researcher/issues/3)/[R3](https://github.com/PawelCzajkowski/reddit-market-researcher/issues/4) (`docs/research/`). Design: [D1](https://github.com/PawelCzajkowski/reddit-market-researcher/issues/5)/[D2](https://github.com/PawelCzajkowski/reddit-market-researcher/issues/6)/[D3](https://github.com/PawelCzajkowski/reddit-market-researcher/issues/7) (`docs/design/`).
