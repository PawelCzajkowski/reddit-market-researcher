# Analysis Pipeline Design (D2, #6) — LOCKED

How the analyze stage turns a `RawCache` ([D3](cache-schema.md)) into an `AnalysisResult` ([D1](output-schema.md)). Built on [R2](../research/langchain-extraction.md) (thin LangChain slice; coherence is our design) and [R3](../research/openai-models.md) (models/cost).

## Locked decisions

1. **LLM extracts, code aggregates.** The LLM does per-comment extraction, theme canonicalization, and the prose summary. All counting — prevalence, sentiment distributions, quote selection — is deterministic Python. Cheaper, reproducible, and no fabricated numbers/quotes.
2. **Theme coherence = LLM canonicalization** (R2 option A): dedupe the map step's candidate labels → one call yields a coherent ~10-theme taxonomy + a `raw-label → canonical-theme` map; code assigns comments via that map. (Embed+cluster deferred to fog — only worth it at far larger scale.)
3. **Sentiment = per-comment → code aggregation.** Each comment gets `label`+`score` in the map step; `distribution` is exact label tallies, `score` is the mean, `label` derives by rule with **"mixed" when positive/negative are both substantial** (neither ≳60%).
4. **Default model = `gpt-5.4-mini` for *all* steps** (~$0.29/run, R3). Configurable up to the **mini(map)+sol(reduce) split** (~$0.50) or **single Terra** (~$0.96); switched by config, not code.
5. **Cost tracking = LangSmith.** Tracing computes actual per-run cost automatically (needs `LANGSMITH_API_KEY`). Plus a **cheap pre-flight estimate** (`count_tokens_approximately` × a local price constant); if projected cost > **$2.00 soft ceiling**, **warn + require confirm** (`--yes` skips). No hard abort — iteration against the frozen cache stays frictionless.

## Pipeline

```
RawCache
 │
 1. FILTER  (Python, 0 tokens)
 │    keep comments with score ≥ cutoff (default 5) AND relevance match
 │    (relevance default = topic-keyword match in body; semantic filtering → fog)
 │
 2. MAP  (default gpt-5.4-mini; model.batch() over chunks sized to a token budget)
 │    per comment → CommentExtract {
 │       comment_id, sentiment{label, score},
 │       candidate_theme_labels: list[str],           # short free-text phrases
 │       is_feature_request: bool, feature_request_text: str | None,
 │       competitor_mentions: list[{name, relationship | None}],
 │       quote_worthy: bool }
 │    with_structured_output(CommentExtract, method="json_schema", strict=True)
 │
 3. CANONICALIZE  (default gpt-5.4-mini; 1 call)
 │    input: deduped candidate_theme_labels + occurrence counts (+ raw competitor names)
 │    output: ThemeTaxonomy {
 │       themes: [{id, label, description, is_pain_point}],   # ~10, coherent
 │       label_map: { raw_label -> theme_id },
 │       competitor_map: { raw_name -> canonical_name } }
 │
 4. AGGREGATE  (Python, 0 tokens)
 │    • assign each comment → canonical theme(s) via label_map
 │    • per theme: mention_count, comment_percentage (of total analyzed comments),
 │      sentiment distribution (tally) + mean score, is_pain_point (from taxonomy)
 │    • representative_quotes = top-K (≤3) by comment.score among quote_worthy
 │      comments in the theme, built from cache fields → D1 Quote
 │    • order themes by prevalence desc, keep top ~10
 │    • overall sentiment = distribution + mean over all analyzed comments
 │    • feature_requests: group via canonicalized text, prevalence + quotes
 │    • competitor_mentions: group via competitor_map, relationship = mode,
 │      sentiment distribution, mention_count, quotes
 │
 5. SUMMARIZE  (default gpt-5.4-mini; 1 call)
 │    input: aggregated top themes + counts + overall sentiment (all code-computed)
 │    output: executive_summary prose (grounded in the real numbers)
 │
 └─→ AnalysisPayload {overall, themes, feature_requests, competitor_mentions}
     + run_metadata (cache metadata + params + model ids)  →  AnalysisResult (D1)
     →  results/<topic-slug>__<YYYYMMDDTHHMMSSZ>.json  +  .md report
```

**Caps/ordering (D1 constraint):** "≤3 quotes/theme, ~10 themes, ordered by prevalence" are enforced in **code** (step 4 selection), not left to the model — strict mode can't cap arrays.

**LangChain surface (R2):** only the thin slice — `init_chat_model` (provider/model swap via config), `with_structured_output`, `model.batch()` for the map step, `usage_metadata`. No agents/LangGraph. Map-reduce orchestration is ours (steps 3–4).

## Cost

- Default mini-only ≈ **$0.29/run** at default volume (R3). Map step dominates tokens; canonicalize + summarize are two small calls.
- **LangSmith** enabled via `LANGSMITH_API_KEY` / `LANGCHAIN_TRACING_V2` → authoritative per-run cost. ⚠️ Traces include the comment text sent to the model and go to LangSmith's cloud — acceptable for personal use; recorded for S1. If LangSmith lacks pricing for a brand-new model id, it may report tokens without dollars — the local pre-flight estimate is the fallback.
- Pre-flight soft ceiling **$2.00** → warn + confirm.

## Handoffs → S1

- **Secrets/config:** Reddit (`client_id`/`secret`/`user_agent`) + OpenAI (`OPENAI_API_KEY`) + **LangSmith (`LANGSMITH_API_KEY`)**. Privacy note: LLM traces (with comment text) go to LangSmith cloud.
- **Defaults:** model `gpt-5.4-mini` all steps; score cutoff 5; relevance = keyword match; quotes ≤3/theme; ~10 themes.
- **Confirm OpenAI API no-training** (open ToS item).
