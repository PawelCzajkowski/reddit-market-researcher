# Research: OpenAI Models & Cost for the Analysis Step (R3, #4)

**Task:** Qualitative extraction from Reddit comments — theme-clustered pain points, sentiment, representative quotes, feature requests, competitor mentions.

**Pricing fetched:** 2026-07-22, from OpenAI's official developer docs (`developers.openai.com`). Prices change frequently; re-verify before relying on the arithmetic below.

---

## Recommendation (summary)

- **Default: a two-model split.**
  - **Map stage** (per-post extraction: sentiment, raw pain points, candidate quotes, competitor/feature mentions) → **GPT-5.4-mini** (`gpt-5.4-mini`, $0.75 in / $4.50 out per 1M). Cheap, high-volume, structured extraction is well within its ability.
  - **Reduce stage** (cross-post theme clustering, quote selection, final report) → **GPT-5.6 Sol** (`gpt-5.6-sol`, $5 in / $30 out per 1M). The synthesis step is where frontier reasoning pays off, and it runs on a small token budget.
  - **Estimated cost: ~$0.50 per default run** (75 posts, 3 subreddits, comment trees included).
- **Simpler single-model alternative: GPT-5.6 Terra** (`gpt-5.6-terra`, $2.50 / $15) — one model for the whole pipeline, ~**$0.96 per run**. Use this if the two-stage orchestration isn't worth the complexity.
- **When to switch tiers:**
  - Bulk sentiment-only / MVP → **GPT-5.4-mini single-model** (~$0.29/run).
  - Quote fidelity and clustering quality are paramount, volume is low → **GPT-5.6 Sol single-model** (~$1.92/run).
  - Runs are async / not latency-sensitive → apply the **Batch API (−50%)** to roughly halve any figure below.

---

## 1. Model options (current, mid-2026)

The current frontier family is **GPT-5.6** (released 2026-07-09), with three variants that share a **1.05M-token context window and 128K max output**. The prior **GPT-5.4** generation and its `mini`/`nano` tiers remain available and are the cost-efficient workhorses.

| Model | ID | Context | Max output | Positioning (official) | In / Out $/1M |
|---|---|---|---|---|---|
| GPT-5.6 Sol | `gpt-5.6-sol` | 1,050,000 | 128,000 | Frontier model for complex professional work / reasoning & coding | 5.00 / 30.00 |
| GPT-5.6 Terra | `gpt-5.6-terra` | 1,050,000 | 128,000 | Balances intelligence and cost (everyday production) | 2.50 / 15.00 |
| GPT-5.6 Luna | `gpt-5.6-luna` | 1,050,000 | 128,000 | Cost-sensitive, high-volume workloads | 1.00 / 6.00 |
| GPT-5.5 | `gpt-5.5` | ~1.1M | 128,000 | Prior flagship (superseded by Sol at same price) | 5.00 / 30.00 |
| GPT-5.4 | `gpt-5.4` | 1,050,000 | 128,000 | Prior frontier; strong cost/quality balance | 2.50 / 15.00 |
| GPT-5.4-mini | `gpt-5.4-mini` | 400,000 | 128,000 | Strongest mini yet (coding, computer use, subagents) | 0.75 / 4.50 |
| GPT-5.4-nano | `gpt-5.4-nano` | (not published in docs) | 128,000 | Fastest/cheapest tier | 0.20 / 1.25 |

**Capability vs price tradeoff for this task:**
- Qualitative *extraction per post* (sentiment, pull candidate quotes, tag competitor/feature mentions) is a structured, bounded task — **mini/nano/Luna** handle it well and are 5–25x cheaper than the frontier tier.
- *Cross-post theme clustering and choosing genuinely representative quotes* is the reasoning-heavy step where **Terra/Sol** produce noticeably tighter, less-hallucinated groupings. This step operates on already-condensed summaries, so it's cheap even on a frontier model.
- Context window is a non-issue: even the 400K mini window dwarfs a single subreddit's comment payload (~35–50K tokens), so the whole map stage can run per-post or per-subreddit without truncation.

*Notes:* GPT-5.5 offers no advantage over GPT-5.6 Sol at identical pricing — prefer Sol. GPT-5.4 and GPT-5.6 Terra are priced identically ($2.50/$15); Terra is the newer generation, so prefer it for the single-model path.

**Sources:**
- Models list — https://developers.openai.com/api/docs/models
- GPT-5.4 — https://developers.openai.com/api/docs/models/gpt-5.4
- GPT-5.4-mini — https://developers.openai.com/api/docs/models/gpt-5.4-mini
- GPT-5.6 Luna — https://developers.openai.com/api/docs/models/gpt-5.6-luna

---

## 2. Pricing (per 1M tokens, fetched 2026-07-22)

| Model | Input | Cached input | Output |
|---|---|---|---|
| GPT-5.6 Sol | $5.00 | $0.50 | $30.00 |
| GPT-5.6 Terra | $2.50 | $0.25 | $15.00 |
| GPT-5.6 Luna | $1.00 | $0.10 | $6.00 |
| GPT-5.5 | $5.00 | $0.50 | $30.00 |
| GPT-5.4 | $2.50 | $0.25 | $15.00 |
| GPT-5.4-mini | $0.75 | $0.075 | $4.50 |
| GPT-5.4-nano | $0.20 | $0.02 | $1.25 |

- **Cached input** = 10% of standard input rate (applies to all models); useful because our system/instruction prompt is reused across every map call.
- **Batch API** = flat **−50%** on both input and output for async jobs.
- A prompt-caching or batch strategy can therefore cut the numbers in section 3 substantially; the estimates below use **standard, non-cached, non-batch** rates as the conservative ceiling.

**Sources:**
- Pricing — https://developers.openai.com/api/docs/pricing
- (Per-model cached/input/output figures cross-checked against the individual model pages cited in section 1.)

---

## 3. Token budgeting — cost of ONE default run

### Volume & token assumptions (stated explicitly)

| Parameter | Value |
|---|---|
| Subreddits | 3 |
| Posts per subreddit | 25 → **75 posts total** |
| Avg comments included per post | 40 |
| Avg tokens per comment | 30 |
| Avg tokens per post title + body | 200 |
| Instruction/system overhead per map call | 600 tokens |
| Extraction output per post (map) | 400 tokens |
| Reduce-stage output (final clustered report) | 3,000 tokens |

### Raw content size

- Comment trees: `75 posts × 40 comments × 30 tokens = 90,000 tokens`
- Post title+body: `75 × 200 = 15,000 tokens`
- **Total content ≈ 105,000 input tokens**

### Pipeline token totals (map = 1 call/post, reduce = 1 call)

- **Map input:** `105,000 content + (75 × 600 overhead) = 150,000 tokens`
- **Map output:** `75 × 400 = 30,000 tokens`
- **Reduce input:** `30,000 (per-post summaries) + 2,000 (instructions) = 32,000 tokens`
- **Reduce output:** `3,000 tokens`
- **Run totals: ~182,000 input / ~33,000 output tokens** (round to **185K in / 33K out**).

### Single-model cost (whole pipeline on one model)

Cost = `input_tokens/1e6 × in_price + output_tokens/1e6 × out_price`, using 0.185M in / 0.033M out.

**Cheap tier — GPT-5.4-mini ($0.75 / $4.50):**
- Input: `0.185 × 0.75 = $0.139`
- Output: `0.033 × 4.50 = $0.149`
- **Total ≈ $0.29 / run**

**Mid tier — GPT-5.6 Terra ($2.50 / $15):**
- Input: `0.185 × 2.50 = $0.463`
- Output: `0.033 × 15 = $0.495`
- **Total ≈ $0.96 / run**

**Strong tier — GPT-5.6 Sol ($5 / $30):**
- Input: `0.185 × 5 = $0.925`
- Output: `0.033 × 30 = $0.990`
- **Total ≈ $1.92 / run**

### Two-model split cost (recommended)

Map stage on **mini** (150K in / 30K out), reduce stage on **Sol** (32K in / 3K out):
- Mini map: `0.150 × 0.75 + 0.030 × 4.50 = 0.1125 + 0.135 = $0.248`
- Sol reduce: `0.032 × 5 + 0.003 × 30 = 0.160 + 0.090 = $0.250`
- **Total ≈ $0.50 / run** — near mini-only cost, with frontier-quality synthesis.

### Scaling & levers

- Cost is dominated by the **map stage input** (the comment payload), which scales linearly with posts × comments/post. Doubling comment depth to 80/post roughly adds `+90K` input tokens (≈ +$0.07 on mini, +$0.45 on Sol).
- **Prompt caching** the reused instruction prompt and **Batch API (−50%)** each cut cost materially; a batched two-model split lands near **~$0.25/run**.

---

## 4. Recommendation

**Default: two-model split — GPT-5.4-mini (map) + GPT-5.6 Sol (reduce), ~$0.50/run.**
This matches the task shape: the expensive, high-volume part (reading every comment) is mechanical extraction that a cheap model does reliably; the part that needs judgment (clustering themes, picking representative quotes without fabrication) runs on the frontier model but over a tiny, pre-condensed token budget, so it costs almost nothing extra.

**Switch tiers when:**
- **Bulk sentiment / MVP / cost-first:** GPT-5.4-mini for the whole pipeline (~$0.29/run). Acceptable if clustering can be coarse.
- **Balanced, single-model simplicity:** GPT-5.6 Terra for the whole pipeline (~$0.96/run) — fewer moving parts than the split, still strong synthesis.
- **Quality-critical / low volume:** GPT-5.6 Sol single-model (~$1.92/run) when quote fidelity and cluster precision matter more than cost.
- **Async pipelines:** add Batch API (−50%) and prompt caching on top of any option above.

**Suggested config knobs:** `map_model` (default `gpt-5.4-mini`), `reduce_model` (default `gpt-5.6-sol`), plus a `single_model` fallback (`gpt-5.6-terra`) for a simple mode. This lets cost/quality be tuned without code changes.
