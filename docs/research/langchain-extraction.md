# LangChain patterns for reliable structured extraction from Reddit comments (OpenAI)

Research ticket R2 (GitHub issue #3). Date: 2026-07-22.
Stack: Python + LangChain + OpenAI. Sources are the official LangChain docs (`docs.langchain.com`) and API reference (`reference.langchain.com`); URLs cited inline.

---

## Recommendation (summary)

- **Structured output**: Use OpenAI's native structured output via LangChain. For a stateless "extract from one comment/chunk" call, use `model.with_structured_output(Schema, method="json_schema", strict=True)` with a **Pydantic** schema. For an agentic flow, `create_agent(..., response_format=Schema)` auto-selects the provider-native strategy. Pydantic gives you runtime validation; `strict=True` makes OpenAI enforce the JSON Schema at decode time. This is the most reliable combination the docs offer.
- **Batching / map-reduce**: LangChain's building blocks are `model.batch()` / `batch_as_completed()` (client-side parallelism) for the *map* step, and **LangGraph's map-reduce via the `Send` API + a reducer channel** for fan-out/fan-in. There is **no built-in cross-batch theme reconciliation** — coherent clustering is your design, not a framework feature (see Section 2). The reliable pattern is two-pass: per-chunk extract -> global reduce/canonicalize against a shared theme taxonomy (or embed + cluster the extracted spans).
- **Cost control**: LangChain surfaces per-call token usage on `AIMessage.usage_metadata` and aggregate usage via `UsageMetadataCallbackHandler` / `get_usage_metadata_callback()`. Pre-flight counting uses `get_num_tokens()` or `count_tokens_approximately()`. Chunk sizing is done with `RecursiveCharacterTextSplitter`. LangChain helps aggregate usage but does **not** compute dollar cost and can hide token growth inside agent loops/retries.
- **Fit verdict**: For this pipeline (high-volume, stateless, per-comment extraction + a merge step), **LangChain is worth it only for its thin adapters** — `with_structured_output`, `batch`, `usage_metadata`, provider-swap via `init_chat_model`. The heavy machinery (agents, LangGraph) is overkill for pure extraction. A defensible middle path: use `langchain-openai`'s `ChatOpenAI.with_structured_output` + `batch` and orchestrate the map-reduce yourself, or drop to the plain OpenAI SDK if you never need provider portability. See Section 4.

---

## 1. Structured output — schema-validated JSON from OpenAI

**Two entry points, same underlying mechanism.**

### 1a. Direct on the model (recommended for stateless extraction)

`with_structured_output` binds a schema and returns parsed objects.
Source: https://docs.langchain.com/oss/python/langchain/models#structured-output

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.chat_models import init_chat_model

class CommentExtract(BaseModel):
    """Structured analysis of a single Reddit comment."""
    sentiment: Literal["positive", "neutral", "negative"] = Field(description="Overall sentiment")
    themes: list[str] = Field(description="Themes mentioned. Lowercase, 1-3 words each.")
    is_actionable_feedback: bool = Field(description="True if it contains a concrete product suggestion or complaint")

model = init_chat_model("openai:gpt-5.4-mini")
extractor = model.with_structured_output(CommentExtract, method="json_schema", strict=True)
result = extractor.invoke("This app keeps crashing on launch, super frustrating")
# CommentExtract(sentiment='negative', themes=['crash on launch'], is_actionable_feedback=True)
```

**Schema types supported** (all via the same API):
- **Pydantic `BaseModel`** — richest: field validation, descriptions, nested models, constraints (`ge`, `le`), returns a *validated instance*. **Recommended.**
- **TypedDict** / **dataclass** — no runtime validation, returns a dict.
- **JSON Schema** dict — maximum control/interoperability, returns a dict, manual validation.

Source (schema types + validation note): https://docs.langchain.com/oss/python/langchain/models#structured-output

**`method` parameter** (per the "Key considerations" note on the same page):
- `'json_schema'` — OpenAI's dedicated structured-output feature (constrained decoding). Preferred.
- `'function_calling'` — forces a tool call that follows the schema. Fallback for models/paths without native support.
- `'json_mode'` — older; emits valid JSON but the schema must be described in the prompt.

**`strict=True`** (OpenAI): the provider enforces the schema during generation, the highest-reliability mode.
Source: https://docs.langchain.com/oss/javascript/integrations/chat/openai#structured-output (documented on the OpenAI integration page; the Python `.with_structured_output` accepts the same `strict` flag).

### 1b. Via `create_agent` (only if you also need tools)

Source: https://docs.langchain.com/oss/python/langchain/structured-output

```python
from langchain.agents import create_agent
agent = create_agent(model="openai:gpt-5.5", response_format=CommentExtract)
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
result["structured_response"]  # CommentExtract(...)
```

Passing a schema type directly makes LangChain choose a strategy automatically:
- **`ProviderStrategy`** — used when the model supports native structured output (OpenAI, Anthropic, xAI, Gemini). "This is the most reliable method when available… the model provider enforces the schema."
- **`ToolStrategy`** — fallback via tool calling for models without native support.

Source (strategy selection): https://docs.langchain.com/oss/python/langchain/structured-output#provider-strategy

### Reliability, failure modes, and validation-error handling

- **Get the raw message alongside the parse** with `include_raw=True` — returns `{"raw": AIMessage, "parsed": ..., "parsing_error": ...}`. Useful both for token accounting and for detecting parse failures without exceptions.
  Source: https://docs.langchain.com/oss/python/langchain/models#structured-output (the "Message output alongside parsed structure" accordion).

- **Automatic retry on validation failure** (agent / `ToolStrategy` path): when the model returns output that fails Pydantic validation, LangChain feeds the validation error back as a `ToolMessage` and prompts a retry. Example from the docs: a model returns `rating=10` for a `Field(ge=1, le=5)`, the agent replies `Error: … Input should be less than or equal to 5 …`, and the model corrects to `rating=5`.
  Source: https://docs.langchain.com/oss/python/langchain/structured-output (the "Schema validation error" section).

- **`handle_errors` on `ToolStrategy`** controls this behavior:
  - `True` (default) — catch all, retry with default error template.
  - `str` — always retry with a fixed custom message.
  - `Exception` type / tuple — retry only on those, otherwise raise.
  - `Callable[[Exception], str]` — custom handler (branch on `StructuredOutputValidationError` vs `MultipleStructuredOutputsError`).
  - `False` — no retry, exceptions propagate.
  Source: https://docs.langchain.com/oss/python/langchain/structured-output (the "Error handling strategies" section).

**Failure modes to expect at scale:**
- **Multiple structured outputs** — with a `Union` schema the model may call several tools at once; the agent errors and prompts it to pick one (`MultipleStructuredOutputsError`). Source: same page, "Multiple structured outputs error".
- **Validation failures** — only Pydantic (and JSON Schema, if you validate) catch these; TypedDict/dataclass silently return whatever the model produced. Prefer Pydantic for extraction you rely on.
- **Direct `with_structured_output` has no built-in retry loop** — retries are an agent/`ToolStrategy` feature. For the direct model call, wrap with `include_raw=True` and check `parsing_error`, or add `.with_retry()` yourself.

---

## 2. Batching / map-reduce over a corpus larger than one context window

Reddit comments are independent records, so this is embarrassingly parallel for the *map* step; the hard part is the *reduce* step staying coherent.

### 2a. Chunk sizing (fit each unit into context)

Split with `RecursiveCharacterTextSplitter` (the recommended general splitter) — but note comments are natural units, so prefer **packing N whole comments per request** over splitting mid-comment.
Source: https://docs.langchain.com/oss/python/deepagents/rag#split-documents

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

### 2b. Map step — client-side parallel calls

`model.batch()` runs independent requests in parallel; `batch_as_completed()` yields each as it finishes (out of order, each carries its input index). Cap parallelism with `max_concurrency`.
Source: https://docs.langchain.com/oss/python/langchain/models#batch

```python
extractor = model.with_structured_output(CommentExtract, method="json_schema", strict=True)
results = extractor.batch(list_of_comment_batches, config={"max_concurrency": 5})
```

> Note from the docs: `batch()` is **client-side parallelization**, distinct from provider batch APIs (OpenAI's async Batch API at 50% discount). LangChain does not wrap the OpenAI Batch API here — if you want that discount you call it via the OpenAI SDK directly.
> Source: https://docs.langchain.com/oss/python/langchain/models#batch (the "distinct from batch APIs supported by inference providers" note).

### 2c. Fan-out / fan-in — LangGraph `Send` API + reducer

For an orchestrated map-reduce with aggregation, LangGraph provides the `Send` API to fan out one node per item and a reducer channel (`Annotated[list, operator.add]`) to gather results into shared state.
Source: https://docs.langchain.com/oss/python/langgraph/use-graph-api#map-reduce-and-the-send-api

```python
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.types import Send

class OverallState(TypedDict):
    comment_batches: list[str]
    extracts: Annotated[list[CommentExtract], operator.add]  # reducer accumulates across all map nodes
    final_themes: list

def fan_out(state): return [Send("extract", {"batch": b}) for b in state["comment_batches"]]
# "extract" node appends to state["extracts"]; a final "reduce" node canonicalizes.
```

### 2d. THE HARD PART — keeping theme clustering coherent across batches

**Honest finding: LangChain/LangGraph give you the map-reduce *plumbing* (Send + reducers, collapse patterns) but NOT cross-batch semantic coherence.** The docs' map-reduce and summarization examples all show *per-chunk output simply concatenated by a reducer* — nothing reconciles that "app crashes" in batch 1 and "keeps crashing" in batch 7 are the same theme. Naive per-batch extraction therefore produces exactly the **per-batch silos** the ticket warns about. Coherence is a design responsibility. Patterns that work, in increasing robustness:

1. **Two-pass extract-then-canonicalize (recommended).** Pass 1: per-chunk map extracts raw theme phrases (the `batch()` call above). Pass 2: a single *reduce* LLM call receives the *deduplicated union of all raw theme phrases* (not the comments — cheap) and returns a canonical theme taxonomy; then a third cheap pass maps each comment's raw themes onto canonical IDs. This is the LangChain-idiomatic realization of "collapse/reduce" and stays within one context window because you're reducing over *themes*, not *comments*.

2. **Shared taxonomy injected into the map prompt.** Pass a fixed/enumerated `Literal[...]` set of allowed theme labels into every map call so batches classify into the *same* buckets. Eliminates silos by construction but requires knowing the label set up front (good for a stable market-research rubric, bad for open discovery).

3. **Embed + cluster (non-LLM canonicalization).** Extract free-text themes per comment, embed them (`OpenAIEmbeddings`), cluster the vectors, then label clusters once. Most robust for open-ended discovery; the clustering itself is deterministic and batch-invariant. LangChain provides the embeddings adapter but not the clustering (use scikit-learn/HDBSCAN).
   Embeddings source: https://docs.langchain.com/oss/python/integrations/embeddings/openai

**Refine vs map-reduce:** LangChain's iterative "refine"-style pattern appears in the docs only as *conversation summarization* — extend a running summary with each new batch, using the prior summary as context.
Source: https://docs.langchain.com/oss/python/langgraph/add-memory#full-example-summarize-messages
This *does* preserve cross-batch coherence (each batch sees the running state) but is **sequential (no parallelism), order-dependent, and lets early batches bias the taxonomy** — acceptable for a narrative summary, poor for unbiased theme frequency counts. Prefer the two-pass approach (option 1) for market research.

---

## 3. Cost control — token counting, chunk sizing, minimizing calls

### Per-call and aggregate token usage
OpenAI returns token usage on each response; LangChain attaches it to `AIMessage.usage_metadata`. For totals across many calls, use the callback or context manager.
Source: https://docs.langchain.com/oss/python/langchain/models#token-usage

```python
from langchain_core.callbacks import get_usage_metadata_callback

with get_usage_metadata_callback() as cb:
    extractor.batch(list_of_comment_batches, config={"max_concurrency": 5})
    print(cb.usage_metadata)
# {'gpt-5.4-mini': {'input_tokens': ..., 'output_tokens': ..., 'total_tokens': ...,
#                   'input_token_details': {'cache_read': ...}, ...}}
```

`UsageMetadataCallbackHandler` is the equivalent handler you can attach via `config={"callbacks": [cb]}`. Both break down cache-read and reasoning tokens.
Source: same page, "Token usage" section.

> Caveat from the docs: for **streaming** with OpenAI you must opt in to receive usage data. Source: https://docs.langchain.com/oss/python/langchain/models#token-usage (note referencing "streaming usage metadata").

### Pre-flight token counting (for chunk packing / budgeting)
- `model.get_num_tokens(text)` — model-specific count (uses tiktoken for OpenAI). "Useful for checking if an input fits in a model's context window."
  Source: https://reference.langchain.com/python/langchain-core/language_models/base/BaseLanguageModel/get_num_tokens
- `count_tokens_approximately` (`langchain_core.messages.utils`) — fast char-based approximation for budgeting, used in the docs' summarization node.
  Source: https://docs.langchain.com/oss/python/langgraph/add-memory#full-example-summarize-messages

### Prompt caching
OpenAI supports **implicit** prompt caching (automatic cost savings on repeated prefixes, no config). Cache hits show up in `usage_metadata` under `input_token_details.cache_read`. Putting your long fixed extraction instructions/rubric as a stable prompt prefix lets many comment calls hit the cache.
Source: https://docs.langchain.com/oss/python/langchain/models (prompt caching section).

### Minimizing calls
- Pack multiple comments per request (respecting `get_num_tokens` budget) instead of one call per comment.
- Use a small model (`gpt-5.4-mini`-class) for the map step; reserve a larger model only for the reduce/canonicalize step.
- Reduce over *extracted themes*, not raw comments (Section 2d) — collapses reduce-step token volume by orders of magnitude.
- `max_tokens` caps output length per call. Source: https://docs.langchain.com/oss/python/langchain/models (the `max_tokens` param field).

### Where LangChain helps vs. obscures
- **Helps**: uniform `usage_metadata` shape across providers; one-line aggregate accounting via the callback/context manager; `get_num_tokens` with the right tokenizer selected per model.
- **Obscures**:
  - **No dollar cost.** LangChain reports tokens, never money — you maintain your own price table. (LangSmith adds cost tracking, but that's a separate paid product.)
  - **Hidden calls.** In the agent/`ToolStrategy` path, validation-retry loops and tool round-trips add calls/tokens you don't see unless you inspect `usage_metadata` or trace. The direct `with_structured_output` path is far more predictable.
  - **`strict`/`json_schema` overhead.** Constrained decoding can add latency and, with verbose schemas, input tokens (the schema is sent each call) — keep schemas flat and minimal.

---

## 4. Fit check — LangChain vs plain OpenAI SDK

**Verdict: Use a thin slice of LangChain, not the whole stack. For a pure high-volume extraction pipeline, LangChain's payoff is modest and mostly in adapters; its agent/graph machinery is overkill and adds token/cost opacity.**

### What genuinely pays off for this pipeline
- `with_structured_output(..., strict=True)` — the same OpenAI feature you'd call directly, but with Pydantic parsing/validation wired in and a clean `include_raw` escape hatch. Small, real convenience.
- `init_chat_model("openai:...")` — provider portability; swap OpenAI ↔ Anthropic without rewriting call sites. Worth it *only if* you actually expect to switch or A/B models.
- `batch()` / `batch_as_completed()` + `max_concurrency` — saves you writing an asyncio semaphore pool.
- `usage_metadata` + `get_usage_metadata_callback()` — uniform token accounting.

### What is overkill / net-negative here
- **`create_agent` / LangGraph** — designed for tool-using, multi-step, stateful agents. Per-comment extraction is stateless and single-shot; the graph adds concepts (state channels, reducers, Send) without payoff. Only reach for LangGraph if the *reduce/orchestration* genuinely needs checkpointing, retries, or fan-out you don't want to hand-roll.
- **Retry-via-ToolStrategy** adds invisible extra calls; for extraction, an explicit `.with_retry()` or a `parsing_error` check is more transparent.
- **Cost opacity** (Section 3) — agent loops make token spend harder to predict; a concern at Reddit-corpus scale.

### The honest tradeoff table

| Concern | LangChain | Plain OpenAI SDK |
|---|---|---|
| Structured output | `with_structured_output` + Pydantic validation, `strict` passthrough | `client.chat.completions.parse` / `response_format` — native, you add Pydantic yourself (minimal work) |
| Validation retry | Built-in on agent path | You write it (a few lines) |
| Parallel map | `batch()` + `max_concurrency` | asyncio + semaphore (you write it) |
| Provider portability | Strong (`init_chat_model`) | None (rewrite to switch) |
| Token accounting | Uniform `usage_metadata` + callback | Read `response.usage` per call, aggregate yourself |
| **OpenAI Batch API (50% off)** | **Not wrapped** — use SDK | **Direct access** |
| Cost ($) | Not provided | Not provided (both) |
| Cross-batch theme coherence | Not provided (you design it) | Not provided (you design it) |
| Overhead / indirection | Real; abstractions can hide token growth | Minimal; you see every call |

**Bottom line for the Reddit analyzer:** adopt `langchain-openai` for `ChatOpenAI.with_structured_output` + `batch` + `usage_metadata`, and **write the map-reduce/theme-canonicalization logic yourself** (Section 2d) rather than reaching for LangGraph. If provider portability is a non-goal and you want the OpenAI Batch API discount at corpus scale, the plain OpenAI SDK with `response_format` + your own Pydantic models is a defensible, lower-abstraction alternative — the reliability of the extraction comes from OpenAI's structured-output feature, which both paths use identically.

---

## Sources
- Structured output (agents / strategies / error handling): https://docs.langchain.com/oss/python/langchain/structured-output
- Structured output + token usage + batch (models): https://docs.langchain.com/oss/python/langchain/models
- Map-reduce via Send API: https://docs.langchain.com/oss/python/langgraph/use-graph-api#map-reduce-and-the-send-api
- Iterative summarize (refine-style): https://docs.langchain.com/oss/python/langgraph/add-memory#full-example-summarize-messages
- Text splitting: https://docs.langchain.com/oss/python/deepagents/rag#split-documents
- OpenAI structured output `strict`: https://docs.langchain.com/oss/javascript/integrations/chat/openai#structured-output
- OpenAI embeddings: https://docs.langchain.com/oss/python/integrations/embeddings/openai
- `get_num_tokens`: https://reference.langchain.com/python/langchain-core/language_models/base/BaseLanguageModel/get_num_tokens
