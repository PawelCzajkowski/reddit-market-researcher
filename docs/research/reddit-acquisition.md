# Research: Reddit Data Acquisition (R1, #2)

**Question:** How do we fetch Reddit discussion for a topic within a chosen set of subreddits, and what constraints govern it?

**Sourcing caveat (important):** Reddit's own policy/help pages (`reddit.com`, `redditinc.com`, `support.reddithelp.com`) **block automated fetch (403)**, so the two background research agents assigned to this ticket both died at the "read the terms" step. This doc was compiled inline from: (a) **PRAW official docs** (`praw.readthedocs.io`) — authoritative for API *mechanics*; and (b) **reputable secondary sources** for rate limits and ToS tiers. **Before build/ship, a human should open the actual [Data API Terms](https://www.redditinc.com/policies/data-api-terms) and Responsible Builder Policy in a browser and confirm the ToS section below** — especially the commercial-use classification, which genuinely affects cost/legality.

---

## Recommendation (summary)

- **Use PRAW in read-only mode** (a "script" app) against the **official Reddit Data API**. It handles OAuth, rate-limit headers + auto-sleep, pagination, and comment-tree expansion — no reason to hand-roll HTTP or scrape.
- **Fetch pattern:** `subreddit.search(topic, sort="top", time_filter="year")` per subreddit, then **filter to the 90-day window client-side** (the API's `time_filter` has no 90-day granularity — see §4). Cap at the default ~25 posts/subreddit.
- **Comments:** `submission.comments.replace_more(limit=0)` by default (drop "load more" rather than pay for it), plus a **score/depth cutoff**, to bound cost. Expanding *all* comments (`limit=None`) can blow the rate budget on big threads.
- **Treat the disk cache as ephemeral** — honor deletions, short TTL — not a permanent dataset (see §6).
- ⚠️ **Classify commercial vs. personal use before any non-personal deployment.** The free tier clearly covers personal/research use; "market research" used commercially may require Reddit's paid/approved tier. This is a human decision, flagged for the spec.

---

## 1. API options

- **Official Reddit Data API** is the only sanctioned route. **PRAW (Python Reddit API Wrapper)** is the recommended client — it wraps OAuth, respects `X-Ratelimit` headers and sleeps automatically, paginates `ListingGenerator`s, and expands comment `MoreComments`. → **Recommended.**
- **Alternatives, and why not:**
  - Raw HTTP to `oauth.reddit.com` — works but re-implements everything PRAW already does; no upside for a Python app.
  - **Pushshift** — historically the go-to for bulk historical Reddit data, but since the 2023 API lockdown it is **restricted to subreddit moderators** and not available for general use.
  - HTML scraping — violates Reddit's terms; avoid.

Source: [PRAW Quick Start](https://praw.readthedocs.io/en/stable/getting_started/quick_start.html).

## 2. Auth / app registration

1. Register an app at `https://www.reddit.com/prefs/apps` → app type **"script"** (single-user/personal).
2. You receive a **client_id** and **client_secret**.
3. For our use (reading public posts/comments) **read-only mode is sufficient** — instantiate PRAW with just `client_id`, `client_secret`, and `user_agent`; `reddit.read_only` is then `True`. Username/password are only needed for actions taken *as* a user (voting, posting), which we don't do.
4. **user_agent** must be unique and descriptive: `<platform>:<app id>:<version> (by u/<username>)`, e.g. `python:reddit-market-researcher:v0.1 (by u/yourname)`. Reddit throttles/blocks generic or missing user agents.

```python
import praw
reddit = praw.Reddit(
    client_id="...",
    client_secret="...",
    user_agent="python:reddit-market-researcher:v0.1 (by u/yourname)",
)
assert reddit.read_only  # True
```

Source: [PRAW Quick Start](https://praw.readthedocs.io/en/stable/getting_started/quick_start.html); credential steps per [OAuth2 Quick Start](https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps).

## 3. Rate limits

- **OAuth clients: 100 queries/minute (QPM)** per **client_id**, **averaged over a 10-minute window** (bursts allowed → ~1,000 requests / 10 min). Non-OAuth: **10 QPM**.
- Reddit reports live budget in **`X-Ratelimit-Remaining` / `X-Ratelimit-Reset`** headers. **PRAW respects these and auto-sleeps**; its `ratelimit_seconds` (default **5s**) is the threshold above which it raises `RedditAPIException` instead of sleeping.
- **Budget for a default run** (3 subreddits × 25 posts = 75 posts):
  - Listing/search: `search`/`top` return up to 100 items/request → **~1 request per subreddit** for 25 posts (a few more with pagination) ⇒ ~3–6 requests.
  - Comments: **≥1 request per submission** for the tree, **plus** each `replace_more()` expansion is additional requests (≈`ceil(expanded/100)`). With `replace_more(limit=0)` it's **~1 req/post ⇒ ~75 requests** total. Comfortably under 1,000/10-min.
  - ⚠️ `replace_more(limit=None)` on large threads can add **hundreds** of requests each — this is the main way to blow the budget. Hence the depth/score cutoff (see §5, and D3).

Sources: PRAW [Ratelimits](https://praw.readthedocs.io/en/stable/getting_started/ratelimits.html); rate-limit numbers corroborated by [SocialCrawl: Reddit Data API 2026](https://www.socialcrawl.dev/blog/reddit-data-api-2026).

## 4. Search within subreddits

`subreddit.search()` does exactly what we need — a keyword query scoped to one subreddit:

```python
reddit.subreddit("productivity").search(
    "notion", sort="top", time_filter="year", syntax="lucene", limit=25
)
```

- **query** (str) — the topic keyword.
- **sort** — `relevance` (default) | `hot` | `top` | `new` | `comments`.
- **time_filter** — `all` | `year` | `month` | `week` | `day` | `hour`.
- **syntax** — `lucene` (default) | `cloudsearch` | `plain`.

⚠️ **90-day window has no direct mapping** — `time_filter` granularity jumps `month → year`. **Recommended:** query `time_filter="year"` sorted by `top`, then **filter client-side** on `submission.created_utc >= now − 90d`. (Finalize this mapping in **D3**.) `subreddit.top(time_filter=...)` / `.hot()` / `.new()` are available for non-keyword sampling.

Source: [PRAW Subreddit.search](https://praw.readthedocs.io/en/stable/code_overview/models/subreddit.html).

## 5. Comment retrieval

- `submission.comments` returns a **`CommentForest`** (top-level comments, each with a reply forest).
- **`replace_more(limit=...)`** handles the "load more comments" (`MoreComments`) placeholders:
  - `limit=0` → **remove** them without fetching (cheapest);
  - `limit=32` (default) → expand up to 32 via API calls;
  - `limit=None` → expand everything (expensive).
- **`.list()`** flattens the tree breadth-first for easy iteration.
- `threshold` skips small `MoreComments` groups; **`submission.num_comments` may exceed** what you retrieve due to deleted/removed content.

```python
submission.comments.replace_more(limit=0)          # drop "load more"
for comment in submission.comments.list():          # flattened
    if comment.score >= MIN_SCORE:                  # our cutoff
        ...
```

**For our defaults:** `replace_more(limit=0)` + a score/depth cutoff keeps cost and noise down while retaining the high-signal comments market research cares about. (D3 sets the exact cutoffs.)

Source: [PRAW Comment Extraction tutorial](https://praw.readthedocs.io/en/stable/tutorials/comments.html).

## 6. ToS / legal (⚠️ verify against primary source before build)

Compiled from secondary sources ([SocialCrawl 2026](https://www.socialcrawl.dev/blog/reddit-data-api-2026); [TechCrunch, 2024 policy lockdown](https://techcrunch.com/2024/05/09/reddit-locks-down-its-public-data-in-new-content-policy-says-use-now-requires-a-contract/)) because Reddit's policy pages block automated fetch. **Re-confirm in a browser.**

- **Tiers:**
  - **Free / self-serve** — personal projects, bots, mod tools, **research**. No review; rate-limited to 100 QPM.
  - **Commercial** — ~**$0.24 / 1K calls**, manual approval (weeks). Reddit lists **brand monitoring, competitor tracking, lead generation, reselling** as commercial.
  - **Data licensing** — negotiated contract; required for **AI/ML training** and large-scale ingestion.
- ⚠️ **Commercial-use flag for THIS project:** a *market-research* analyzer used for business purposes may fall under **commercial use** → paid/approved tier. Personal/internal research use is fine on free. **This is a human classification decision — surface it in the spec (S1).**
- **Training vs. inference (key distinction in our favor):** Reddit prohibits using content to **train** ML/AI models without a licensing deal — that's why the Google/OpenAI deals exist. **Our app does *inference/analysis*** (sending comments to an LLM to summarize/extract), **not training**, which appears permissible under the free tier for non-commercial use. ⚠️ Two follow-ups: (a) sending Reddit content to a third-party (OpenAI) is data sharing — ensure OpenAI's **API data is not used for training** (OpenAI's API default is no-training, but confirm); (b) confirm Reddit doesn't separately restrict third-party processing.
- **Storage / retention:** caching to reduce API calls is *encouraged*, **but you must honor deletions** — retaining content a user later deleted (even anonymized) violates policy; secondary sources cite a **~48h routine-deletion** recommendation. **Implication:** treat our disk cache as an **ephemeral working cache with a short TTL / re-fetch**, not a permanent corpus. (Cache identity/expiry design → D3.)

---

## Handoffs to design tickets

- **D3 (fetch-and-cache stage):** finalize the 90-day window mapping (§4), the `replace_more`/score/depth cutoffs (§5), rate-limit pacing via `X-Ratelimit` (§3), and cache TTL/deletion-honoring (§6).
- **S1 (spec assembly):** record the **commercial-vs-personal use decision** and the **OpenAI no-training confirmation** as explicit spec notes (§6).
