# reddit-research

CLI tool for personal, non-commercial market research on Reddit. Give it a topic
and an explicit list of subreddits; it fetches the matching discussion, runs LLM
extraction over a frozen cache, and produces a structured JSON artifact plus a
Markdown report.

See [`docs/SPEC.md`](docs/SPEC.md) for the locked specification.

## Setup

```bash
uv sync                       # install dependencies
cp .env.example .env          # then fill in your secrets
```

## Usage

```bash
reddit-research fetch   --topic "Notion" --subreddits r/productivity r/Notion
reddit-research analyze --topic "Notion" --subreddits r/productivity r/Notion --use-cached
reddit-research cache purge --older-than 30
```
