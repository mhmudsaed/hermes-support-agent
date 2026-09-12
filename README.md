# hermes-support-agent

A bilingual (Arabic + English) customer-support agent that answers questions
**only from your own docs** — every claim carries a `[Source: document —
section]` citation — and opens a **structured ticket** for a human whenever
it isn't confident enough to answer.

Built by Mahmoud Saeed as a portfolio project backing
[mahmoudsaeed.com/work/support-agent-rag](https://mahmoudsaeed.com/work/support-agent-rag).

## Demo

![Demo — a real offline run](docs/assets/demo.gif)

One command runs the whole pipeline offline with zero credentials: `./scripts/demo.sh`. The suite is also one command — `pytest -q` → **48 passed**, no network. The GIF above is a real recording of the demo (sped up; it opens with a title card); the [full-quality MP4](docs/assets/demo.mp4) is in this repo.

## Why this exists

Small teams answer the same ten questions over and over. A naive chatbot
either invents answers or dead-ends with "I don't understand". This one does
the two things that make a docs-bot trustworthy enough to leave running:

1. **It shows where every answer came from** — an answer without a document
   and section behind it is treated as a failure and never sent.
2. **It hands off cleanly** — below a confidence threshold it summarizes the
   question (plus the closest doc passages) into a ticket a human starts ahead
   on, and can re-answer the ticket once the docs improve.

## Architecture

```text
sample_docs/*.md|*.txt
        │ ingest (chunk: doc + section labels kept per chunk)
        ▼
embeddings (HashEmbedder offline · OpenAIEmbedder live)
        │ store
        ▼
SQLite + in-process cosine   ← offline default (tests, demo)
Postgres + pgvector          ← live backend (docker-compose.yml)
        │ ask
        ▼
retrieve top-k (0.7 cosine + 0.3 lexical overlap, Arabic-normalized)
        │ confidence >= threshold?
   yes  ▼                          no ▼
extractive answer               structured ticket
(sentences copied from         (summary + candidates
 chunks + [Source] markers)     + contact + full log)
        │                                │
        ▼                                ▼
CLI ask/ingest/tickets  ·  Telegram bot (FakeTransport offline)
```

Key files:

- `support_agent/chunking.py` — section-aware Markdown/TXT chunking (Arabic OK)
- `support_agent/embed.py` — pluggable embedders + Arabic token normalization
- `support_agent/store.py` — `SqliteStore` / `PostgresStore` (same interface)
- `support_agent/answer.py` — extractive composer + citation enforcement
- `support_agent/service.py` — `SupportAgent`: ingest/ask/tickets in one place
- `support_agent/telegram.py` — Telegram transport (live + fake)
- `support_agent/llm.py` — optional live-LLM rephraser (offline default)
- `support_agent/guardrails.py` — prompt-injection stripping, input validation

## Install

Requires Python 3.11+.

```bash
uv venv .venv -p 3.11
uv pip install --python .venv/bin/python -e '.[dev]'   # dev: + pytest
# live Postgres backend only:
uv pip install --python .venv/bin/python -e '.[pg]'
```

## Quickstart (offline demo, no credentials)

```bash
bash scripts/demo.sh
```

This ingests `sample_docs/`, asks one English + one Arabic question, forces a
low-confidence ticket, and lists tickets — all on SQLite with the offline
hash embedder. Also try:

```bash
./.venv/bin/python -m support_agent ingest sample_docs
./.venv/bin/python -m support_agent ask "How do I request a refund?"
./.venv/bin/python -m support_agent ask "كيف أطلب استرداد المبلغ؟"
./.venv/bin/python -m support_agent tickets list
```

## Configuration

All settings come from the environment; see `.env.example` (copy to `.env`):

| Variable | Default | Meaning |
|---|---|---|
| `SUPPORT_BACKEND` | `sqlite` | `sqlite` (offline) or `postgres` (live) |
| `SUPPORT_DB_PATH` | `data/support.db` | SQLite file path |
| `DATABASE_URL` | — | Postgres DSN for the live backend |
| `SUPPORT_TOP_K` | `3` | chunks retrieved per question |
| `SUPPORT_THRESHOLD` | `0.12` | min top-hit score to auto-answer; else ticket |
| `SUPPORT_CHUNK_MAX_CHARS` / `SUPPORT_CHUNK_OVERLAP` | `1200` / `150` | chunking sizes |
| `SUPPORT_EMBEDDER` | `hash` | `hash` (offline) or `openai` (live) |
| `SUPPORT_EMBEDDING_DIM` | `512` | hash-embedder dimension |
| `LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_KEY` | — | optional live rephrasing (`openai`) |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | — | Telegram transport (empty = offline fake) |
| `SUPPORT_CONTACT` | — | default contact stored on tickets |

## Live mode

Needs real credentials — full step-by-step in `docs/TESTING.md`:

- **Postgres + pgvector:** `docker compose up -d db`, then
  `DATABASE_URL=postgresql://support:support@localhost:5433/support_agent SUPPORT_BACKEND=postgres ...`
- **Telegram bot:** token from [@BotFather](https://t.me/BotFather), then
  `TELEGRAM_BOT_TOKEN=... python -m support_agent serve`
- **Real embeddings / LLM rephrasing:** `SUPPORT_EMBEDDER=openai` /
  `LLM_PROVIDER=openai` with `LLM_API_KEY=...` (see `.env.example`)
- **Gmail as a second transport:** documented as future work in
  `docs/STATUS.md` (interface sketch in `docs/TESTING.md`).

## Evaluation

Fixed 10-question EN/AR set in `tests/eval_set.json`:

```bash
bash scripts/eval.sh        # offline score (hash embedder + SQLite)
.venv/bin/python -m pytest -q
```

Latest measured result: **10/10 correctly cited (offline)** — see
`docs/STATUS.md`. Live-LLM numbers are marked "requires live run".

## Docs

- `docs/TESTING.md` — exact test steps + every live credential, step by step
- `docs/STATUS.md` — living status: done / mocked / limitations / next steps
- `docs/SPEC.md` — the build spec this repo implements
