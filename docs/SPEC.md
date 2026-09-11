# SPEC — Support Agent (hermes-support-agent)

Owner: Mahmoud Saeed · Built by: autonomous build agent · Backs: mahmoudsaeed.com/work/support-agent-rag

## Mission

A support agent that answers customer questions from a company's own documentation —
bilingual (Arabic + English), with a citation for every claim, and a clean hand-off to a
human when it is not confident. It must be trustworthy enough to leave running.

## Problem

Small teams answer the same ten questions over and over. The usual chatbot either invents
answers or dead-ends with "I don't understand". The useful version does two things a naive
bot doesn't: it shows where every answer came from, and it hands off cleanly to a human
when it isn't sure.

## Scope (what to build)

1. **Ingestion pipeline**: load documents (Markdown/TXT first; PDF optional if cheap),
   chunk them with section awareness (keep document + section labels per chunk),
   embed and store in Postgres + pgvector. Provide a second storage backend (SQLite +
   in-process cosine) that works fully offline for tests and demos — selected by config.
2. **Retrieval + answering**: given a question, retrieve top-k chunks, and answer ONLY
   from those chunks, citing document + section for every claim. An answer without a
   source is a failure and must not be sent.
3. **Confidence gate**: below a configurable threshold, do NOT answer — create a
   structured ticket (question summary, top candidate chunks, contact info) instead,
   and follow up when a human replies.
4. **Bilingual**: Arabic and English both work from day one; chunking and the evaluation
   set must cover Arabic.
5. **Interfaces**: a CLI (`ask`, `ingest`, `tickets` subcommands at minimum) plus a
   Telegram adapter for live use behind an interface with a fake for tests. Gmail
   optional as a second transport (document, don't block on it).
6. **Evaluation set**: a fixed set of question/expected-source pairs (ship in repo,
   synthetic) used to measure "≥ 9/10 answers correctly cited" honestly offline with the
   fake or local embedder; report the real measured number in docs/STATUS.md, mark
   live-LLM numbers as "requires live run" if not measured.

## Hard design decisions (carry these through)

- **Every answer carries its source** — an answer without a document and section behind
  it is treated as a failure, not sent.
- **Arabic and English from the start** — not a later translation layer.
- **A ticket instead of a dead end** — when confidence is low, the agent's job is
  summarizing the question well enough that the human starts ahead.

## Security & guardrails

- Untrusted document content is treated as data, never as instructions (prompt-injection
  safe: strip/skip instruction-looking content from answers; cite, don't obey).
- A confidence threshold gates every auto-answer.
- Full conversation logs kept per ticket for review.
- No secrets in code/logs; .env.example documents everything.

## Stack

Python 3.11+, Postgres + pgvector (docker-compose provided) with a SQLite offline
fallback, Telegram adapter (fake in tests), pluggable embedding backend (local fake +
documented real options). Keep dependencies small and pinned.

## Testing requirements

- All tests run offline: fake embedder, SQLite backend, fake Telegram transport.
- Include: chunking unit tests, retrieval quality test against the eval set, citation
  enforcement test (no source → rejected), threshold/ticket-flow tests, Arabic path test.
- docker-compose.yml for the live Postgres+pgvector path; document it in README.

## Definition of Done (checklist)

- [ ] `pytest -q` fully green offline, no credentials needed.
- [ ] `python -m support_agent ingest` + `ask` demo runs offline against sample docs in
      the repo (one-command: scripts/demo.sh).
- [ ] Eval set measured and reported honestly (docs/STATUS.md).
- [ ] README.md + docs/TESTING.md + docs/STATUS.md + .env.example complete.
- [ ] docker-compose.yml for live Postgres+pgvector path.
- [ ] All work committed; git status clean.

## Credentials for live mode (expand into docs/TESTING.md)

- LLM API key (provider + model configurable; document exact env vars).
- Telegram bot token + chat id (optional transport; BotFather steps).
- Postgres/pgvector for the real backend (docker compose documented; or connection URL).
- Gmail (optional; document only).
