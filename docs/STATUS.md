# STATUS — hermes-support-agent (living document)

Last updated: 2026-09-11. Decisions are recorded here per AGENT_RULES.md.

## Done

- Ingestion pipeline: `.md`/`.txt` (+ `.pdf` via optional `pypdf`) →
  section-aware chunks with `doc` + `section` labels, stored with embeddings.
- Storage: `SqliteStore` (offline, cosine in-process) + `PostgresStore`
  (pgvector) behind one interface; `docker-compose.yml` (pg16, port 5433).
- Retrieval: hybrid 0.7 cosine + 0.3 lexical overlap with Arabic
  normalization (alef/ta-marbuta/diacritics/ال-stripping, prefix term match)
  + same-language preference.
- Answering: extractive composer — sentences copied from chunks, every line
  carries `[Source: doc — section]`; `NoAnswer` (never an uncited reply) when
  ungroundable or below `SUPPORT_THRESHOLD` (default 0.12).
- Tickets: structured (question, summary, top candidates, contact) + full
  per-ticket conversation log (`user`/`agent`/`human`), reply/close/retry
  flows shared by CLI and Telegram.
- Bilingual EN/AR from day one: sample docs (3 EN + 1 EN/AR-md + 1 AR-txt),
  Arabic chunking path, Arabic eval items, Arabic ticket messages.
- Interfaces: CLI (`ingest`, `ask [--json]`, `tickets list|show|reply|retry`,
  `serve [--once]`) + Telegram (`TelegramTransport` live, `FakeTransport`
  offline, `/start` + `/tickets` commands).
- Guardrails: untrusted docs treated as data (sentence-level
  instruction-strip, EN+AR patterns), input validation, secret redaction in
  logs, no secrets in code (`.env.example` documents all 13 variables).
- Suite: 48 tests green offline (`pytest -q`); `scripts/demo.sh` one-command
  demo; `scripts/eval.sh` honest scorer.

## Measured (real output, 2026-09-11, offline)

- `pytest -q`: **48 passed** (no network, no credentials).
- `scripts/eval.sh`: **10/10 correctly cited** (hash embedder + SQLite,
  threshold 0.12; criterion from SPEC: ≥ 9/10).
- `scripts/demo.sh`: ingests 6 files → 15 chunks; EN + AR answers cited;
  off-topic question opens a ticket; `tickets list` shows it.

## Mocked / requires live run

- Postgres+pgvector path: implemented + composed, **not run** (no Docker
  daemon exercised in this build) — marked "requires live run".
- Real embeddings (`SUPPORT_EMBEDDER=openai`) + LLM rephrasing
  (`LLM_PROVIDER=openai`): implemented, **not run** (no API key) —
  eval-with-live-LLM number "requires live run".
- Telegram live transport: implemented, **not run** (no bot token) —
  covered by `FakeTransport` tests only. BotFather steps in docs/TESTING.md.
- PDF ingest: implemented via optional `pypdf` dependency, **not tested**
  (no PDF fixture shipped; `pypdf` not installed).

## Known limitations

- SQLite retrieval scans all chunk vectors per query (fine for ~10³ chunks,
  not for 10⁶ — use Postgres at scale).
- HashEmbedder is lexical-ish: heavy paraphrase with zero shared terms can
  miss; the lexical-overlap term (0.3) deliberately favors precision over
  recall so misses become tickets, not hallucinations.
- Answers come from the single best chunk (up to 3 sentences); multi-doc
  synthesis across chunks is intentionally out of scope.
- Arabic normalization is heuristic (no real stemmer); dialectal phrasing
  may underperform MSA.
- Telegram long-poll is single-process, no webhook mode.

## Next steps (out of scope for this build)

1. Gmail transport (OAuth poller mapping threads ↔ tickets; sketch in
   docs/TESTING.md §E).
2. Webhook mode + multi-admin routing for Telegram.
3. Threshold auto-tuning from ticket outcomes; eval trend tracking.
4. Live Postgres + live-LLM eval numbers filled in above once creds exist.
