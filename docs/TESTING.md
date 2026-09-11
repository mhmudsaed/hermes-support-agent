# TESTING — hermes-support-agent

## What works WITHOUT credentials (offline)

Everything below runs on this machine with zero network and zero keys:
SQLite backend + deterministic hash embedder + fake Telegram transport.

```bash
# 1. environment
uv venv .venv -p 3.11
uv pip install --python .venv/bin/python -e '.[dev]'

# 2. full test suite (must be green)
.venv/bin/python -m pytest -q
# expected: 48 passed

# 3. one-command demo (ingest + EN ask + AR ask + ticket + list)
bash scripts/demo.sh

# 4. eval score on the fixed 10-question set
bash scripts/eval.sh
# expected: eval: 10/10 correctly cited (offline, hash embedder)
```

Manual CLI tour (after `ingest` once):

```bash
export SUPPORT_BACKEND=sqlite SUPPORT_DB_PATH=$PWD/data/support.db
.venv/bin/python -m support_agent ingest sample_docs
.venv/bin/python -m support_agent ask "When does a downgrade take effect?"
.venv/bin/python -m support_agent ask "كم مدة صلاحية رابط إعادة تعيين كلمة المرور؟"
.venv/bin/python -m support_agent ask "What is your quantum teleportation SLA?"  # -> ticket
.venv/bin/python -m support_agent tickets list
.venv/bin/python -m support_agent tickets show <ticket-id>
.venv/bin/python -m support_agent tickets reply <ticket-id> "Human answer here" --close
.venv/bin/python -m support_agent tickets retry <ticket-id>   # re-answer vs current KB
```

## What needs credentials (live)

### A. Postgres + pgvector backend

Why: the real vector store for production data sizes (SQLite is exact but
scans all rows per query — fine for thousands of chunks, not millions).

1. Install Docker: https://docs.docker.com/get-docker/
2. Start the DB (user `support`, password `support`, db `support_agent`,
   host port **5433** to avoid clashing with a local Postgres):
   ```bash
   docker compose up -d db
   docker compose exec db pg_isready -U support   # expect: accepting connections
   ```
3. Install the driver + point the app at it:
   ```bash
   uv pip install --python .venv/bin/python -e '.[pg]'
   export SUPPORT_BACKEND=postgres
   export DATABASE_URL=postgresql://support:support@localhost:5433/support_agent
   .venv/bin/python -m support_agent ingest sample_docs
   .venv/bin/python -m support_agent ask "How do I request a refund?"
   ```
4. Schema (`chunks`, `tickets`, `messages` tables + `vector` extension) is
   created automatically on first connect. Least privilege: the compose user
   owns only the `support_agent` database; for hosted Postgres, create a role
   with `GRANT ALL ON DATABASE support_agent` and nothing else.

### B. Telegram bot transport

Why: live Q&A inside Telegram; offline tests use `FakeTransport` instead.

1. Open [@BotFather](https://t.me/BotFather) in Telegram, send `/newbot`,
   follow the prompts (name + username ending in `bot`).
2. BotFather replies with a token like `123456:ABC-DEF...` — copy it.
3. Find your chat id: message your new bot once, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser and read
   `message.chat.id`.
4. Run:
   ```bash
   export TELEGRAM_BOT_TOKEN=<token from step 2>
   export TELEGRAM_CHAT_ID=<chat id from step 3>
   .venv/bin/python -m support_agent serve --once   # one poll batch, then exit
   .venv/bin/python -m support_agent serve          # long-poll loop
   ```
5. In Telegram: send a question → cited answer; send `/tickets` → open
   tickets; send `/start` → help. Rate limiting: 1 msg/sec default
   (`TelegramTransport(..., rate_limit_s=...)`).

### C. Real embeddings (OpenAI)

Why: better semantic recall than the offline hash embedder on paraphrased
questions. The eval's live-LLM number requires this.

1. Get a key at https://platform.openai.com/api-keys (billing enabled).
2. Run:
   ```bash
   export SUPPORT_EMBEDDER=openai
   export LLM_API_KEY=sk-...
   export LLM_MODEL=text-embedding-3-small   # optional override
   # NOTE: embeddings differ in dimension from the hash backend — use a fresh DB:
   export SUPPORT_DB_PATH=$PWD/data/live.db  # or fresh Postgres DB
   .venv/bin/python -m support_agent ingest sample_docs
   bash scripts/eval.sh   # honest live number goes to docs/STATUS.md
   ```
   Any OpenAI-compatible endpoint works by subclassing `OpenAIEmbedder` with a
   different `base_url` (documented in `support_agent/embed.py`).

### D. Optional LLM rephrasing

Why: smoother wording; answers stay grounded because the LLM only rewords the
extractive draft (citations re-attached if dropped).

1. Same key as (C).
2. Run: `export LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini LLM_API_KEY=sk-...`
3. Ask anything; compare with/without by unsetting `LLM_PROVIDER`.

### E. Gmail as a second transport (documented, not built)

Sketch (not implemented — see `docs/STATUS.md` next steps):

1. In Google Cloud Console: create project → enable Gmail API → OAuth consent
   screen (Internal) → create OAuth client (Desktop) → download
   `credentials.json`.
2. First run opens a browser to authorize `gmail.readonly` + `gmail.send`
   scopes; token cached as `token.json` (git-ignored).
3. Daemon shape mirrors `TelegramBot.serve_forever`: poll `users.messages.list`
   with `q=is:unread`, map each thread to a ticket (`reply_ticket` on human
   follow-ups), send answers via `users.messages.send`, label threads
   `support-agent/answered|ticket`.

## Environment variables (all of them)

See `.env.example` — every variable is commented there. Copy it with
`cp .env.example .env` and fill only the live section you need; offline
defaults work as-is.
