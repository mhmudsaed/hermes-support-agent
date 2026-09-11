#!/usr/bin/env bash
# One-command offline demo: ingest sample docs, ask EN + AR, show a ticket.
set -euo pipefail
cd "$(dirname "$0")/.."

export SUPPORT_BACKEND=sqlite
export SUPPORT_DB_PATH="$PWD/data/demo.db"
rm -f "$SUPPORT_DB_PATH"

PY=./.venv/bin/python
if [ ! -x "$PY" ]; then PY=python3; fi

echo "=== 1. ingest sample_docs ==="
$PY -m support_agent ingest sample_docs

echo
echo "=== 2. ask (English) ==="
$PY -m support_agent ask "How do I request a refund?"

echo
echo "=== 3. ask (Arabic) ==="
$PY -m support_agent ask "كيف أطلب استرداد المبلغ؟"

echo
echo "=== 4. low-confidence question -> ticket ==="
$PY -m support_agent ask "What is your quantum teleportation SLA?"

echo
echo "=== 5. tickets list ==="
$PY -m support_agent tickets list
echo
echo "demo done (offline, SQLite + fake embeddings)"
