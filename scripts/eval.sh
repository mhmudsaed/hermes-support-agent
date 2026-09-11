#!/usr/bin/env bash
# Offline eval: run the fixed question/source set, print the honest score.
set -euo pipefail
cd "$(dirname "$0")/.."

export SUPPORT_BACKEND=sqlite
export SUPPORT_DB_PATH="$PWD/data/eval.db"
rm -f "$SUPPORT_DB_PATH"

PY=./.venv/bin/python
if [ ! -x "$PY" ]; then PY=python3; fi

$PY -m support_agent ingest sample_docs >/dev/null
$PY - "$@" <<'EOF'
import json, sys
sys.path.insert(0, ".")
from support_agent.answer import answer_question
from support_agent.config import Settings, load_settings
from support_agent.embed import make_embedder
from support_agent.store import open_store

settings = load_settings()
store = open_store(settings)
embedder = make_embedder(settings)
items = json.load(open("tests/eval_set.json", encoding="utf-8"))

good, rows = 0, []
for it in items:
    try:
        ans = answer_question(store, embedder, it["question"],
                              top_k=settings.top_k,
                              threshold=settings.confidence_threshold)
        labels = [f"{c.doc} — {c.section}" for c in ans.citations]
        ok = any(it["doc"].lower() in c.doc.lower()
                 and it["section_key"].lower() in c.section.lower()
                 for c in ans.citations)
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        labels, ok = [f"{type(exc).__name__}: {exc.args[0]}"], False
    good += ok
    rows.append((ok, it["question"][:60], " | ".join(labels)[:90]))

print(f"eval: {good}/{len(items)} correctly cited (offline, hash embedder)")
for ok, q, lab in rows:
    print(f"  [{'ok' if ok else 'MISS'}] {q}  ->  {lab}")
store.close()
sys.exit(0 if good >= 9 else 1)
EOF
