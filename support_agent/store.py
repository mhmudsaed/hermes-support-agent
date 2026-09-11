"""Storage backends: SQLite (offline) and Postgres+pgvector (live).

Both expose the same interface so the rest of the app is backend-agnostic:

- ``add_chunks`` / ``all_chunks`` / ``search_vectors`` for retrieval
- ``create_ticket`` / ``get_ticket`` / ``list_tickets`` / ``add_message`` /
  ``ticket_messages`` / ``set_ticket_status`` for the ticket flow
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field

from support_agent.embed import cosine


@dataclass
class StoredChunk:
    doc: str
    path: str
    section: str
    text: str
    lang: str
    embedding: list[float] = field(default_factory=list)
    id: int | None = None


@dataclass
class Ticket:
    id: str
    question: str
    summary: str
    contact: str
    status: str  # "open" | "answered" | "closed"
    candidates: list[dict]  # top candidate chunks (doc/section/snippet/score)
    created_at: float


def new_ticket_id() -> str:
    return "t-" + uuid.uuid4().hex[:8]


class SqliteStore:
    """Fully offline backend: SQLite + in-process cosine similarity."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS chunks(
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 doc TEXT NOT NULL, path TEXT NOT NULL, section TEXT NOT NULL,
                 text TEXT NOT NULL, lang TEXT NOT NULL, embedding TEXT NOT NULL)"""
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
        cur.execute(
            """CREATE TABLE IF NOT EXISTS tickets(
                 id TEXT PRIMARY KEY, question TEXT NOT NULL, summary TEXT NOT NULL,
                 contact TEXT NOT NULL, status TEXT NOT NULL,
                 candidates TEXT NOT NULL, created_at REAL NOT NULL)"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS messages(
                 id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT NOT NULL,
                 role TEXT NOT NULL, text TEXT NOT NULL, created_at REAL NOT NULL)"""
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_ticket ON messages(ticket_id)")
        self._conn.commit()

    # -- chunks ----------------------------------------------------------
    def add_chunks(self, chunks: list[StoredChunk]) -> int:
        cur = self._conn.cursor()
        paths = {c.path for c in chunks}
        for p in paths:  # re-ingest replaces that file's chunks
            cur.execute("DELETE FROM chunks WHERE path = ?", (p,))
        for c in chunks:
            cur.execute(
                "INSERT INTO chunks(doc, path, section, text, lang, embedding)"
                " VALUES(?,?,?,?,?,?)",
                (c.doc, c.path, c.section, c.text, c.lang,
                 json.dumps(c.embedding)))
        self._conn.commit()
        return len(chunks)

    def count_chunks(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def all_chunks(self) -> list[StoredChunk]:
        rows = self._conn.execute("SELECT * FROM chunks ORDER BY id").fetchall()
        return [self._row_chunk(r) for r in rows]

    @staticmethod
    def _row_chunk(r) -> StoredChunk:
        return StoredChunk(id=r["id"], doc=r["doc"], path=r["path"],
                           section=r["section"], text=r["text"], lang=r["lang"],
                           embedding=json.loads(r["embedding"]))

    def search_vectors(self, query_vec: list[float],
                       top_k: int) -> list[tuple[float, StoredChunk]]:
        scored = [(cosine(query_vec, c.embedding), c) for c in self.all_chunks()]
        scored.sort(key=lambda s: s[0], reverse=True)
        return scored[:top_k]

    # -- tickets ---------------------------------------------------------
    def create_ticket(self, question: str, summary: str, contact: str,
                      candidates: list[dict]) -> Ticket:
        ticket = Ticket(id=new_ticket_id(), question=question, summary=summary,
                        contact=contact or "", status="open",
                        candidates=candidates, created_at=time.time())
        self._conn.execute(
            "INSERT INTO tickets(id, question, summary, contact, status,"
            " candidates, created_at) VALUES(?,?,?,?,?,?,?)",
            (ticket.id, ticket.question, ticket.summary, ticket.contact,
             ticket.status, json.dumps(candidates, ensure_ascii=False),
             ticket.created_at))
        self._conn.execute(
            "INSERT INTO messages(ticket_id, role, text, created_at)"
            " VALUES(?,?,?,?)",
            (ticket.id, "user", question, ticket.created_at))
        self._conn.commit()
        return ticket

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        row = self._conn.execute(
            "SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if row is None:
            return None
        return Ticket(id=row["id"], question=row["question"],
                      summary=row["summary"], contact=row["contact"],
                      status=row["status"],
                      candidates=json.loads(row["candidates"]),
                      created_at=row["created_at"])

    def list_tickets(self, status: str = "") -> list[Ticket]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM tickets WHERE status = ? ORDER BY created_at DESC",
                (status,)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tickets ORDER BY created_at DESC").fetchall()
        return [Ticket(id=r["id"], question=r["question"], summary=r["summary"],
                       contact=r["contact"], status=r["status"],
                       candidates=json.loads(r["candidates"]),
                       created_at=r["created_at"]) for r in rows]

    def set_ticket_status(self, ticket_id: str, status: str) -> bool:
        cur = self._conn.execute(
            "UPDATE tickets SET status = ? WHERE id = ?", (status, ticket_id))
        self._conn.commit()
        return cur.rowcount > 0

    def add_message(self, ticket_id: str, role: str, text: str) -> None:
        if role not in ("user", "agent", "human"):
            raise ValueError(f"invalid role: {role}")
        if self.get_ticket(ticket_id) is None:
            raise KeyError(f"unknown ticket: {ticket_id}")
        self._conn.execute(
            "INSERT INTO messages(ticket_id, role, text, created_at)"
            " VALUES(?,?,?,?)", (ticket_id, role, text, time.time()))
        self._conn.commit()

    def ticket_messages(self, ticket_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT role, text, created_at FROM messages WHERE ticket_id = ?"
            " ORDER BY id", (ticket_id,)).fetchall()
        return [{"role": r["role"], "text": r["text"],
                 "created_at": r["created_at"]} for r in rows]

    def close(self) -> None:
        self._conn.close()


class PostgresStore:
    """Live backend: Postgres + pgvector. Requires ``psycopg`` + a running DB.

    Schema is created on first connect (``CREATE EXTENSION IF NOT EXISTS
    vector`` needs a superuser-or-granted role; docker-compose grants it).
    """

    def __init__(self, dsn: str, dim: int = 512):
        try:
            import psycopg  # type: ignore  # optional live dependency
        except ImportError as exc:
            raise RuntimeError(
                "Postgres backend needs the 'pg' extra: pip install -e '.[pg]'"
            ) from exc
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._dim = dim
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS chunks(
                      id SERIAL PRIMARY KEY, doc TEXT NOT NULL,
                      path TEXT NOT NULL, section TEXT NOT NULL,
                      text TEXT NOT NULL, lang TEXT NOT NULL,
                      embedding vector({dim}) NOT NULL)"""
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
            cur.execute(
                """CREATE TABLE IF NOT EXISTS tickets(
                     id TEXT PRIMARY KEY, question TEXT NOT NULL,
                     summary TEXT NOT NULL, contact TEXT NOT NULL,
                     status TEXT NOT NULL, candidates JSONB NOT NULL,
                     created_at DOUBLE PRECISION NOT NULL)"""
            )
            cur.execute(
                """CREATE TABLE IF NOT EXISTS messages(
                     id SERIAL PRIMARY KEY, ticket_id TEXT NOT NULL,
                     role TEXT NOT NULL, text TEXT NOT NULL,
                     created_at DOUBLE PRECISION NOT NULL)"""
            )

    def add_chunks(self, chunks: list[StoredChunk]) -> int:
        with self._conn.cursor() as cur:
            for p in {c.path for c in chunks}:
                cur.execute("DELETE FROM chunks WHERE path = %s", (p,))
            for c in chunks:
                cur.execute(
                    "INSERT INTO chunks(doc, path, section, text, lang, embedding)"
                    " VALUES(%s,%s,%s,%s,%s,%s)",
                    (c.doc, c.path, c.section, c.text, c.lang, c.embedding))
        return len(chunks)

    def count_chunks(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def all_chunks(self) -> list[StoredChunk]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, doc, path, section, text, lang, embedding::text"
                " FROM chunks ORDER BY id")
            return [StoredChunk(id=r[0], doc=r[1], path=r[2], section=r[3],
                                text=r[4], lang=r[5],
                                embedding=json.loads(r[6])) for r in cur.fetchall()]

    def search_vectors(self, query_vec: list[float],
                       top_k: int) -> list[tuple[float, StoredChunk]]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, doc, path, section, text, lang, embedding::text,"
                " 1 - (embedding <=> %s::vector) AS score"
                " FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
                (query_vec, query_vec, top_k))
            out: list[tuple[float, StoredChunk]] = []
            for r in cur.fetchall():
                chunk = StoredChunk(id=r[0], doc=r[1], path=r[2],
                                    section=r[3], text=r[4], lang=r[5],
                                    embedding=json.loads(r[6]))
                out.append((float(r[7]), chunk))
            return out

    def create_ticket(self, question: str, summary: str, contact: str,
                      candidates: list[dict]) -> Ticket:
        ticket = Ticket(id=new_ticket_id(), question=question, summary=summary,
                        contact=contact or "", status="open",
                        candidates=candidates, created_at=time.time())
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tickets(id, question, summary, contact, status,"
                " candidates, created_at) VALUES(%s,%s,%s,%s,%s,%s,%s)",
                (ticket.id, ticket.question, ticket.summary, ticket.contact,
                 ticket.status, json.dumps(candidates, ensure_ascii=False),
                 ticket.created_at))
            cur.execute(
                "INSERT INTO messages(ticket_id, role, text, created_at)"
                " VALUES(%s,%s,%s,%s)",
                (ticket.id, "user", question, ticket.created_at))
        return ticket

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT id, question, summary, contact, status,"
                        " candidates, created_at FROM tickets WHERE id = %s",
                        (ticket_id,))
            r = cur.fetchone()
        if r is None:
            return None
        cand = r[5] if isinstance(r[5], list) else json.loads(r[5])
        return Ticket(id=r[0], question=r[1], summary=r[2], contact=r[3],
                      status=r[4], candidates=cand, created_at=r[6])

    def list_tickets(self, status: str = "") -> list[Ticket]:
        with self._conn.cursor() as cur:
            if status:
                cur.execute(
                    "SELECT id, question, summary, contact, status, candidates,"
                    " created_at FROM tickets WHERE status = %s"
                    " ORDER BY created_at DESC", (status,))
            else:
                cur.execute(
                    "SELECT id, question, summary, contact, status, candidates,"
                    " created_at FROM tickets ORDER BY created_at DESC")
            out = []
            for r in cur.fetchall():
                cand = r[5] if isinstance(r[5], list) else json.loads(r[5])
                out.append(Ticket(id=r[0], question=r[1], summary=r[2],
                                  contact=r[3], status=r[4], candidates=cand,
                                  created_at=r[6]))
            return out

    def set_ticket_status(self, ticket_id: str, status: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute("UPDATE tickets SET status = %s WHERE id = %s",
                        (status, ticket_id))
            return cur.rowcount > 0

    def add_message(self, ticket_id: str, role: str, text: str) -> None:
        if role not in ("user", "agent", "human"):
            raise ValueError(f"invalid role: {role}")
        if self.get_ticket(ticket_id) is None:
            raise KeyError(f"unknown ticket: {ticket_id}")
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages(ticket_id, role, text, created_at)"
                " VALUES(%s,%s,%s,%s)", (ticket_id, role, text, time.time()))

    def ticket_messages(self, ticket_id: str) -> list[dict]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT role, text, created_at FROM messages WHERE ticket_id = %s"
                " ORDER BY id", (ticket_id,))
            return [{"role": r[0], "text": r[1], "created_at": r[2]}
                    for r in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


def open_store(settings) -> SqliteStore | PostgresStore:
    import os

    if settings.backend == "postgres":
        if not settings.database_url:
            raise RuntimeError(
                "SUPPORT_BACKEND=postgres needs DATABASE_URL (see .env.example)")
        return PostgresStore(settings.database_url,
                             dim=settings.embedding_dim
                             if settings.embedder == "openai" else 512
                             if settings.embedding_dim == 512
                             else settings.embedding_dim)
    if settings.backend != "sqlite":
        raise ValueError(f"unknown backend: {settings.backend}")
    parent = os.path.dirname(os.path.abspath(settings.db_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return SqliteStore(settings.db_path)
