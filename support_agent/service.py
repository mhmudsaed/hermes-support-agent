"""Agent service: ingest + ask + ticket follow-up in one place (CLI + Telegram share it)."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

from support_agent.answer import NoAnswer, answer_question, compose_answer
from support_agent.chunking import chunk_file, detect_lang
from support_agent.embed import make_embedder
from support_agent.guardrails import redact_secrets, validate_message_text
from support_agent.llm import make_rephraser
from support_agent.store import StoredChunk, Ticket


@dataclass
class AskResult:
    kind: str  # "answer" | "ticket"
    text: str
    citations: list[dict]
    confidence: float
    ticket: Ticket | None = None


class SupportAgent:
    def __init__(self, settings, store=None, embedder=None, rephraser=None):
        from support_agent.store import open_store

        self.settings = settings
        self.store = store or open_store(settings)
        self.embedder = embedder or make_embedder(settings)
        self.rephraser = rephraser or make_rephraser(settings)

    # -- ingestion ------------------------------------------------------
    def ingest_path(self, path: str) -> int:
        chunks = chunk_file(path, self.settings.chunk_max_chars,
                            self.settings.chunk_overlap_chars)
        stored = [StoredChunk(doc=c.doc, path=os.path.abspath(path),
                              section=c.section, text=c.text, lang=c.lang,
                              embedding=self.embedder.embed(c.text))
                  for c in chunks]
        return self.store.add_chunks(stored)

    def ingest_dir(self, directory: str,
                   patterns: tuple[str, ...] = ("*.md", "*.txt")) -> dict:
        total, files = 0, 0
        for pattern in patterns:
            for path in sorted(glob.glob(os.path.join(directory, pattern))):
                total += self.ingest_path(path)
                files += 1
        return {"files": files, "chunks": total}

    # -- Q&A ------------------------------------------------------------
    def _ticket_candidates(self, hits) -> list[dict]:
        return [{"doc": h.chunk.doc, "section": h.chunk.section,
                 "snippet": h.chunk.text[:300], "score": round(h.score, 4)}
                for h in hits[: self.settings.top_k]]

    def _summarize(self, question: str, hits) -> str:
        lang = detect_lang(question)
        if not hits:
            head = "No passages retrieved." if lang == "en" \
                else "لم يتم العثور على مقاطع ذات صلة."
        else:
            top = hits[0]
            head = (f"Closest match: {top.chunk.doc}"
                    f" ({top.chunk.section or 'no section'}), "
                    f"score {top.score:.3f}.")
            if lang == "ar":
                head = (f"أقرب نتيجة: {top.chunk.doc}"
                        f" ({top.chunk.section or 'بدون قسم'})، "
                        f"الثقة {top.score:.3f}.")
        q = question.strip()
        short = q if len(q) <= 140 else q[:137] + "..."
        return f"{short}\n{head}"

    def ask(self, question: str, contact: str = "") -> AskResult:
        from support_agent.answer import Answer  # noqa: F401 (re-export clarity)

        try:
            answer = answer_question(
                self.store, self.embedder, question,
                top_k=self.settings.top_k,
                threshold=self.settings.confidence_threshold,
                rephraser=self.rephraser)
        except NoAnswer as exc:
            summary = self._summarize(question, exc.hits)
            ticket = self.store.create_ticket(
                question=question.strip(), summary=summary,
                contact=contact or self.settings.support_contact,
                candidates=self._ticket_candidates(exc.hits))
            lang = detect_lang(question)
            if lang == "ar":
                text = (f"عذراً، لست واثقاً بما يكفي للإجابة من الوثائق "
                        f"(الثقة أقل من الحد {self.settings.confidence_threshold}). "
                        f"فتحت تذكرة {ticket.id} وسيرد عليك فريق الدعم قريباً.")
            else:
                text = (f"Sorry — I'm not confident enough to answer from the docs "
                        f"(below threshold {self.settings.confidence_threshold}). "
                        f"I opened ticket {ticket.id}; a human will follow up.")
            return AskResult(kind="ticket", text=text, citations=[],
                             confidence=exc.hits[0].score if exc.hits else 0.0,
                             ticket=ticket)
        return AskResult(
            kind="answer",
            text=answer.text,
            citations=[{"doc": c.doc, "section": c.section,
                        "marker": c.marker()} for c in answer.citations],
            confidence=answer.confidence)

    # -- tickets ----------------------------------------------------------
    def list_tickets(self, status: str = "") -> list[Ticket]:
        return self.store.list_tickets(status)

    def reply_ticket(self, ticket_id: str, text: str,
                     close: bool = False) -> Ticket:
        """Record a human reply on a ticket (follow-up path)."""
        body = validate_message_text(text)
        ticket = self.store.get_ticket(ticket_id)
        if ticket is None:
            raise KeyError(f"unknown ticket: {ticket_id}")
        self.store.add_message(ticket_id, "human", body)
        if close:
            self.store.set_ticket_status(ticket_id, "answered")
        updated = self.store.get_ticket(ticket_id)
        assert updated is not None
        return updated

    def ticket_log(self, ticket_id: str) -> tuple[Ticket, list[dict]]:
        ticket = self.store.get_ticket(ticket_id)
        if ticket is None:
            raise KeyError(f"unknown ticket: {ticket_id}")
        messages = self.store.ticket_messages(ticket_id)
        # Never leak contact PII into the review log view raw.
        safe = [{**m, "text": redact_secrets(m["text"])} for m in messages]
        return ticket, safe

    def try_answer_from_context(self, ticket_id: str) -> AskResult:
        """Re-run the ticket's question through the current KB (follow-up)."""
        ticket = self.store.get_ticket(ticket_id)
        if ticket is None:
            raise KeyError(f"unknown ticket: {ticket_id}")
        try:
            answer = compose_answer(ticket.question,
                                    self._hits_for(ticket.question))
        except NoAnswer:
            return AskResult(kind="ticket", text=ticket.summary, citations=[],
                             confidence=0.0, ticket=ticket)
        self.store.add_message(ticket_id, "agent", answer.text)
        self.store.set_ticket_status(ticket_id, "answered")
        return AskResult(kind="answer", text=answer.text,
                         citations=[{"doc": c.doc, "section": c.section}
                                    for c in answer.citations],
                         confidence=answer.confidence, ticket=ticket)

    def _hits_for(self, question: str):
        from support_agent.retrieval import retrieve

        return retrieve(self.store, self.embedder, question,
                        top_k=self.settings.top_k, lang=detect_lang(question))
