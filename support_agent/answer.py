"""Answering: extractive composition with citation enforcement.

The answer is built ONLY from retrieved chunks: relevant sentences are copied
out and each carries a ``[Source: doc — section]`` marker. Composition never
invents facts; a confidence gate decides answer-vs-ticket.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from support_agent.chunking import detect_lang
from support_agent.embed import Embedder, norm_token, norm_tokens, term_match
from support_agent.guardrails import (
    split_sentences,
    strip_instructions,
    validate_question,
)
from support_agent.retrieval import Hit, retrieve

_STOPWORDS = frozenset(
    norm_token(w) for w in """
the a an and or of to in on for with is are was were be been do does did how
what when where which who whom why can could should would will your you our we
it its this that these those from at as by ما هو هي في من على إلى أن إن كان
كانت هل كيف متى أين لماذا كم مع هذا هذه ذلك التي الذي وما ولا قد كل عن غير بين
""".split())


def _keywords(text: str) -> list[str]:
    return [t for t in norm_tokens(text) if t not in _STOPWORDS and len(t) > 1]


@dataclass
class Citation:
    doc: str
    section: str

    def label(self) -> str:
        return f"{self.doc} — {self.section}" if self.section else self.doc

    def marker(self) -> str:
        return f"[Source: {self.label()}]"


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    confidence: float
    hits: list[Hit] = field(default_factory=list, repr=False)
    ticket_id: str | None = None  # set when gated to a ticket instead


@dataclass
class NoAnswer(Exception):
    """Raised when no grounded answer can be produced (caller opens a ticket)."""

    reason: str
    hits: list[Hit]


def _candidate_sentences(chunk_text: str) -> list[str]:
    clean = strip_instructions(chunk_text)
    return split_sentences(clean)


def _sentence_score(sent: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    toks = norm_tokens(sent)
    hit = sum(1 for kw in keywords
              if any(term_match(kw, st) for st in toks))
    if hit == 0:
        return 0.0
    # Coverage of question keywords, discounted for very long sentences.
    return hit / len(keywords) * (1.0 / (1.0 + max(0, len(toks) - 40) / 40))


def compose_answer(question: str, hits: list[Hit]) -> Answer:
    """Build a cited answer from hits. Raises NoAnswer if nothing grounds it."""
    q = validate_question(question)
    lang = detect_lang(q)
    keywords = _keywords(q)
    if not hits:
        raise NoAnswer("no documents retrieved", hits)

    picked: list[tuple[float, str, Hit]] = []
    for hit in hits:
        for sent in _candidate_sentences(hit.chunk.text):
            score = _sentence_score(sent, keywords)
            if score > 0:
                picked.append((score + 0.1 * hit.score, sent, hit))
    if not picked:
        raise NoAnswer("no retrieved passage mentions the question's terms", hits)

    picked.sort(key=lambda p: p[0], reverse=True)
    # Keep up to 3 sentences, all from the single best-grounding chunk: the
    # top hit by retrieval score that actually contains question terms.
    # (Mixing chunks with shared vocabulary, e.g. "refund" vs "recovery
    # codes", produced off-topic third sentences in the demo.)
    best_chunk_key = None
    for _, _, hit in picked:
        best_chunk_key = (hit.chunk.id if hit.chunk.id is not None
                          else id(hit.chunk))
        break
    chosen: list[tuple[str, Hit]] = []
    for _, sent, hit in picked:
        key = hit.chunk.id if hit.chunk.id is not None else id(hit.chunk)
        if key != best_chunk_key:
            continue
        if sent in {s for s, _ in chosen}:
            continue
        chosen.append((sent, hit))
        if len(chosen) >= 3:
            break

    citations: list[Citation] = []
    lines: list[str] = []
    for sent, hit in chosen:
        cite = Citation(doc=hit.chunk.doc, section=hit.chunk.section)
        if cite not in citations:
            citations.append(cite)
        lines.append(f"{sent} {cite.marker()}")
    if not citations:  # defence in depth: never emit an uncited answer
        raise NoAnswer("no citable source found", hits)
    confidence = max(h.score for h in hits)
    return Answer(text="\n".join(lines), citations=citations,
                  confidence=confidence, hits=hits)


def answer_question(store, embedder: Embedder, question: str, top_k: int = 3,
                    threshold: float = 0.12, rephraser=None) -> Answer:
    """Full pipeline: retrieve → gate → compose. Raises NoAnswer when gated."""
    from support_agent.llm import NoopRephraser  # local import: no cycle

    q = validate_question(question)
    lang = detect_lang(q)
    hits = retrieve(store, embedder, q, top_k=top_k, lang=lang)
    if not hits:
        raise NoAnswer("knowledge base is empty", hits)
    confidence = hits[0].score
    if confidence < threshold:
        raise NoAnswer(
            f"confidence {confidence:.3f} below threshold {threshold:.3f}", hits)
    answer = compose_answer(q, hits)
    answer.confidence = confidence
    if rephraser is None:
        rephraser = NoopRephraser()
    redrafted = rephraser.rephrase(answer.text, q, lang)
    # Re-attach citation markers if the rephraser dropped any (live-LLM path).
    for cite in answer.citations:
        if cite.marker() not in redrafted:
            redrafted = redrafted.rstrip() + f"\n{cite.marker()}"
    answer.text = redrafted
    return answer
