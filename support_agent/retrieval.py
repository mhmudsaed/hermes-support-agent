"""Retrieval: top-k chunks scored by hybrid cosine + lexical overlap."""

from __future__ import annotations

from dataclasses import dataclass

from support_agent.embed import Embedder, lexical_overlap
from support_agent.store import StoredChunk


@dataclass
class Hit:
    score: float
    cosine: float
    overlap: float
    chunk: StoredChunk


def retrieve(store, embedder: Embedder, query: str, top_k: int = 3,
             lang: str = "") -> list[Hit]:
    query_vec = embedder.embed(query)
    scored = store.search_vectors(query_vec, max(top_k * 4, top_k))
    hits: list[Hit] = []
    for cos, chunk in scored:
        overlap = lexical_overlap(query, chunk.text)
        # Hybrid: embedding similarity leads, exact-term overlap breaks ties.
        # Same-language chunks get a small preference (bilingual corpus).
        score = 0.7 * cos + 0.3 * overlap
        if lang and chunk.lang == lang:
            score += 0.02
        hits.append(Hit(score=score, cosine=cos, overlap=overlap, chunk=chunk))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]
