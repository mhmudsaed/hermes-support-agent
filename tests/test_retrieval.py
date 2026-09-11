"""Retrieval quality: fixed eval set, question/expected-source pairs (synthetic)."""

from __future__ import annotations

import json
import os

import pytest

from support_agent.answer import answer_question
from support_agent.retrieval import retrieve

EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_set.json")


def load_eval_set() -> list[dict]:
    with open(EVAL_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def citation_ok(answer, expected: dict) -> bool:
    """A citation counts if doc matches and section contains the expected key."""
    for cite in answer.citations:
        if expected["doc"].lower() not in cite.doc.lower():
            continue
        if expected["section_key"].lower() in cite.section.lower():
            return True
    return False


def test_eval_set_file_has_ten_bilingual_items():
    items = load_eval_set()
    assert len(items) == 10
    langs = {i["lang"] for i in items}
    assert langs == {"en", "ar"}
    for item in items:
        assert {"question", "doc", "section_key", "lang"} <= set(item)


def test_eval_retrieval_hits_expected_source(agent):
    """Top-k retrieval must surface the expected doc/section for each item."""
    items = load_eval_set()
    failures = []
    for item in items:
        hits = retrieve(agent.store, agent.embedder, item["question"],
                        top_k=agent.settings.top_k)
        ok = any(item["doc"].lower() in h.chunk.doc.lower()
                 and item["section_key"].lower() in h.chunk.section.lower()
                 for h in hits)
        if not ok:
            failures.append(item["question"])
    assert not failures, f"retrieval missed expected source for: {failures}"


def test_eval_answers_correctly_cited(agent):
    """End-to-end: >= 9/10 answers carry the expected doc+section citation."""
    items = load_eval_set()
    good, failures = 0, []
    for item in items:
        try:
            answer = answer_question(
                agent.store, agent.embedder, item["question"],
                top_k=agent.settings.top_k,
                threshold=agent.settings.confidence_threshold)
        except Exception as exc:  # noqa: BLE001 - eval must report, not crash
            failures.append(f"{item['question']} -> {type(exc).__name__}: {exc}")
            continue
        if citation_ok(answer, item):
            good += 1
        else:
            failures.append(
                f"{item['question']} -> {[c.label() for c in answer.citations]}")
    assert good >= 9, f"only {good}/10 correctly cited: {failures}"


@pytest.mark.parametrize("idx", range(10))
def test_eval_item_individually(agent, idx):
    """Per-item granularity so regressions point at the exact question."""
    item = load_eval_set()[idx]
    answer = answer_question(
        agent.store, agent.embedder, item["question"],
        top_k=agent.settings.top_k,
        threshold=agent.settings.confidence_threshold)
    assert citation_ok(answer, item), (
        f"Q: {item['question']} got {[c.label() for c in answer.citations]}")
