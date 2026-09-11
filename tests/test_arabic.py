"""Arabic path: ingest, retrieve, answer, and ticket in Arabic."""

from support_agent.answer import answer_question
from support_agent.chunking import detect_lang
from support_agent.retrieval import retrieve


def test_arabic_docs_are_ingested(agent):
    langs = {c.lang for c in agent.store.all_chunks()}
    assert "ar" in langs and "en" in langs


def test_arabic_question_retrieves_arabic_source(agent):
    hits = retrieve(agent.store, agent.embedder, "كيف أطلب استرداد المبلغ؟",
                    top_k=3)
    assert hits
    assert hits[0].chunk.lang == "ar"


def test_arabic_answer_is_arabic_and_cited(agent):
    answer = answer_question(agent.store, agent.embedder,
                             "كيف أطلب استرداد المبلغ؟",
                             top_k=3, threshold=0.0)
    assert answer.citations
    assert detect_lang(answer.text) == "ar"
    assert "[Source:" in answer.text


def test_arabic_ticket_message_is_arabic(agent):
    result = agent.ask("ما هي سياسة وقوف السيارات في مكتب المريخ؟")
    assert result.kind == "ticket"
    assert "تذكرة" in result.text


def test_mixed_arabic_english_question_still_grounded(agent):
    answer = answer_question(agent.store, agent.embedder,
                             "refund كيف أطلب استرداد المبلغ؟",
                             top_k=3, threshold=0.0)
    assert answer.citations
