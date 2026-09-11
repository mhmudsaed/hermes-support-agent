"""Threshold / ticket flow: low confidence opens tickets, humans follow up."""

import os

import pytest

from support_agent.answer import NoAnswer


def test_confident_question_returns_answer(agent):
    result = agent.ask("How do I request a refund?")
    assert result.kind == "answer"
    assert result.ticket is None
    assert "[Source:" in result.text


def test_low_confidence_question_opens_ticket(agent):
    result = agent.ask("What is your quantum teleportation SLA?")
    assert result.kind == "ticket"
    assert result.ticket is not None
    assert result.ticket.status == "open"
    assert result.ticket.question == "What is your quantum teleportation SLA?"


def test_high_threshold_forces_ticket_even_for_known_question(agent, tmp_path):
    from tests.conftest import SAMPLE_DOCS, make_settings

    from support_agent.service import SupportAgent

    ag = SupportAgent(make_settings(tmp_path, confidence_threshold=0.9999))
    try:
        ag.ingest_dir(os.path.abspath(SAMPLE_DOCS))
        result = ag.ask("How do I request a refund?")
        assert result.kind == "ticket"
    finally:
        ag.store.close()


def test_ticket_holds_summary_and_candidates(agent):
    result = agent.ask("Explain the Mars office parking policy in detail?")
    assert result.kind == "ticket"
    ticket = result.ticket
    assert ticket.summary and ticket.question in ticket.summary
    assert isinstance(ticket.candidates, list)


def test_human_reply_is_logged_and_can_close(agent):
    opened = agent.ask("What is your quantum teleportation SLA?")
    ticket = opened.ticket
    updated = agent.reply_ticket(ticket.id, "We don't offer that; see billing.",
                                 close=True)
    assert updated.status == "answered"
    _, messages = agent.ticket_log(ticket.id)
    roles = [m["role"] for m in messages]
    assert roles[0] == "user"
    assert "human" in roles


def test_ticket_retry_after_new_docs_can_answer(tmp_path):
    from tests.conftest import SAMPLE_DOCS, make_settings

    from support_agent.service import SupportAgent

    ag = SupportAgent(make_settings(tmp_path))
    try:
        opened = ag.ask("How do I request a refund?")
        assert opened.kind == "ticket"  # empty KB: nothing to ground on
        ag.ingest_dir(os.path.abspath(SAMPLE_DOCS))
        retried = ag.try_answer_from_context(opened.ticket.id)
        assert retried.kind == "answer"
        _, messages = ag.ticket_log(opened.ticket.id)
        assert any(m["role"] == "agent" for m in messages)
        assert ag.store.get_ticket(opened.ticket.id).status == "answered"
    finally:
        ag.store.close()


def test_unknown_ticket_raises(agent):
    with pytest.raises(KeyError):
        agent.reply_ticket("t-doesnotexist", "hello")
    with pytest.raises(KeyError):
        agent.try_answer_from_context("t-doesnotexist")


def test_empty_question_rejected(agent):
    with pytest.raises((ValueError, NoAnswer)):
        agent.ask("   ")
