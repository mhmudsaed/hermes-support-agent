"""Guardrails: untrusted document/message content is data, never instructions.

The composer is extractive (it copies relevant sentences, never follows them),
and this module additionally strips instruction-looking lines before any text
reaches the answer, plus validates inputs at the boundaries.
"""

from __future__ import annotations

import re

# Lines that look like prompt-injection attempts are dropped from answer
# source text. Patterns cover both English and Arabic.
_INSTRUCTION_PATTERNS = [
    r"^\s*(system|assistant|user|developer)\s*:",          # fake chat roles
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above|your)",
    r"you\s+are\s+now\s+",
    r"forget\s+(your|all|everything)",
    r"reveal\s+(your|the)\s+(system\s+)?(prompt|instructions|keys|password)",
    r"send\s+.*\s+to\s+\S+@\S+",                            # exfiltration
    r"delete\s+(all|everything|the\s+database)",
    r"تجاهل\s+.*(التعليمات|تعليمات)",                       # Arabic: ignore instructions
    r"انس[َ]?ى?\s+.*(التعليمات|تعليمات)",                   # Arabic: forget instructions
    r"اكشف\s+.*(النظام|كلمة|سر)",                           # Arabic: reveal system/secret
]
_INSTRUCTION_RES = [re.compile(p, re.IGNORECASE) for p in _INSTRUCTION_PATTERNS]


_SENT_SPLIT = re.compile(r"(?<=[.!?؟])\s+")


def split_sentences(text: str) -> list[str]:
    """Split into sentences, joining soft line-wraps first.

    Markdown sources wrap lines mid-sentence; a lone newline is a space, not
    a sentence boundary. Only sentence-ending punctuation splits.
    """
    joined = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return [s.strip() for s in _SENT_SPLIT.split(joined) if len(s.strip()) > 20]


def strip_instructions(text: str) -> str:
    """Remove instruction-looking *sentences*; keep everything else verbatim.

    Filtering per sentence (not per line) matters because a poisoned line can
    share a line with legitimate content — dropping the whole line would
    destroy grounded material and turn a citable answer into a refusal.
    """
    segments = re.split(r"(?<=[.!?؟])\s+|\n+", text)
    kept = [seg for seg in segments
            if seg.strip() and not any(rx.search(seg) for rx in _INSTRUCTION_RES)]
    return "\n".join(kept)


def contains_instruction(text: str) -> bool:
    return any(rx.search(line) for line in text.splitlines()
               for rx in _INSTRUCTION_RES)


def validate_question(question: str, max_chars: int = 2000) -> str:
    q = (question or "").strip()
    if not q:
        raise ValueError("question must not be empty")
    if len(q) > max_chars:
        raise ValueError(f"question too long ({len(q)} > {max_chars} chars)")
    return q


def validate_message_text(text: str, max_chars: int = 4000) -> str:
    t = (text or "").strip()
    if not t:
        raise ValueError("message text must not be empty")
    if len(t) > max_chars:
        raise ValueError(f"message too long ({len(t)} > {max_chars} chars)")
    return t


def redact_secrets(text: str) -> str:
    """Scrub likely secrets from text before logging."""
    text = re.sub(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*\S+",
                  r"\1=[redacted]", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                  "[redacted-email]", text)
    return text
