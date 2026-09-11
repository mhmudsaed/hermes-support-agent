"""Optional live-LLM rephrasing, behind an adapter with an offline default.

The offline composer (answer.py) is the source of truth: it builds answers by
extracting sentences from retrieved chunks, so answers are grounded with or
without an LLM. When configured, ``OpenAIRephraser`` only *rewords* that draft
— the prompt forbids adding facts — and citations are re-attached afterwards.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass


class Rephraser:
    def rephrase(self, draft: str, question: str, lang: str) -> str:
        raise NotImplementedError


class NoopRephraser(Rephraser):
    """Offline default: return the extractive draft unchanged."""

    def rephrase(self, draft: str, question: str, lang: str) -> str:
        return draft


@dataclass
class OpenAIRephraser(Rephraser):
    """Live option: reword via a chat-completions endpoint (needs key + net)."""

    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"

    def rephrase(self, draft: str, question: str, lang: str) -> str:
        if not self.api_key:
            raise RuntimeError("LLM rephrasing needs LLM_API_KEY (see .env.example)")
        lang_name = "Arabic" if lang == "ar" else "English"
        system = (
            "You reword a customer-support draft answer. Rules: use ONLY the "
            "facts in the draft, add no new facts, keep every [Source: ...] "
            f"marker exactly as-is, and reply in {lang_name}."
        )
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",
                 "content": f"Question: {question}\n\nDraft:\n{draft}"},
            ],
            "temperature": 0.2,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


def make_rephraser(settings) -> Rephraser:
    if settings.llm_provider in ("openai", "compatible") and settings.llm_api_key:
        return OpenAIRephraser(model=settings.llm_model or "gpt-4o-mini",
                               api_key=settings.llm_api_key)
    return NoopRephraser()
