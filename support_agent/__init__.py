"""hermes-support-agent: bilingual support agent with cited answers and ticket handoff."""

from support_agent.answer import Answer, Citation, answer_question
from support_agent.config import Settings, load_settings

__all__ = ["Answer", "Citation", "Settings", "answer_question", "load_settings"]
__version__ = "1.0.0"
