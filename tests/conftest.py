"""Shared test fixtures: offline agent over a temp SQLite DB + sample docs."""

from __future__ import annotations

import glob
import os

import pytest

from support_agent.config import Settings
from support_agent.service import SupportAgent

SAMPLE_DOCS = os.path.join(os.path.dirname(__file__), "..", "sample_docs")


def make_settings(tmp_path, **overrides) -> Settings:
    kwargs = {
        "backend": "sqlite",
        "db_path": str(tmp_path / "test.db"),
        "top_k": 3,
        "confidence_threshold": 0.12,
        "embedder": "hash",
        "embedding_dim": 512,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


@pytest.fixture
def agent(tmp_path) -> SupportAgent:
    ag = SupportAgent(make_settings(tmp_path))
    ag.ingest_dir(os.path.abspath(SAMPLE_DOCS))
    yield ag
    ag.store.close()


@pytest.fixture
def sample_doc_paths() -> list[str]:
    base = os.path.abspath(SAMPLE_DOCS)
    paths = sorted(glob.glob(os.path.join(base, "*.md"))
                   + glob.glob(os.path.join(base, "*.txt")))
    return [p for p in paths if os.path.basename(p) != "README.md"]
