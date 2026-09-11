"""Pluggable embedding backends.

- ``HashEmbedder``: deterministic, offline, dependency-free. Character-trigram
  hashing into a fixed-dim vector (Unicode-safe, so Arabic works). Used by
  tests, the demo, and the default config.
- ``OpenAIEmbedder``: live option over HTTP (stdlib urllib). Never used offline;
  construction does not touch the network.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from dataclasses import dataclass

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Arabic orthography varies (alef forms, ta-marbuta, diacritics, the "ال"
# definite article, pronoun suffixes). Normalization + prefix matching keep
# retrieval working across those variants. English tokens pass through.
_ALEF_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ؤ": "و", "ئ": "ي",
                           "ة": "ه", "ى": "ي"})
_DIACRITICS_RE = re.compile(r"[ً-ٰٓـ]")


def norm_token(tok: str) -> str:
    t = tok.lower().translate(_ALEF_MAP)
    t = _DIACRITICS_RE.sub("", t)
    if t.startswith("ال") and len(t) > 4:
        t = t[2:]
    return t


def tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def norm_tokens(text: str) -> list[str]:
    return [norm_token(t) for t in tokens(text) if norm_token(t)]


def term_match(a: str, b: str) -> bool:
    """Normalized term match: equal, or one is a prefix-stem of the other."""
    if a == b:
        return True
    if len(a) >= 3 and len(b) >= 3 and (a.startswith(b) or b.startswith(a)):
        return True
    return False


class Embedder:
    dim: int

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class HashEmbedder(Embedder):
    """Deterministic char-trigram hash embeddings, L2-normalized."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, feature: str) -> int:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        lowered = text.lower()
        # Char trigrams (with padding) catch Arabic morphology; word unigrams
        # add exact-term signal. Both hashed into the same space.
        padded = f"  {lowered}  "
        for i in range(len(padded) - 2):
            vec[self._bucket("3:" + padded[i:i + 3])] += 1.0
        for tok in set(tokens(text)):
            vec[self._bucket("w:" + tok)] += 2.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


@dataclass
class OpenAIEmbedder(Embedder):
    """Live embedding backend (requires LLM_API_KEY + network). Documented only."""

    model: str = "text-embedding-3-small"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    dim: int = 1536

    def embed(self, text: str) -> list[float]:
        if not self.api_key:
            raise RuntimeError("OpenAIEmbedder needs LLM_API_KEY (see .env.example)")
        payload = json.dumps({"model": self.model, "input": text}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/embeddings", data=payload,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data["data"][0]["embedding"]


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


def lexical_overlap(query: str, doc: str) -> float:
    """Normalized token-overlap in [0, 1]: matched |Q| terms / |Q|."""
    q = [t for t in norm_tokens(query)]
    if not q:
        return 0.0
    d = norm_tokens(doc)
    matched = sum(1 for qt in q if any(term_match(qt, dt) for dt in d))
    return matched / len(q)


def make_embedder(settings) -> Embedder:
    if settings.embedder == "openai":
        return OpenAIEmbedder(model=settings.llm_model or "text-embedding-3-small",
                              api_key=settings.llm_api_key,
                              dim=settings.embedding_dim
                              if settings.embedding_dim != 512 else 1536)
    return HashEmbedder(dim=settings.embedding_dim)
