"""Embedding backend shared by the RAG retriever and the semantic cache.

Two backends, selected at import time:

* ``sentence-transformers`` (all-MiniLM-L6-v2) when it is installed — real
  sentence embeddings.
* A domain-aware lexical embedder otherwise, so the app still boots and the
  demo still runs on a machine without torch. It normalises order numbers and
  maps eCommerce synonyms onto shared tokens before hashing, which is enough
  for "where is my order" and "what's the status of my order" to land in the
  same cache bucket.

``ACTIVE_BACKEND`` is surfaced on /api/health so a demo never silently runs on
the weaker path without anyone noticing.
"""

from __future__ import annotations

import hashlib
import re
import threading

import numpy as np
from langchain_core.embeddings import Embeddings

_DIM_FALLBACK = 512
_MODEL_NAME = "all-MiniLM-L6-v2"

# Words that mean the same thing to a support bot. Collapsing them is what
# lets the lexical fallback behave semantically on this domain.
_SYNONYMS = {
    "wheres": "status", "where": "status", "track": "status",
    "tracking": "status", "locate": "status", "eta": "status",
    "arrive": "status", "arriving": "status", "delivery": "status",
    "deliver": "status", "delivered": "status", "update": "status",
    "purchase": "order", "package": "order", "parcel": "order",
    "shipment": "order", "item": "order", "buy": "order",
    "refund": "return", "returning": "return", "send back": "return",
    "money back": "return", "exchange": "return",
    "policy": "policy", "rule": "policy", "terms": "policy",
    "guarantee": "warranty", "broken": "damaged", "defective": "damaged",
    "faulty": "damaged", "cancel": "cancel", "cancelling": "cancel",
}

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "i", "my", "me", "you", "your", "it", "its", "of", "to", "for",
    "on", "in", "at", "and", "or", "can", "could", "would", "should", "will",
    "please", "hi", "hello", "hey", "thanks", "thank", "what", "whats", "how",
    "when", "why", "s", "t", "im", "ive", "got", "have", "has", "get",
}

_ORDER_RE = re.compile(r"\bORD[-\s]?\d{3,}\b", re.IGNORECASE)
_NUM_RE = re.compile(r"\b\d{4,}\b")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Strip order-specific identifiers so the *pattern* is what gets cached.

    "Where is ORD-10042?" and "Where is ORD-10099?" normalise to the same
    string — the cache stores the shape of the answer, and the order data is
    re-injected per request.
    """
    lowered = text.lower().strip()
    lowered = _ORDER_RE.sub(" <order_id> ", lowered)
    lowered = _NUM_RE.sub(" <number> ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for raw in _TOKEN_RE.findall(normalize(text)):
        if raw in _STOPWORDS:
            continue
        out.append(_SYNONYMS.get(raw, raw))
    return out


class _LexicalEmbedder:
    """Hashing bag-of-words with synonym folding. No external model needed."""

    name = "lexical-fallback"
    dim = _DIM_FALLBACK

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = _tokens(text)
            if not toks:
                continue
            for tok in toks:
                digest = hashlib.md5(tok.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, idx] += sign
            # Bigrams add a little word-order sensitivity.
            for a, b in zip(toks, toks[1:]):
                digest = hashlib.md5(f"{a}_{b}".encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "big") % self.dim
                matrix[row, idx] += 0.5
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class _TransformerEmbedder:
    name = f"sentence-transformers/{_MODEL_NAME}"

    def __init__(self, model) -> None:
        self._model = model
        self.dim = model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


_embedder = None
_lock = threading.Lock()


def get_embedder():
    """Load the best available backend once, lazily."""
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is not None:
            return _embedder
        try:
            from sentence_transformers import SentenceTransformer

            _embedder = _TransformerEmbedder(SentenceTransformer(_MODEL_NAME))
        except Exception:
            # No torch, no network, or a model download failure — the demo
            # still needs to run, so fall back rather than crash on boot.
            _embedder = _LexicalEmbedder()
        return _embedder


def embed(texts: list[str]) -> np.ndarray:
    return get_embedder().encode(texts)


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for already-normalised vectors (safe if they aren't)."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def backend_name() -> str:
    return get_embedder().name


class OptiBotEmbeddings(Embeddings):
    """LangChain embedding adapter over the app's local/fallback backend."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return embed_one(text).tolist()
