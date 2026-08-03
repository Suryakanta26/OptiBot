"""Optional GPTCache get/put backend for policy-response semantic reuse."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from app.services.embeddings import embed_one, get_embedder


class GPTCacheBackend:
    """Process-local GPTCache with SQLite scalar storage and FAISS vectors."""

    def __init__(self, *, threshold: float, max_entries: int) -> None:
        from gptcache import Cache, Config
        from gptcache.manager import CacheBase, VectorBase, get_data_manager
        from gptcache.processor.pre import get_prompt
        from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

        self._tempdir = tempfile.TemporaryDirectory(prefix="optibot-gptcache-")
        index_path = str(Path(self._tempdir.name) / "semantic.faiss")
        manager = get_data_manager(
            CacheBase("sqlite", sql_url="sqlite:///:memory:"),
            VectorBase("faiss", dimension=get_embedder().dim, index_path=index_path),
            max_size=max_entries,
            eviction="LRU",
        )
        self._cache = Cache()
        self._cache.init(
            pre_embedding_func=get_prompt,
            embedding_func=lambda text, **_: embed_one(text),
            data_manager=manager,
            similarity_evaluation=SearchDistanceEvaluation(),
            config=Config(similarity_threshold=threshold, disable_report=True),
        )

    @staticmethod
    def _prompt(query: str, context_key: str) -> str:
        return f"context:{context_key}\nquery:{query}"

    def get(self, query: str, context_key: str) -> dict[str, Any] | None:
        from gptcache.adapter.api import get

        raw = get(self._prompt(query, context_key), cache_obj=self._cache)
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        if float(value.get("expires_at", 0)) <= time.time():
            return None
        return value

    def put(self, query: str, context_key: str, value: dict[str, Any]) -> None:
        from gptcache.adapter.api import put

        put(
            self._prompt(query, context_key),
            json.dumps(value, separators=(",", ":")),
            cache_obj=self._cache,
        )

    def clear(self) -> None:
        self._cache.flush()
