"""Layered response caching for the optimized workflow.

L1 is an exact, bounded TTL cache keyed by the complete effective context. L2
is a semantic cache used only for policy/FAQ turns. Keeping the two lookups
explicit makes cache decisions visible to LangGraph and prevents a transparent
LLM cache from bypassing OptiBot's validation and audit trail.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import settings
from app.services.embeddings import cosine, embed_one

CACHE_SCHEMA_VERSION = "response-cache-v2"
PROMPT_VERSION = "optimized-v1"
GUARDRAIL_VERSION = "deterministic-v1"


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_exact_query(text: str) -> str:
    """Conservative normalization: retain identifiers, numbers and negation."""
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    normalized = " ".join(normalized.split())
    return normalized.rstrip(".?! ")


@dataclass(frozen=True)
class CacheIdentity:
    exact_key: str
    context_key: str
    query: str
    tier: str
    semantic_allowed: bool


def build_identity(
    *,
    query: str,
    tier: str,
    order_contexts: list[dict],
    policy_chunks: list[dict],
    conversation_context: list[dict] | None = None,
    mode: str = "optimized",
) -> CacheIdentity:
    """Hash facts and versions, not merely order IDs/source filenames."""
    policy_facts = [
        {
            "source": chunk.get("source"),
            "heading": chunk.get("heading"),
            "text": chunk.get("text") or chunk.get("page_content"),
        }
        for chunk in policy_chunks
    ]
    context = {
        "orders": order_contexts,
        "policies": policy_facts,
        "conversation": conversation_context or [],
    }
    context_key = _stable_hash(context)
    identity = {
        "schema": CACHE_SCHEMA_VERSION,
        "mode": mode,
        "query": normalize_exact_query(query),
        "tier": tier,
        "prompt": PROMPT_VERSION,
        "guardrails": GUARDRAIL_VERSION,
        "context": context_key,
    }
    # Medium is the policy/FAQ tier. Order and complex turns are deliberately
    # excluded from semantic reuse even though exact reuse remains available.
    semantic_allowed = tier == "medium" and not order_contexts and not conversation_context
    return CacheIdentity(
        exact_key=_stable_hash(identity),
        context_key=context_key,
        query=query,
        tier=tier,
        semantic_allowed=semantic_allowed,
    )


@dataclass
class CacheEntry:
    query: str
    response: str
    confidence: float
    sources: list[str]
    context_key: str
    expires_at: float
    tier: str
    vector: np.ndarray | None = None
    hits: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class CacheLookup:
    hit: bool
    entry: CacheEntry | None = None
    similarity: float = 0.0
    level: str = "none"
    age_ms: int = 0


class LayeredResponseCache:
    def __init__(self) -> None:
        self._exact: OrderedDict[str, CacheEntry] = OrderedDict()
        self._semantic: list[CacheEntry] = []
        self._lock = threading.RLock()
        self.lookups = 0
        self.exact_hits = 0
        self.semantic_hits = 0
        self._gptcache = None
        if settings.semantic_cache_backend.strip().lower() == "gptcache":
            from app.services.gptcache_backend import GPTCacheBackend

            self._gptcache = GPTCacheBackend(
                threshold=settings.cache_threshold,
                max_entries=settings.cache_max_entries,
            )

    @staticmethod
    def context_key(order_ids: list[str], policy_sources: list[str]) -> str:
        """Compatibility helper for callers not yet migrated to build_identity."""
        return _stable_hash({"orders": sorted(order_ids), "policies": sorted(policy_sources)})

    def lookup(self, query: str, context_key: str) -> CacheLookup:
        """Compatibility semantic-only lookup used by the legacy path."""
        identity = CacheIdentity("", context_key, query, "medium", True)
        return self.lookup_identity(identity, exact=False)

    def lookup_identity(self, identity: CacheIdentity, *, exact: bool = True) -> CacheLookup:
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            self.lookups += 1
            if exact:
                entry = self._exact.get(identity.exact_key)
                if entry is not None:
                    entry.hits += 1
                    self.exact_hits += 1
                    self._exact.move_to_end(identity.exact_key)
                    return self._result(entry, "exact", 1.0, now)

            if not identity.semantic_allowed:
                return CacheLookup(hit=False)

            if self._gptcache is not None:
                cached = self._gptcache.get(identity.query, identity.context_key)
                if cached is None:
                    return CacheLookup(hit=False)
                entry = CacheEntry(
                    query=str(cached.get("query", identity.query)),
                    response=str(cached["response"]),
                    confidence=float(cached["confidence"]),
                    sources=[str(item) for item in cached.get("sources", [])],
                    context_key=identity.context_key,
                    expires_at=float(cached["expires_at"]),
                    tier=identity.tier,
                    created_at=float(cached.get("created_at", now)),
                )
                self.semantic_hits += 1
                return self._result(entry, "semantic", settings.cache_threshold, now)

            q_vec = embed_one(identity.query)
            best: CacheEntry | None = None
            best_sim = 0.0
            for entry in self._semantic:
                if entry.context_key != identity.context_key or entry.vector is None:
                    continue
                sim = cosine(q_vec, entry.vector)
                if sim > best_sim:
                    best, best_sim = entry, sim
            if best is not None and best_sim >= settings.cache_threshold:
                best.hits += 1
                self.semantic_hits += 1
                return self._result(best, "semantic", best_sim, now)
            return CacheLookup(hit=False, similarity=round(best_sim, 4))

    @staticmethod
    def _result(entry: CacheEntry, level: str, similarity: float, now: float) -> CacheLookup:
        return CacheLookup(
            hit=True,
            entry=entry,
            similarity=round(similarity, 4),
            level=level,
            age_ms=max(0, int((now - entry.created_at) * 1000)),
        )

    def store_identity(
        self,
        identity: CacheIdentity,
        response: str,
        confidence: float,
        sources: list[str],
    ) -> None:
        ttl = settings.cache_ttl_order if identity.tier == "simple" else settings.cache_ttl_policy
        entry = CacheEntry(
            query=identity.query,
            response=response,
            confidence=confidence,
            sources=list(sources),
            context_key=identity.context_key,
            expires_at=time.time() + ttl,
            tier=identity.tier,
            vector=embed_one(identity.query) if identity.semantic_allowed else None,
        )
        with self._lock:
            self._exact[identity.exact_key] = entry
            self._exact.move_to_end(identity.exact_key)
            if identity.semantic_allowed:
                self._semantic.append(entry)
                if self._gptcache is not None:
                    self._gptcache.put(
                        identity.query,
                        identity.context_key,
                        {
                            "query": identity.query,
                            "response": response,
                            "confidence": confidence,
                            "sources": list(sources),
                            "created_at": entry.created_at,
                            "expires_at": entry.expires_at,
                        },
                    )
            while len(self._exact) > settings.cache_max_entries:
                _, evicted = self._exact.popitem(last=False)
                self._semantic = [item for item in self._semantic if item is not evicted]

    def store(
        self,
        query: str,
        response: str,
        confidence: float,
        sources: list[str],
        context_key: str,
        tier: str,
    ) -> None:
        """Compatibility write used while the workflow migration is staged."""
        identity = CacheIdentity(_stable_hash([query, context_key, tier]), context_key, query, tier, tier == "medium")
        self.store_identity(identity, response, confidence, sources)

    def _evict_expired(self, now: float) -> None:
        expired = {id(entry) for entry in self._exact.values() if entry.expires_at <= now}
        self._exact = OrderedDict(
            (key, entry) for key, entry in self._exact.items() if entry.expires_at > now
        )
        if expired:
            self._semantic = [entry for entry in self._semantic if id(entry) not in expired]

    def clear(self) -> None:
        with self._lock:
            self._exact.clear()
            self._semantic.clear()
            if self._gptcache is not None:
                self._gptcache.clear()
            self.lookups = self.exact_hits = self.semantic_hits = 0

    def stats(self) -> dict:
        with self._lock:
            self._evict_expired(time.time())
            total_hits = self.exact_hits + self.semantic_hits
            return {
                "entries": len(self._exact),
                "semantic_entries": len(self._semantic),
                "lookups": self.lookups,
                "hits": total_hits,
                "exact_hits": self.exact_hits,
                "semantic_hits": self.semantic_hits,
                "hit_rate": round(total_hits / self.lookups, 4) if self.lookups else 0.0,
                "threshold": settings.cache_threshold,
            }


semantic_cache = LayeredResponseCache()
response_cache = semantic_cache
