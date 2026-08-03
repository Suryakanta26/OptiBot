"""LangChain document ingestion and in-memory retrieval for policy Markdown."""

from __future__ import annotations

import hashlib
import re
import threading

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import POLICY_DIR, settings
from app.services.embeddings import OptiBotEmbeddings

_index_lock = threading.Lock()
_documents: list[Document] = []
_store: InMemoryVectorStore | None = None

_HEADERS = [("#", "title"), ("##", "heading"), ("###", "subheading")]


def _chunk_id(document: Document) -> str:
    payload = "|".join(
        [
            str(document.metadata.get("source", "")),
            str(document.metadata.get("title", "")),
            str(document.metadata.get("heading", "")),
            str(document.metadata.get("subheading", "")),
            document.page_content,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_and_split() -> list[Document]:
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS,
        strip_headers=False,
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_tokens * 4,
        chunk_overlap=settings.rag_chunk_overlap * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    sections: list[Document] = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        loaded = TextLoader(str(path), encoding="utf-8").load()
        for raw in loaded:
            for section in markdown_splitter.split_text(raw.page_content):
                section.metadata.update(
                    {
                        "source": path.name,
                        "path": str(path),
                        "document_type": "policy",
                    }
                )
                sections.append(section)

    chunks = recursive_splitter.split_documents(sections)
    for chunk in chunks:
        chunk.metadata["chunk_id"] = _chunk_id(chunk)
    return chunks


def build_index(force: bool = False) -> int:
    """Load, structurally split, embed, and index the policy corpus once."""
    global _documents, _store
    with _index_lock:
        if _store is not None and not force:
            return len(_documents)
        if not POLICY_DIR.exists():
            raise FileNotFoundError(f"Policy directory not found: {POLICY_DIR}")
        documents = _load_and_split()
        if not documents:
            raise RuntimeError(f"No policy chunks produced from {POLICY_DIR}")
        store = InMemoryVectorStore(embedding=OptiBotEmbeddings())
        store.add_documents(documents, ids=[doc.metadata["chunk_id"] for doc in documents])
        _documents, _store = documents, store
        return len(_documents)


def _rerank(query: str, candidates: list[dict]) -> list[dict]:
    q_terms = {term for term in re.findall(r"[a-z]{4,}", query.lower())}
    seen_sources: set[str] = set()
    for candidate in candidates:
        text_terms = set(re.findall(r"[a-z]{4,}", candidate["text"].lower()))
        overlap = len(q_terms & text_terms) / max(len(q_terms), 1)
        penalty = 0.05 if candidate["source"] in seen_sources else 0.0
        seen_sources.add(candidate["source"])
        candidate["rerank_score"] = round(candidate["score"] + 0.25 * overlap - penalty, 4)
    return sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    if _store is None:
        build_index()
    assert _store is not None
    k = top_k or settings.rag_top_k
    results = _store.similarity_search_with_score(query, k=max(k * 3, 8))
    candidates = [
        {
            "source": str(doc.metadata.get("source", "unknown")),
            "heading": str(doc.metadata.get("heading") or doc.metadata.get("title") or ""),
            "text": doc.page_content,
            "score": round(float(score), 4),
            "chunk_id": str(doc.metadata.get("chunk_id", "")),
        }
        for doc, score in results
        if float(score) >= settings.rag_min_score
    ]
    return _rerank(query, candidates)[:k] if candidates else []


def index_stats() -> dict:
    if _store is None:
        build_index()
    sources: dict[str, int] = {}
    for document in _documents:
        source = str(document.metadata.get("source", "unknown"))
        sources[source] = sources.get(source, 0) + 1
    dimension = len(OptiBotEmbeddings().embed_query("dimension probe"))
    return {"chunks": len(_documents), "dimensions": dimension, "sources": sources}
