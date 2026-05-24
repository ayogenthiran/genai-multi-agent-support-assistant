"""Retriever for policy RAG at query time (implementation layer).

Provides query rewriting, similarity search with lightweight reranking, and
grounded answer generation. MCP/agents call tools/document_tools.py.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.prompts.system_prompts import POLICY_ANSWER_PROMPT, POLICY_QUERY_REWRITE_PROMPT
from src.rag.vector_store import NO_DOCUMENTS_MESSAGE, get_vector_store, has_indexed_documents

INITIAL_RETRIEVAL_K = 12
FINAL_TOP_K = 5

POLICY_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "refund": ("refund", "refunds", "refunded", "return", "returns", "returned",
               "returning", "money back", "reimburse", "reimbursement"),
    "warranty": ("warranty", "warranties", "warrantied", "repair", "repairs",
                 "replacement", "replace", "replacing", "defect", "defective",
                 "malfunction", "broken"),
    "shipping": ("shipping", "shipment", "ship", "shipped", "delivery", "deliver",
                 "delivered", "delay", "delayed", "tracking", "track", "courier",
                 "in transit"),
    "cancellation": ("cancellation", "cancellations", "cancel", "cancelled",
                     "canceled", "cancelling", "canceling"),
}


def _require_openai_api_key() -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file before searching "
            "or answering policy questions."
        )


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _keyword_overlap_score(query: str, text: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    text_tokens = _tokenize(text)
    return len(query_tokens & text_tokens) / len(query_tokens)


def _infer_query_policy_types(query: str) -> set[str]:
    """Infer preferred policy_type categories from the user query.

    Returns the set of policy_type values (e.g. ``{"refund"}``) that the
    query clearly references via keyword cues. Mixed queries can return
    multiple types; general queries return an empty set so the caller can
    fall back to plain semantic reranking.
    """
    text = query.lower()
    matched: set[str] = set()
    for policy_type, keywords in POLICY_TYPE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            matched.add(policy_type)
    return matched


def _build_policy_type_filter(preferred_policy_types: set[str]) -> dict[str, Any] | None:
    """Build a Chroma metadata filter for the preferred policy categories."""
    if not preferred_policy_types:
        return None
    if len(preferred_policy_types) == 1:
        return {"policy_type": next(iter(preferred_policy_types))}
    return {"policy_type": {"$in": sorted(preferred_policy_types)}}


def _retrieve_candidates(
    store: Any,
    query: str,
    k: int,
    preferred_policy_types: set[str],
) -> list[Document]:
    """Retrieve candidate chunks, preferring metadata-filtered search when possible."""
    if preferred_policy_types:
        metadata_filter = _build_policy_type_filter(preferred_policy_types)
        try:
            filtered = store.similarity_search(query, k=k, filter=metadata_filter)
            if filtered:
                return filtered
        except Exception:
            pass

    return store.similarity_search(query, k=k)


def _rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int,
    preferred_policy_types: set[str] | None = None,
) -> list[Document]:
    if not documents:
        return []

    preferred = preferred_policy_types or set()
    has_preferred_match = any(
        doc.metadata.get("policy_type") in preferred for doc in documents
    )
    effective_preferred: set[str] = preferred if has_preferred_match else set()

    scored: list[tuple[int, float, int, Document]] = []
    for index, doc in enumerate(documents):
        policy_match = (
            1 if doc.metadata.get("policy_type") in effective_preferred else 0
        )
        overlap = _keyword_overlap_score(query, doc.page_content)
        scored.append((policy_match, overlap, index, doc))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [doc for _, _, _, doc in scored[:top_k]]


def _document_to_result(doc: Document) -> dict[str, Any]:
    return {
        "content": doc.page_content,
        "metadata": dict(doc.metadata),
        "source": doc.metadata.get("source", "unknown"),
        "page": doc.metadata.get("page"),
        "chunk_id": doc.metadata.get("chunk_id"),
        "policy_type": doc.metadata.get("policy_type"),
    }


def rewrite_policy_query(query: str) -> str:
    """Rewrite a user query into a concise policy-focused search query."""
    cleaned = query.strip()
    if not cleaned:
        return cleaned

    try:
        _require_openai_api_key()
        settings = get_settings()
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        response = (POLICY_QUERY_REWRITE_PROMPT | llm).invoke({"query": cleaned})
        rewritten = str(response.content).strip()
        return rewritten or cleaned
    except Exception:
        return cleaned


def search_policy_documents(query: str, top_k: int = 5) -> list[dict[str, Any]] | dict[str, str]:
    """Search indexed policy documents and return reranked chunk results."""
    cleaned = query.strip()
    if not cleaned:
        return []

    if not has_indexed_documents():
        return {"message": NO_DOCUMENTS_MESSAGE}

    rewritten = rewrite_policy_query(cleaned)
    preferred_policy_types = _infer_query_policy_types(cleaned)

    try:
        store = get_vector_store()
        candidates = _retrieve_candidates(
            store,
            rewritten,
            INITIAL_RETRIEVAL_K,
            preferred_policy_types,
        )
        ranked = _rerank_documents(
            cleaned,
            candidates,
            min(top_k, FINAL_TOP_K),
            preferred_policy_types=preferred_policy_types,
        )
        return [_document_to_result(doc) for doc in ranked]
    except Exception as exc:
        return {"message": f"Policy document search failed: {exc}"}


def _format_context(documents: list[dict[str, Any]]) -> str:
    formatted: list[str] = []
    for index, doc in enumerate(documents, start=1):
        source = doc.get("source", "unknown")
        page = doc.get("page", "?")
        formatted.append(
            f"[{index}] Source: {source} (page {page})\n{doc.get('content', '')}"
        )
    return "\n\n".join(formatted)


def _extract_sources(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, Any, str | None]] = set()
    for doc in documents:
        source = str(doc.get("source", "unknown"))
        page = doc.get("page")
        chunk_id = doc.get("chunk_id")
        key = (source, page, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": source, "page": page, "chunk_id": chunk_id})
    return sources


def answer_policy_question(query: str) -> dict[str, Any]:
    """Answer a policy question using retrieved context and source references."""
    cleaned = query.strip()
    if not cleaned:
        return {
            "answer": "Please provide a policy question.",
            "sources": [],
            "rewritten_query": "",
            "retrieved_context_count": 0,
        }

    if not has_indexed_documents():
        return {
            "answer": NO_DOCUMENTS_MESSAGE,
            "sources": [],
            "rewritten_query": cleaned,
            "retrieved_context_count": 0,
        }

    rewritten = rewrite_policy_query(cleaned)
    search_results = search_policy_documents(cleaned, top_k=FINAL_TOP_K)

    if isinstance(search_results, dict):
        return {
            "answer": search_results.get("message", "Policy search failed."),
            "sources": [],
            "rewritten_query": rewritten,
            "retrieved_context_count": 0,
        }

    if not search_results:
        return {
            "answer": (
                "I could not find relevant policy information for that question. "
                "Try rephrasing your question or upload additional policy documents."
            ),
            "sources": [],
            "rewritten_query": rewritten,
            "retrieved_context_count": 0,
        }

    context = _format_context(search_results)
    sources = _extract_sources(search_results)

    try:
        _require_openai_api_key()
        settings = get_settings()
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        response = (POLICY_ANSWER_PROMPT | llm).invoke(
            {"context": context, "question": cleaned},
        )
        answer = str(response.content).strip()
    except Exception as exc:
        return {
            "answer": f"Failed to generate a policy answer: {exc}",
            "sources": sources,
            "rewritten_query": rewritten,
            "retrieved_context_count": len(search_results),
        }

    if not answer:
        answer = (
            "The uploaded policy documents do not contain enough information "
            "to answer that question."
        )

    return {
        "answer": answer,
        "sources": sources,
        "rewritten_query": rewritten,
        "retrieved_context_count": len(search_results),
    }
