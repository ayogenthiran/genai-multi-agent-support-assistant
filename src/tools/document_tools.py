"""MCP- and agent-facing document search tools (thin wrapper layer).

This module is the *public tool surface*: LangChain @tool functions registered
with agents and the MCP server. It should NOT reimplement RAG logic.

Implementation lives in `src/rag/`:
  - document_loader.py → load PDF pages from policy files
  - vector_store.py    → chunk, embed, and persist to ChromaDB
  - retriever.py       → query rewrite, search, rerank, and grounded answers

Import from rag/ here and expose pdf_ingestion, policy_document_search, and
policy_question_answer as MCP-style tools.
"""

from __future__ import annotations

from typing import Any

from src.rag.retriever import (
    answer_policy_question as _answer_policy_question,
    search_policy_documents as _search_policy_documents,
)
from src.rag.vector_store import add_pdf_to_vector_store


def pdf_ingestion(file_path: str) -> dict[str, Any]:
    """Index a policy PDF into the vector store."""
    return add_pdf_to_vector_store(file_path)


def policy_document_search(query: str) -> str:
    """Search indexed policy documents and return relevant passages."""
    results = _search_policy_documents(query, top_k=5)

    if isinstance(results, dict):
        return str(results.get("message", "Policy document search failed."))

    if not results:
        return "No relevant policy passages found for your query."

    formatted: list[str] = []
    for index, doc in enumerate(results, start=1):
        source = doc.get("source", "unknown")
        page = doc.get("page", "?")
        formatted.append(
            f"[{index}] Source: {source} (page {page})\n{doc.get('content', '')}"
        )
    return "\n\n".join(formatted)


def policy_question_answer(query: str) -> dict[str, Any]:
    """Answer a policy question using retrieved document context.

    Returns a structured payload with the grounded answer, source references,
    the rewritten search query, and the number of retrieved context chunks
    so callers (agents, UI, MCP clients) can render or audit the RAG result.
    """
    result = _answer_policy_question(query)
    return {
        "answer": str(result.get("answer", "")).strip(),
        "sources": list(result.get("sources") or []),
        "rewritten_query": str(result.get("rewritten_query", "")).strip(),
        "retrieved_context_count": int(result.get("retrieved_context_count", 0) or 0),
    }


def format_policy_answer(result: dict[str, Any]) -> str:
    """Format a policy_question_answer dict into a human-readable string."""
    answer = str(result.get("answer", "")).strip()
    sources = result.get("sources") or []
    if not sources:
        return answer

    source_lines = [
        f"- {item.get('source', 'unknown')} (page {item.get('page', '?')})"
        for item in sources
    ]
    return f"{answer}\n\nSources:\n" + "\n".join(source_lines)


def get_document_tools() -> list[Any]:
    """Return LangChain tools for MCP and agent tool registries."""
    from langchain_core.tools import tool

    @tool
    def search_policy_documents_tool(query: str) -> str:
        """Search indexed company policy documents for relevant passages."""
        return policy_document_search(query)

    @tool
    def answer_policy_question_tool(query: str) -> str:
        """Answer a customer policy question using indexed policy documents."""
        return format_policy_answer(policy_question_answer(query))

    @tool
    def add_policy_document(file_path: str) -> dict[str, Any]:
        """Index a policy PDF so it can be searched and used for answers."""
        return pdf_ingestion(file_path)

    return [
        search_policy_documents_tool,
        answer_policy_question_tool,
        add_policy_document,
    ]
