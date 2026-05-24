"""Document tools exposed to agents and the MCP server."""

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
    """Answer a policy question and return structured source metadata."""
    result = _answer_policy_question(query)
    return {
        "answer": str(result.get("answer", "")).strip(),
        "sources": list(result.get("sources") or []),
        "rewritten_query": str(result.get("rewritten_query", "")).strip(),
        "retrieved_context_count": int(result.get("retrieved_context_count", 0) or 0),
    }


def _sort_pages(pages: set[Any]) -> list[Any]:
    return sorted(
        pages,
        key=lambda page: (
            not isinstance(page, int),
            page if isinstance(page, int) else str(page),
        ),
    )


def _format_source_reference(source: str, pages: set[Any]) -> str:
    sorted_pages = _sort_pages(pages)
    label = "page" if len(sorted_pages) == 1 else "pages"
    return f"{source} ({label} {', '.join(str(page) for page in sorted_pages)})"


def format_source_references(sources: list[dict[str, Any]]) -> str:
    """Format source metadata with duplicate file/page pairs removed."""
    source_pages: dict[str, set[Any]] = {}
    for item in sources:
        source = str(item.get("source", "unknown"))
        page = item.get("page", "?")
        source_pages.setdefault(source, set()).add(page)

    references = [
        _format_source_reference(source, pages)
        for source, pages in source_pages.items()
    ]
    if not references:
        return ""
    if len(references) == 1:
        return f"Sources: {references[0]}"
    return "Sources:\n" + "\n".join(f"- {reference}" for reference in references)


def format_policy_answer(result: dict[str, Any]) -> str:
    """Format a policy_question_answer dict into a human-readable string."""
    answer = str(result.get("answer", "")).strip()
    sources = result.get("sources") or []
    if not sources:
        return answer

    source_block = format_source_references(sources)
    if not source_block:
        return answer
    return f"{answer}\n\n{source_block}"


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
