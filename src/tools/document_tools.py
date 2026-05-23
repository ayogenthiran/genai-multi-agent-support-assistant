"""MCP- and agent-facing document search tools (thin wrapper layer).

This module is the *public tool surface*: LangChain @tool functions registered
with agents and the MCP server. It should NOT reimplement RAG logic.

Implementation lives in `src/rag/`:
  - document_loader.py → load and chunk PDFs
  - vector_store.py    → embed and persist to Chroma
  - retriever.py       → similarity search at query time

Import from rag/ here and expose e.g. `search_policy_documents()`.
"""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.rag.retriever import retrieve_documents
from src.rag.vector_store import (
    NO_DOCUMENTS_MESSAGE,
    add_document_to_vector_store,
    has_indexed_documents,
)


def _format_passages(docs: list[Any]) -> str:
    formatted: list[str] = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("filename") or doc.metadata.get("source", "unknown")
        formatted.append(f"[{index}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(formatted)


def search_policy_documents(query: str, k: int = 4) -> str:
    """Search indexed policy documents and return relevant passages."""
    cleaned_query = query.strip()
    if not cleaned_query:
        return "Please provide a search query."

    if not has_indexed_documents():
        return NO_DOCUMENTS_MESSAGE

    docs = retrieve_documents(cleaned_query, k=k)
    if not docs:
        return "No relevant policy passages found for your query."

    return _format_passages(docs)


def answer_policy_question(query: str) -> str:
    """Answer a policy question using retrieved document context."""
    cleaned_query = query.strip()
    if not cleaned_query:
        return "Please provide a policy question."

    if not has_indexed_documents():
        return NO_DOCUMENTS_MESSAGE

    settings = get_settings()
    docs = retrieve_documents(cleaned_query, k=settings.rag_top_k)
    if not docs:
        return (
            "I could not find relevant policy information for that question. "
            "Try rephrasing your question or upload additional policy documents."
        )

    context = _format_passages(docs)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful customer support assistant. "
                "Answer the user's question using only the provided policy excerpts. "
                "If the excerpts do not contain enough information, say so clearly. "
                "Keep the answer concise and practical.",
            ),
            (
                "human",
                "Policy excerpts:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    response = (prompt | llm).invoke(
        {"context": context, "question": cleaned_query},
    )
    return str(response.content)


def get_document_tools() -> list[Any]:
    """Return LangChain tools for MCP and agent tool registries."""
    from langchain_core.tools import tool

    @tool
    def search_policy_documents_tool(query: str, k: int = 4) -> str:
        """Search indexed company policy documents for relevant passages."""
        return search_policy_documents(query, k=k)

    @tool
    def answer_policy_question_tool(query: str) -> str:
        """Answer a customer policy question using indexed policy documents."""
        return answer_policy_question(query)

    @tool
    def add_policy_document(file_path: str) -> dict[str, Any]:
        """Index a policy PDF so it can be searched and used for answers."""
        return add_document_to_vector_store(file_path)

    return [
        search_policy_documents_tool,
        answer_policy_question_tool,
        add_policy_document,
    ]
