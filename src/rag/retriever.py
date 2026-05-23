"""Retriever for fetching relevant policy passages at query time (implementation layer).

Used internally by the RAG agent and by tools/document_tools.search_policy_documents().
Do not register this module directly with MCP — keep the tool surface in tools/.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from src.config import get_settings
from src.rag.vector_store import get_vector_store, has_indexed_documents


def get_retriever(vector_store: Any, top_k: int | None = None) -> Any:
    """Return a LangChain retriever backed by the vector store."""
    settings = get_settings()
    k = top_k or settings.rag_top_k
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve_documents(query: str, k: int | None = None) -> list[Document]:
    """Run similarity search against indexed policy documents."""
    if not has_indexed_documents():
        return []

    settings = get_settings()
    top_k = k or settings.rag_top_k
    store = get_vector_store()
    return store.similarity_search(query, k=top_k)
