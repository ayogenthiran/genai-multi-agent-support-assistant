"""RAG implementation layer: ingestion, storage, and retrieval.

This package owns the *internals* of document search. Agents and MCP should
call `src/tools/document_tools.py`, which wraps these modules — do not expose
rag/ functions directly to MCP unless through that thin tool layer.
"""

__all__ = [
    "document_loader",
    "vector_store",
    "retriever",
]
