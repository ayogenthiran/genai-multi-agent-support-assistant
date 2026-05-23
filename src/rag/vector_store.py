"""Chroma vector store for embedded policy document chunks (implementation layer).

Handles embedding and persistence only. Query-time search is in retriever.py;
MCP/agent exposure is in tools/document_tools.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.config import get_settings
from src.rag.document_loader import load_documents, process_pdf

COLLECTION_NAME = "policy_documents"
NO_DOCUMENTS_MESSAGE = (
    "No policy documents are indexed yet. "
    "Please upload a policy PDF first using add_document_to_vector_store()."
)

_store: Chroma | None = None
_store_persist_dir: Path | None = None


def _get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )


def get_vector_store(persist_dir: Path | None = None) -> Chroma:
    """Return a Chroma vector store connected to the persist directory."""
    global _store, _store_persist_dir

    settings = get_settings()
    target_dir = persist_dir or settings.chroma_persist_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    if _store is None or _store_persist_dir != target_dir:
        _store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_get_embeddings(),
            persist_directory=str(target_dir),
        )
        _store_persist_dir = target_dir

    return _store


def get_document_count(persist_dir: Path | None = None) -> int:
    """Return the number of indexed chunks in the vector store."""
    settings = get_settings()
    target_dir = persist_dir or settings.chroma_persist_dir
    if not target_dir.exists():
        return 0

    client = chromadb.PersistentClient(path=str(target_dir))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return 0
    return collection.count()


def has_indexed_documents() -> bool:
    """Return True when at least one document chunk is stored in Chroma."""
    return get_document_count() > 0


def add_document_to_vector_store(file_path: str) -> dict[str, Any]:
    """Process a PDF and add its chunks to the persisted Chroma collection."""
    chunks = process_pdf(file_path)
    if not chunks:
        return {
            "file": file_path,
            "chunks_added": 0,
            "message": f"No chunks were created from PDF: {file_path}",
        }

    store = get_vector_store()
    ids = store.add_documents(chunks)
    return {
        "file": file_path,
        "chunks_added": len(ids),
        "message": f"Indexed {len(ids)} chunks from {Path(file_path).name}.",
    }


def index_documents(persist_dir: Path, policies_dir: Path) -> None:
    """Embed and store all policy documents in the vector database."""
    store = get_vector_store(persist_dir)
    chunks = load_documents(policies_dir)
    if chunks:
        store.add_documents(chunks)
