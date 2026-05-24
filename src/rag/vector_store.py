"""Chroma vector store helpers for policy document chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.rag.document_loader import load_documents, load_pdf_pages

COLLECTION_NAME = "policy_documents"
NO_DOCUMENTS_MESSAGE = (
    "No policy documents are indexed yet. "
    "Please upload or generate policy PDFs, then click "
    "'Process Policy PDF' in the sidebar to index them."
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_store: Chroma | None = None
_store_persist_dir: Path | None = None


def _require_openai_api_key() -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file before indexing "
            "or searching policy documents."
        )


def _get_embeddings() -> OpenAIEmbeddings:
    _require_openai_api_key()
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


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split page-level documents into overlapping chunks with metadata preserved."""
    if not documents:
        return []

    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size or CHUNK_SIZE,
        chunk_overlap=settings.rag_chunk_overlap or CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        chunk.metadata["chunk_id"] = f"{source}_p{page}_c{index}_{uuid4().hex[:8]}"

    return chunks


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


def add_pdf_to_vector_store(file_path: str) -> dict[str, Any]:
    """Load a PDF, chunk it, and add the chunks to the persisted Chroma collection."""
    try:
        page_documents = load_pdf_pages(file_path)
        chunks = chunk_documents(page_documents)
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
    except ValueError as exc:
        return {
            "file": file_path,
            "chunks_added": 0,
            "message": str(exc),
        }
    except Exception as exc:
        return {
            "file": file_path,
            "chunks_added": 0,
            "message": f"Failed to index PDF: {exc}",
        }


def index_documents(persist_dir: Path | None = None, policies_dir: Path | None = None) -> int:
    """Embed and store all policy PDFs from the policies directory."""
    page_documents = load_documents(policies_dir)
    chunks = chunk_documents(page_documents)
    if not chunks:
        return 0

    store = get_vector_store(persist_dir)
    ids = store.add_documents(chunks)
    return len(ids)


def reset_vector_store(persist_dir: Path | None = None) -> None:
    """Delete the policy_documents collection and clear the cached store."""
    global _store, _store_persist_dir

    settings = get_settings()
    target_dir = persist_dir or settings.chroma_persist_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(target_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    _store = None
    _store_persist_dir = None
