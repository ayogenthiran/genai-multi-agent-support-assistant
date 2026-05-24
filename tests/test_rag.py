"""Tests for the policy RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from langchain_core.documents import Document

from src.create_policy_pdfs import create_policy_pdfs
from src.rag.document_loader import infer_policy_type, load_documents, load_pdf_pages
from src.rag.retriever import (
    _build_policy_type_filter,
    _infer_query_policy_types,
    _rerank_documents,
    rewrite_policy_query,
    search_policy_documents,
)
from src.rag.vector_store import has_indexed_documents, reset_vector_store
from src.tools import document_tools


def test_infer_policy_type_from_filename() -> None:
    assert infer_policy_type("refund_policy.pdf") == "refund"
    assert infer_policy_type("warranty_policy.pdf") == "warranty"
    assert infer_policy_type("shipping_policy.pdf") == "shipping"
    assert infer_policy_type("cancellation_terms.pdf") == "cancellation"
    assert infer_policy_type("general_terms.pdf") == "general"


def test_create_policy_pdfs_writes_expected_files(tmp_path: Path) -> None:
    created = create_policy_pdfs(tmp_path)
    assert len(created) == 3
    for filename in ("refund_policy.pdf", "warranty_policy.pdf", "shipping_policy.pdf"):
        path = tmp_path / filename
        assert path.exists()
        pages = load_pdf_pages(str(path))
        assert pages
        assert pages[0].metadata["policy_type"] == infer_policy_type(filename)


def test_load_documents_searches_policy_subdirectories(tmp_path: Path) -> None:
    nested_policy_dir = tmp_path / "sample"
    create_policy_pdfs(nested_policy_dir)

    documents = load_documents(tmp_path)

    sources = {doc.metadata["source"] for doc in documents}
    assert {
        "refund_policy.pdf",
        "warranty_policy.pdf",
        "shipping_policy.pdf",
    }.issubset(sources)


def test_rewrite_policy_query_returns_non_empty_string() -> None:
    mock_response = MagicMock()
    mock_response.content = "refund eligibility window and conditions"

    with patch("src.rag.retriever.ChatOpenAI") as mock_llm_cls:
        mock_llm_cls.return_value.invoke.return_value = mock_response
        rewritten = rewrite_policy_query("Can a customer get a refund after 20 days?")

    assert rewritten
    assert isinstance(rewritten, str)


def test_rewrite_policy_query_falls_back_to_original_query() -> None:
    with patch("src.rag.retriever.ChatOpenAI", side_effect=RuntimeError("llm down")):
        query = "What is the warranty coverage?"
        assert rewrite_policy_query(query) == query


def test_document_tools_expose_policy_question_answer() -> None:
    assert callable(document_tools.policy_question_answer)
    assert callable(document_tools.policy_document_search)
    assert callable(document_tools.pdf_ingestion)


class _UploadedPolicyPdf:
    name = "refund_policy.pdf"

    def getbuffer(self) -> bytes:
        return b"%PDF-1.4 sample refund policy"


def test_streamlit_pdf_processing_resets_index_before_ingestion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policies_dir = tmp_path / "policies"
    monkeypatch.setenv("POLICIES_DIR", str(policies_dir))

    from src.config import get_settings

    get_settings.cache_clear()
    import app

    calls: list[str | tuple[str, str]] = []

    def _record_reset() -> None:
        calls.append("reset")

    def _record_ingestion(file_path: str) -> dict[str, object]:
        calls.append(("ingest", file_path))
        return {
            "file": file_path,
            "chunks_added": 1,
            "message": "Indexed 1 chunks from refund_policy.pdf.",
        }

    try:
        with patch.object(app, "reset_vector_store", side_effect=_record_reset), patch.object(
            app,
            "pdf_ingestion",
            side_effect=_record_ingestion,
        ), patch.object(app.st, "spinner"), patch.object(app.st, "success") as success:
            app._process_uploaded_pdf(_UploadedPolicyPdf())
    finally:
        get_settings.cache_clear()

    assert calls[0] == "reset"
    assert calls[1][0] == "ingest"
    assert Path(calls[1][1]).name == "refund_policy.pdf"
    assert (policies_dir / "refund_policy.pdf").exists()
    success.assert_called_once()


def test_policy_question_answer_returns_required_fields() -> None:
    mock_payload = {
        "answer": "Refunds are available within 30 days of delivery.",
        "sources": [{"source": "refund_policy.pdf", "page": 1, "chunk_id": "chunk-1"}],
        "rewritten_query": "refund eligibility window and conditions",
        "retrieved_context_count": 3,
    }

    with patch(
        "src.tools.document_tools._answer_policy_question",
        return_value=mock_payload,
    ):
        result = document_tools.policy_question_answer("What is the refund policy?")

    assert result["answer"]
    assert isinstance(result["sources"], list)
    assert result["sources"]
    assert result["rewritten_query"]
    assert result["retrieved_context_count"] == 3


def test_format_policy_answer_dedupes_and_groups_sources() -> None:
    result = {
        "answer": "Refunds are available within 30 days.",
        "sources": [
            {"source": "refund_policy.pdf", "page": 1, "chunk_id": "chunk-1"},
            {"source": "refund_policy.pdf", "page": 1, "chunk_id": "chunk-2"},
            {"source": "refund_policy.pdf", "page": 2, "chunk_id": "chunk-3"},
        ],
    }

    formatted = document_tools.format_policy_answer(result)

    assert "Sources: refund_policy.pdf (pages 1, 2)" in formatted
    assert formatted.count("refund_policy.pdf") == 1


def test_infer_query_policy_types_detects_refund_keywords() -> None:
    assert _infer_query_policy_types("What is the refund policy?") == {"refund"}
    assert _infer_query_policy_types("Can I return this item?") == {"refund"}
    assert _infer_query_policy_types("How do I get my money back?") == {"refund"}


def test_infer_query_policy_types_detects_warranty_keywords() -> None:
    assert _infer_query_policy_types("Is the repair covered under warranty?") == {
        "warranty"
    }
    assert _infer_query_policy_types("I need a replacement for a defective unit") == {
        "warranty"
    }


def test_infer_query_policy_types_detects_shipping_keywords() -> None:
    assert _infer_query_policy_types("My delivery is delayed") == {"shipping"}
    assert _infer_query_policy_types("Where is the tracking number?") == {"shipping"}


def test_infer_query_policy_types_detects_cancellation_keywords() -> None:
    assert _infer_query_policy_types("How do I cancel my order?") == {"cancellation"}


def test_infer_query_policy_types_returns_empty_for_general_query() -> None:
    assert _infer_query_policy_types("What are your company policies?") == set()


def test_infer_query_policy_types_supports_mixed_queries() -> None:
    matched = _infer_query_policy_types("Refund and warranty terms for laptops")
    assert matched == {"refund", "warranty"}


def _make_doc(policy_type: str, content: str = "") -> Document:
    return Document(
        page_content=content or f"{policy_type} policy text",
        metadata={"policy_type": policy_type, "source": f"{policy_type}_policy.pdf"},
    )


def test_rerank_prefers_matching_policy_type_chunks() -> None:
    candidates = [
        _make_doc("shipping"),
        _make_doc("warranty"),
        _make_doc("refund"),
        _make_doc("shipping"),
    ]
    ranked = _rerank_documents(
        "refund eligibility window",
        candidates,
        top_k=4,
        preferred_policy_types={"refund"},
    )
    assert ranked[0].metadata["policy_type"] == "refund"
    assert len(ranked) == 4


def test_rerank_falls_back_when_no_candidate_matches_preferred_type() -> None:
    candidates = [
        _make_doc("shipping", "delivery times and tracking details"),
        _make_doc("warranty", "warranty coverage and repair details"),
    ]
    ranked = _rerank_documents(
        "refund eligibility",
        candidates,
        top_k=2,
        preferred_policy_types={"refund"},
    )
    assert len(ranked) == 2
    assert {doc.metadata["policy_type"] for doc in ranked} == {"shipping", "warranty"}


def test_rerank_general_query_keeps_semantic_ordering() -> None:
    candidates = [
        _make_doc("shipping"),
        _make_doc("warranty"),
        _make_doc("refund"),
    ]
    ranked = _rerank_documents(
        "company policies",
        candidates,
        top_k=3,
        preferred_policy_types=set(),
    )
    assert [doc.metadata["policy_type"] for doc in ranked] == [
        "shipping",
        "warranty",
        "refund",
    ]


def test_build_policy_type_filter_single_type() -> None:
    assert _build_policy_type_filter({"refund"}) == {"policy_type": "refund"}
    assert _build_policy_type_filter(set()) is None


def test_build_policy_type_filter_mixed_types() -> None:
    assert _build_policy_type_filter({"refund", "warranty"}) == {
        "policy_type": {"$in": ["refund", "warranty"]}
    }


def test_search_policy_documents_uses_metadata_filter_for_refund_query() -> None:
    refund_docs = [_make_doc("refund", f"refund detail {index}") for index in range(3)]
    mock_store = MagicMock()

    def _similarity_search(
        query: str,
        k: int,
        filter: dict[str, str] | None = None,
    ) -> list[Document]:
        if filter == {"policy_type": "refund"}:
            return refund_docs
        return [
            _make_doc("shipping"),
            _make_doc("warranty"),
            _make_doc("refund"),
        ]

    mock_store.similarity_search.side_effect = _similarity_search

    with patch("src.rag.retriever.get_vector_store", return_value=mock_store), patch(
        "src.rag.retriever.has_indexed_documents", return_value=True
    ), patch("src.rag.retriever.rewrite_policy_query", return_value="refund policy"):
        results = search_policy_documents("What is the refund policy?")

    assert isinstance(results, list)
    assert len(results) == 3
    assert {result["policy_type"] for result in results} == {"refund"}
    first_call = mock_store.similarity_search.call_args_list[0]
    assert first_call.kwargs["filter"] == {"policy_type": "refund"}


def test_search_policy_documents_falls_back_when_filtered_search_is_empty() -> None:
    fallback_docs = [
        _make_doc("shipping", "delivery times and tracking details"),
        _make_doc("warranty", "warranty coverage and repair details"),
    ]
    mock_store = MagicMock()
    mock_store.similarity_search.side_effect = [
        [],
        fallback_docs,
    ]

    with patch("src.rag.retriever.get_vector_store", return_value=mock_store), patch(
        "src.rag.retriever.has_indexed_documents", return_value=True
    ), patch("src.rag.retriever.rewrite_policy_query", return_value="refund policy"):
        results = search_policy_documents("What is the refund policy?")

    assert isinstance(results, list)
    assert len(results) == 2
    assert mock_store.similarity_search.call_count == 2
    assert mock_store.similarity_search.call_args_list[0].kwargs["filter"] == {
        "policy_type": "refund"
    }
    assert "filter" not in mock_store.similarity_search.call_args_list[1].kwargs


def test_search_policy_documents_without_index_returns_helpful_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chroma_dir = tmp_path / "chroma_db"
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(chroma_dir))

    from src.config import get_settings

    get_settings.cache_clear()
    reset_vector_store(chroma_dir)

    try:
        assert not has_indexed_documents()
        result = search_policy_documents("What is the refund policy?")
        assert isinstance(result, dict)
        message = result["message"].lower()
        assert "upload" in message or "indexed" in message or "generate" in message
    finally:
        get_settings.cache_clear()
