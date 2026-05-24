"""Tests for specialist agents, Ema Johnson SQL lookups, and response synthesis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.customer_names import match_known_customer
from src.agents.rag_agent import create_rag_agent, run_rag_lookup
from src.agents.response_agent import (
    _ensure_policy_sources,
    create_response_agent,
    synthesize_response,
)
from src.agents.sql_agent import _extract_customer_name, create_sql_agent, run_sql_lookup
from src.mcp_server.server import list_tools
from src.tools.sql_tools import (
    get_customer_by_name,
    get_customer_profile_and_tickets,
    get_high_priority_open_tickets,
    get_open_tickets,
    get_refund_related_tickets,
)

from tests.conftest import DANIEL_SMITH, EMA_JOHNSON, GENERAL_QUERY, PRIYA_PATEL, SQL_QUERY

EXPECTED_MCP_TOOLS = {
    "customer_profile_lookup",
    "customer_ticket_lookup",
    "open_ticket_lookup",
    "refund_ticket_lookup",
    "high_priority_open_ticket_lookup",
    "policy_document_search",
    "policy_question_answer",
    "pdf_ingestion",
}


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "Give me a quick overview of customer Ema's profile and past support ticket details.",
            EMA_JOHNSON,
        ),
        (f"Show me {EMA_JOHNSON}'s open support tickets.", EMA_JOHNSON),
        (
            "Can Ema get a refund based on her past support ticket and the refund policy?",
            EMA_JOHNSON,
        ),
        (f"Show me {EMA_JOHNSON}'s refund-related tickets.", EMA_JOHNSON),
    ],
)
def test_extract_customer_name_from_ema_queries(query: str, expected: str) -> None:
    assert match_known_customer(query) == expected
    assert _extract_customer_name(query) == expected


@pytest.mark.parametrize(
    "factory",
    [create_sql_agent, create_rag_agent, create_response_agent],
    ids=["sql", "rag", "response"],
)
def test_specialist_agent_factories_return_callables(factory) -> None:
    node = factory()
    assert callable(node)


def test_mcp_style_tool_registry_exposes_expected_tools() -> None:
    assert set(list_tools()) == EXPECTED_MCP_TOOLS


def test_get_customer_by_name_returns_ema_profile(seeded_db: Path) -> None:
    customer = get_customer_by_name(EMA_JOHNSON)
    assert customer["customer_id"] == 1
    assert customer["full_name"] == EMA_JOHNSON
    assert customer["customer_tier"] == "Premium"


@pytest.mark.parametrize(
    ("name", "customer_id"),
    [
        (EMA_JOHNSON, 1),
        (DANIEL_SMITH, 2),
        (PRIYA_PATEL, 3),
    ],
)
def test_seeded_demo_customer_names_are_deterministic(
    seeded_db: Path,
    name: str,
    customer_id: int,
) -> None:
    customer = get_customer_by_name(name)
    assert customer["customer_id"] == customer_id
    assert customer["full_name"] == name


def test_get_customer_profile_and_tickets_returns_ema_history(seeded_db: Path) -> None:
    result = get_customer_profile_and_tickets(EMA_JOHNSON)
    assert result["customer"]["customer_id"] == 1
    assert result["ticket_count"] >= 1
    issue_types = {ticket["issue_type"] for ticket in result["tickets"]}
    assert "Refund Request" in issue_types or "Damaged Product" in issue_types


def test_get_open_tickets_returns_ema_open_records(seeded_db: Path) -> None:
    result = get_open_tickets(EMA_JOHNSON)
    assert result["count"] >= 1
    assert all(ticket["status"].lower() == "open" for ticket in result["tickets"])


def test_get_refund_related_tickets_returns_ema_refund_records(seeded_db: Path) -> None:
    result = get_refund_related_tickets(EMA_JOHNSON)
    assert result["count"] >= 1
    for ticket in result["tickets"]:
        text = (ticket["issue_type"] + " " + ticket["description"]).lower()
        assert "refund" in text


def test_get_high_priority_open_tickets_scoped_to_ema(seeded_db: Path) -> None:
    result = get_high_priority_open_tickets(EMA_JOHNSON)
    assert isinstance(result["tickets"], list)
    if result["tickets"]:
        for ticket in result["tickets"]:
            assert ticket["customer_id"] == 1
            assert ticket["status"].lower() == "open"
            assert ticket["priority"].lower() in {"high", "critical"}


def test_sql_lookup_returns_ema_profile_and_tickets(seeded_db: Path) -> None:
    result = run_sql_lookup(SQL_QUERY)
    payload = json.loads(result)
    assert payload["lookup_type"] == "profile_and_tickets"
    assert payload["customer_name"] == EMA_JOHNSON
    assert payload["profile"]["customer_id"] == 1
    assert payload["tickets"]["count"] >= 1


def test_sql_lookup_routes_ema_open_tickets_query(seeded_db: Path) -> None:
    result = run_sql_lookup(f"Show me {EMA_JOHNSON}'s open support tickets.")
    payload = json.loads(result)
    assert payload["lookup_type"] == "open_tickets"
    assert payload["customer_name"] == EMA_JOHNSON
    assert payload["result"]["count"] >= 1


def test_sql_lookup_routes_ema_refund_related_tickets(seeded_db: Path) -> None:
    result = run_sql_lookup(f"Show me {EMA_JOHNSON}'s refund-related tickets.")
    payload = json.loads(result)
    assert payload["lookup_type"] == "refund_related_tickets"
    assert payload["customer_name"] == EMA_JOHNSON
    assert payload["result"]["count"] >= 1


def test_sql_lookup_asks_for_customer_name_when_missing(
    seeded_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")

    from src.config import get_settings

    get_settings.cache_clear()
    try:
        result = run_sql_lookup("Show me the open support tickets.")
    finally:
        get_settings.cache_clear()

    payload = json.loads(result)
    assert payload["lookup_type"] == "open_tickets"
    assert "customer" in payload["message"].lower()
    assert payload["result"]["count"] == 0


def test_rag_lookup_without_documents_returns_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chroma_dir = tmp_path / "chroma_db"
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(chroma_dir))

    from src.config import get_settings
    from src.rag.vector_store import reset_vector_store

    get_settings.cache_clear()
    reset_vector_store(chroma_dir)

    try:
        result = run_rag_lookup("What is the refund policy?")
    finally:
        get_settings.cache_clear()

    assert set(result) >= {
        "answer",
        "sources",
        "rewritten_query",
        "retrieved_context_count",
    }
    assert "upload" in result["answer"].lower() or "indexed" in result["answer"].lower()


def test_response_agent_general_fallback() -> None:
    answer = synthesize_response(GENERAL_QUERY, route="general")
    assert "support assistant" in answer.lower()


def test_response_agent_preserves_policy_sources() -> None:
    rag_context = (
        "Policy answer:\n"
        "Eligible if requested within 30 days.\n\n"
        "Sources: refund_policy.pdf (pages 1, 2)\n\n"
        "Supporting policy excerpts:\n"
        "[1] Source: refund_policy.pdf (page 1)\n"
        "Refunds are available within 30 days."
    )

    answer = _ensure_policy_sources("Ema appears eligible.", rag_context)

    assert "Sources:" in answer
    assert "refund_policy.pdf (pages 1, 2)" in answer
