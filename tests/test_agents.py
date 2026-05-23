"""Tests for specialist agent scaffolding and SQL lookups."""

from __future__ import annotations

import json

import pytest

from src.agents.customer_names import match_known_customer
from src.agents.rag_agent import create_rag_agent, run_rag_lookup
from src.agents.response_agent import create_response_agent, synthesize_response
from src.agents.sql_agent import _extract_customer_name, create_sql_agent, run_sql_lookup
from src.create_dummy_data import create_dummy_data


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "Give me a quick overview of customer Ema's profile and past support ticket details.",
            "Ema Johnson",
        ),
        ("Show me Daniel's open support tickets.", "Daniel Smith"),
        (
            "Can Ema get a refund based on her past support ticket and the refund policy?",
            "Ema Johnson",
        ),
    ],
)
def test_extract_customer_name_from_possessive_queries(
    query: str,
    expected: str,
) -> None:
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


def test_sql_lookup_returns_customer_data(tmp_path) -> None:
    db_path = create_dummy_data(tmp_path / "customers.db")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))

    from src.config import get_settings

    get_settings.cache_clear()

    try:
        result = run_sql_lookup("Show me Ema Johnson's profile and tickets.")
        payload = json.loads(result)
        assert payload["customer_name"] == "Ema Johnson"
        assert payload["profile"]["customer_id"] == 1
        assert payload["tickets"]["ticket_count"] >= 1
    finally:
        get_settings.cache_clear()
        monkeypatch.undo()


def test_rag_lookup_without_documents_returns_guidance() -> None:
    message = run_rag_lookup("What is the refund policy?")
    assert "upload" in message.lower() or "indexed" in message.lower()


def test_response_agent_general_fallback() -> None:
    answer = synthesize_response("Hi, what can you do?", route="general")
    assert "support assistant" in answer.lower()
