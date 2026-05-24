"""Tests for supervisor routing and workflow graph wiring."""

from __future__ import annotations

import pytest

from src.agents.supervisor import classify_query, create_supervisor_agent
from src.graph.workflow import build_support_graph, run_multi_agent_workflow

from tests.conftest import GENERAL_QUERY, MIXED_QUERY, RAG_QUERY, SQL_QUERY


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (SQL_QUERY, "sql"),
        (RAG_QUERY, "rag"),
        (MIXED_QUERY, "both"),
        (GENERAL_QUERY, "general"),
        ("Show me Ema Johnson's open support tickets.", "sql"),
        ("Show me Ema Johnson's refund-related tickets.", "sql"),
        ("What does the warranty policy say?", "rag"),
    ],
)
def test_classify_query_without_llm(query: str, expected: str) -> None:
    assert classify_query(query, use_llm=False) == expected


def test_supervisor_routes_sql_query() -> None:
    node = create_supervisor_agent()
    result = node({"query": SQL_QUERY})
    assert result["route"] == "sql"
    assert result["query"] == SQL_QUERY


def test_supervisor_routes_rag_query() -> None:
    node = create_supervisor_agent()
    result = node({"query": RAG_QUERY})
    assert result["route"] == "rag"
    assert result["query"] == RAG_QUERY


def test_supervisor_routes_mixed_sql_and_rag_query() -> None:
    node = create_supervisor_agent()
    result = node({"query": MIXED_QUERY})
    assert result["route"] == "both"
    assert result["query"] == MIXED_QUERY


def test_supervisor_routes_general_query() -> None:
    node = create_supervisor_agent()
    result = node({"query": GENERAL_QUERY})
    assert result["route"] == "general"
    assert result["query"] == GENERAL_QUERY


def test_support_graph_compiles_and_runs_general_query() -> None:
    graph = build_support_graph()
    assert graph is not None

    result = run_multi_agent_workflow(GENERAL_QUERY)
    assert result["route"] == "general"
    assert result["final_answer"]
    assert "support assistant" in result["final_answer"].lower()
