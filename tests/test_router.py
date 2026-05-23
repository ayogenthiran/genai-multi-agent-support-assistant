"""Tests for supervisor routing and workflow graph wiring."""

from __future__ import annotations

import pytest

from src.agents.supervisor import classify_query, create_supervisor_agent
from src.graph.workflow import build_support_graph, run_multi_agent_workflow


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is the current refund policy?", "rag"),
        (
            "Give me a quick overview of customer Ema's profile and past support ticket details.",
            "sql",
        ),
        (
            "Can Ema get a refund based on her past support ticket and the refund policy?",
            "both",
        ),
        ("Show me Ema's open support tickets.", "sql"),
        ("Hi, what can you do?", "general"),
    ],
)
def test_classify_query_without_llm(query: str, expected: str) -> None:
    assert classify_query(query, use_llm=False) == expected


def test_supervisor_agent_returns_route() -> None:
    node = create_supervisor_agent()
    result = node({"query": "What is the refund policy?"})
    assert result["route"] == "rag"
    assert result["query"] == "What is the refund policy?"


def test_support_graph_compiles_and_runs_general_query() -> None:
    graph = build_support_graph()
    assert graph is not None

    result = run_multi_agent_workflow("Hi, what can you do?")
    assert result["route"] == "general"
    assert result["final_answer"]
    assert "support assistant" in result["final_answer"].lower()
