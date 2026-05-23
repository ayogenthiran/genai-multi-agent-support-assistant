"""RAG agent: answers questions using retrieved policy and FAQ documents.

Calls MCP-style policy tools to search documents and produce grounded answers
for support executives.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.mcp_server.server import call_tool


def _extract_user_query(state: dict[str, Any]) -> str:
    messages = state.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, dict):
            role = message.get("role") or message.get("type")
            if role in {"user", "human"}:
                return str(message.get("content", "")).strip()
            continue
        message_type = getattr(message, "type", None)
        if message_type in {"human", "user"}:
            return str(getattr(message, "content", "")).strip()
    return str(state.get("query", "")).strip()


def run_rag_lookup(query: str) -> str:
    """Run policy document search and question answering via MCP-style tools."""
    cleaned = query.strip()
    if not cleaned:
        return "Please provide a policy-related question."

    answer = call_tool("policy_question_answer", query=cleaned)
    passages = call_tool("policy_document_search", query=cleaned)
    return (
        "Policy answer:\n"
        f"{answer}\n\n"
        "Supporting policy excerpts:\n"
        f"{passages}"
    )


def create_rag_agent() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build and return the RAG specialist LangGraph node."""

    def rag_node(state: dict[str, Any]) -> dict[str, Any]:
        query = _extract_user_query(state)
        rag_context = run_rag_lookup(query)
        return {"rag_context": rag_context}

    return rag_node
