"""RAG agent that answers policy questions using indexed documents."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.mcp_server.server import call_tool
from src.prompts.system_prompts import RAG_LOOKUP_RESPONSE_TEMPLATE
from src.tools.document_tools import format_policy_answer


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


def _empty_rag_result(answer: str, rewritten_query: str = "") -> dict[str, Any]:
    return {
        "answer": answer,
        "sources": [],
        "rewritten_query": rewritten_query,
        "retrieved_context_count": 0,
    }


def _format_rag_context(answer_payload: dict[str, Any], passages: str) -> str:
    return RAG_LOOKUP_RESPONSE_TEMPLATE.format(
        formatted_answer=format_policy_answer(answer_payload),
        passages=passages,
    )


def run_rag_lookup(query: str) -> dict[str, Any]:
    """Run policy Q&A and document search via MCP-style tools."""
    cleaned = query.strip()
    if not cleaned:
        return _empty_rag_result("Please provide a policy-related question.")

    answer_payload = call_tool("policy_question_answer", query=cleaned)
    passages = call_tool("policy_document_search", query=cleaned)
    rag_result = {
        "answer": str(answer_payload.get("answer", "")).strip(),
        "sources": list(answer_payload.get("sources") or []),
        "rewritten_query": str(answer_payload.get("rewritten_query", "")).strip(),
        "retrieved_context_count": int(
            answer_payload.get("retrieved_context_count", 0) or 0,
        ),
    }
    rag_result["supporting_passages"] = passages
    return rag_result


def create_rag_agent() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build and return the RAG specialist LangGraph node."""

    def rag_node(state: dict[str, Any]) -> dict[str, Any]:
        query = _extract_user_query(state)
        rag_result = run_rag_lookup(query)
        passages = str(rag_result.get("supporting_passages", "")).strip()
        rag_context = _format_rag_context(rag_result, passages)
        return {"rag_result": rag_result, "rag_context": rag_context}

    return rag_node
