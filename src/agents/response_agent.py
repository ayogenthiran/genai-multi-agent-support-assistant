"""Response agent: composes the final customer-facing reply.

Merges outputs from the SQL and RAG agents into a single coherent support
message with appropriate tone, source attribution, and next steps.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import get_settings

_GENERAL_CAPABILITY_REPLY = (
    "Hello. I am a customer support assistant for support executives. "
    "I can look up customer profiles and ticket history from the customer database, "
    "answer policy questions using company policy documents, and combine both when needed. "
    "Ask about a customer account, open tickets, refunds, warranties, or cancellations."
)


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


def _source_label(route: str | None, sql_result: str | None, rag_context: str | None) -> str:
    has_sql = bool(sql_result and sql_result.strip())
    has_rag = bool(rag_context and rag_context.strip())

    if route == "both" or (has_sql and has_rag):
        return "customer data and policy documents"
    if route == "sql" or has_sql:
        return "customer data"
    if route == "rag" or has_rag:
        return "policy documents"
    return "general assistant knowledge"


def _fallback_response(
    query: str,
    route: str | None,
    sql_result: str | None,
    rag_context: str | None,
) -> str:
    if route == "general" and not sql_result and not rag_context:
        return _GENERAL_CAPABILITY_REPLY

    source = _source_label(route, sql_result, rag_context)
    sections: list[str] = [
        f"Support summary (based on {source}):",
    ]

    if sql_result and sql_result.strip():
        sections.append("Customer data:")
        sections.append(sql_result.strip())

    if rag_context and rag_context.strip():
        sections.append("Policy guidance:")
        sections.append(rag_context.strip())

    if len(sections) == 1:
        sections.append(
            f"I could not find structured customer or policy information for: {query}"
        )

    return "\n\n".join(sections)


def synthesize_response(
    query: str,
    *,
    route: str | None = None,
    sql_result: str | None = None,
    rag_context: str | None = None,
) -> str:
    """Combine SQL and RAG outputs into one executive-ready answer."""
    cleaned_query = query.strip()
    if route == "general" and not sql_result and not rag_context:
        return _GENERAL_CAPABILITY_REPLY

    settings = get_settings()
    if not settings.openai_api_key:
        return _fallback_response(cleaned_query, route, sql_result, rag_context)

    source = _source_label(route, sql_result, rag_context)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You write concise, professional replies for customer support executives.\n"
                "Use the provided customer data and/or policy guidance when available.\n"
                "Be practical, accurate, and action-oriented.\n"
                "Explicitly state whether the answer is based on customer data, "
                "policy documents, or both.\n"
                "Do not invent facts that are not supported by the provided context.",
            ),
            (
                "human",
                "User question:\n{query}\n\n"
                "Routing label: {route}\n"
                "Expected source basis: {source}\n\n"
                "Customer data context:\n{sql_result}\n\n"
                "Policy context:\n{rag_context}",
            ),
        ]
    )
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )
    response = (prompt | llm).invoke(
        {
            "query": cleaned_query or "No question provided.",
            "route": route or "unknown",
            "source": source,
            "sql_result": sql_result or "None",
            "rag_context": rag_context or "None",
        }
    )
    return str(response.content).strip()


def create_response_agent() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build and return the response synthesis LangGraph node."""

    def response_node(state: dict[str, Any]) -> dict[str, Any]:
        query = _extract_user_query(state)
        route = state.get("route")
        sql_result = state.get("sql_result")
        rag_context = state.get("rag_context")
        final_response = synthesize_response(
            query,
            route=route,
            sql_result=sql_result,
            rag_context=rag_context,
        )
        return {"final_response": final_response}

    return response_node
