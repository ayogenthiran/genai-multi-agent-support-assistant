"""Supervisor agent: routes user queries to specialist agents.

Analyzes intent (account lookup, policy question, mixed) and decides whether
to invoke the SQL agent, RAG agent, both, or the response agent for general
queries.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.agents.customer_names import match_known_customer
from src.config import get_settings

RouteType = Literal["sql", "rag", "both", "general"]

SQL_KEYWORDS: tuple[str, ...] = (
    "customer",
    "profile",
    "ticket",
    "tickets",
    "order",
    "orders",
    "account",
    "history",
    "purchase",
    "billing",
    "subscription",
    "open issue",
    "support case",
)

RAG_KEYWORDS: tuple[str, ...] = (
    "policy",
    "refund",
    "warranty",
    "cancellation",
    "cancel",
    "document",
    "documents",
    "faq",
    "terms",
    "return",
    "exchange",
    "guarantee",
    "coverage",
)

GENERAL_KEYWORDS: tuple[str, ...] = (
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "what can you do",
    "what do you do",
    "help me",
    "capabilities",
    "who are you",
    "how do you work",
    "thanks",
    "thank you",
)

_ROUTE_PATTERN = re.compile(r"\b(sql|rag|both|general)\b", re.IGNORECASE)
_NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    normalized = _normalize(text)
    return sum(1 for keyword in keywords if keyword in normalized)


def _mentions_customer(text: str) -> bool:
    if match_known_customer(text):
        return True
    return bool(_NAME_PATTERN.search(text))


def classify_query(query: str, *, use_llm: bool = True) -> RouteType:
    """Classify a user query into sql, rag, both, or general routing."""
    cleaned = query.strip()
    if not cleaned:
        return "general"

    sql_hits = _keyword_hits(cleaned, SQL_KEYWORDS)
    if _mentions_customer(cleaned):
        sql_hits += 1
    rag_hits = _keyword_hits(cleaned, RAG_KEYWORDS)
    general_hits = _keyword_hits(cleaned, GENERAL_KEYWORDS)

    if sql_hits and rag_hits:
        return "both"
    if sql_hits:
        return "sql"
    if rag_hits:
        return "rag"
    if general_hits and sql_hits == 0 and rag_hits == 0:
        return "general"

    if use_llm:
        settings = get_settings()
        if settings.openai_api_key:
            return _llm_classify_query(cleaned)

    return "general"


def _llm_classify_query(query: str) -> RouteType:
    """Use the configured LLM when keyword routing is inconclusive."""
    settings = get_settings()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You route customer support queries for a support executive assistant.\n"
                "Return exactly one label: sql, rag, both, or general.\n"
                "- sql: customer/profile/ticket/order/account/history lookups\n"
                "- rag: policy/refund/warranty/cancellation/document questions\n"
                "- both: mixed questions needing customer data and policy guidance\n"
                "- general: greetings, capability questions, or small talk\n"
                "Reply with only the label.",
            ),
            ("human", "{query}"),
        ]
    )
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    response = (prompt | llm).invoke({"query": query})
    content = str(response.content).strip().lower()
    match = _ROUTE_PATTERN.search(content)
    if match:
        return match.group(1).lower()  # type: ignore[return-value]
    return "general"


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


def create_supervisor_agent() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build and return the supervisor LangGraph node."""

    def supervisor_node(state: dict[str, Any]) -> dict[str, Any]:
        query = _extract_user_query(state)
        route = classify_query(query)
        return {"route": route, "query": query}

    return supervisor_node
