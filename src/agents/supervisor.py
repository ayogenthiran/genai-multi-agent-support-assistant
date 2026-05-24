"""Supervisor agent that routes queries to SQL, RAG, or response nodes."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from langchain_openai import ChatOpenAI

from src.agents.customer_names import match_known_customer
from src.config import get_settings
from src.prompts.system_prompts import SUPERVISOR_ROUTING_PROMPT

RouteType = Literal["sql", "rag", "both", "general"]

SQL_KEYWORDS: tuple[str, ...] = (
    "customer",
    "profile",
    "ticket",
    "tickets",
    "account",
    "history",
    "open issue",
    "support case",
    "high priority",
    "high-priority",
)

# Words that signal the question is about a *ticket* record rather than a
# policy lookup. When these dominate the query, we prefer the SQL route even
# if a generic policy-domain word like "refund" appears.
SQL_BIAS_KEYWORDS: tuple[str, ...] = (
    "ticket",
    "tickets",
    "open ticket",
    "open tickets",
    "support case",
    "support history",
)

RAG_KEYWORDS: tuple[str, ...] = (
    "policy",
    "policies",
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

# Words that explicitly invoke a policy document, used to keep mixed
# refund-eligibility questions routed to "both".
POLICY_DOC_KEYWORDS: tuple[str, ...] = (
    "policy",
    "policies",
    "warranty",
    "document",
    "documents",
    "faq",
    "terms",
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

    normalized = _normalize(cleaned)
    sql_hits = _keyword_hits(cleaned, SQL_KEYWORDS)
    if _mentions_customer(cleaned):
        sql_hits += 1
    rag_hits = _keyword_hits(cleaned, RAG_KEYWORDS)
    general_hits = _keyword_hits(cleaned, GENERAL_KEYWORDS)

    has_sql_bias = any(keyword in normalized for keyword in SQL_BIAS_KEYWORDS)
    has_policy_doc_cue = any(keyword in normalized for keyword in POLICY_DOC_KEYWORDS)

    if sql_hits and rag_hits:
        if has_policy_doc_cue:
            return "both"
        if has_sql_bias:
            return "sql"
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
            try:
                return _llm_classify_query(cleaned)
            except Exception:
                # LLM unavailable (network, auth, quota): fall back to "general"
                # so the workflow still produces a response.
                return "general"

    return "general"


def _llm_classify_query(query: str) -> RouteType:
    """Use the configured LLM when keyword routing is inconclusive."""
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    response = (SUPERVISOR_ROUTING_PROMPT | llm).invoke({"query": query})
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
