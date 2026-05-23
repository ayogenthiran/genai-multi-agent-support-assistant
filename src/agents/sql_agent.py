"""SQL agent: fetches structured customer and ticket data via MCP-style tools.

Uses the MCP tool registry for customer lookup and ticket retrieval, then
returns clean structured information for downstream synthesis.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.agents.customer_names import match_known_customer
from src.config import get_settings
from src.mcp_server.server import call_tool

_NAME_PATTERN = re.compile(
    r"(?:customer|for|about|profile of|tickets for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
    re.IGNORECASE,
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


def _extract_customer_name(query: str) -> str | None:
    """Resolve a customer name from the query using heuristics and optional LLM."""
    cleaned = query.strip()
    if not cleaned:
        return None

    known_match = match_known_customer(cleaned)
    if known_match:
        return known_match

    match = _NAME_PATTERN.search(cleaned)
    if match:
        return " ".join(part.capitalize() for part in match.group(1).split())

    title_case_match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", cleaned)
    if title_case_match:
        return title_case_match.group(1)

    settings = get_settings()
    if not settings.openai_api_key:
        return None

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract the customer full name from the support query. "
                "If no customer name is present, reply with NONE.",
            ),
            ("human", "{query}"),
        ]
    )
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    response = (prompt | llm).invoke({"query": cleaned})
    content = str(response.content).strip()
    if content.upper() == "NONE" or not content:
        return None
    return content


def _wants_open_tickets_only(query: str) -> bool:
    lowered = query.lower()
    return "open" in lowered and "ticket" in lowered


def _format_sql_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)


def run_sql_lookup(query: str) -> str:
    """Execute MCP-style SQL tools for the given natural-language query."""
    customer_name = _extract_customer_name(query)
    if not customer_name:
        return (
            "No customer name was found in the query. "
            "Ask for a customer profile or ticket lookup using the customer's full name."
        )

    if _wants_open_tickets_only(query):
        result = call_tool("open_ticket_lookup", name=customer_name)
        payload = {
            "lookup_type": "open_tickets",
            "customer_name": customer_name,
            "result": result,
        }
        return _format_sql_result(payload)

    profile = call_tool("customer_profile_lookup", name=customer_name)
    tickets = call_tool("customer_ticket_lookup", name=customer_name)
    payload = {
        "lookup_type": "profile_and_tickets",
        "customer_name": customer_name,
        "profile": profile,
        "tickets": tickets,
    }
    customer_id = profile.get("customer_id") if isinstance(profile, dict) else None
    if customer_id is not None:
        payload["customer_id"] = customer_id
    return _format_sql_result(payload)


def create_sql_agent() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build and return the SQL specialist LangGraph node."""

    def sql_node(state: dict[str, Any]) -> dict[str, Any]:
        query = _extract_user_query(state)
        sql_result = run_sql_lookup(query)
        updates: dict[str, Any] = {"sql_result": sql_result}

        try:
            parsed = json.loads(sql_result)
        except json.JSONDecodeError:
            return updates

        customer_id = parsed.get("customer_id")
        if customer_id is None:
            profile = parsed.get("profile")
            if isinstance(profile, dict):
                customer_id = profile.get("customer_id")
        if customer_id is not None:
            updates["customer_id"] = str(customer_id)
        return updates

    return sql_node
