"""SQL agent that maps support questions to predefined database tools."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from langchain_openai import ChatOpenAI

from src.agents.customer_names import match_known_customer
from src.config import get_settings
from src.mcp_server.server import call_tool
from src.prompts.system_prompts import SQL_CUSTOMER_NAME_EXTRACTION_PROMPT

_NAME_PATTERN = re.compile(
    r"(?:customer|for|about|profile of|tickets for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
    re.IGNORECASE,
)

_NAME_REQUIRED_MESSAGE = (
    "I could not find a customer name in that question. "
    "Please include the customer's full name (e.g. 'Ema Johnson') so I can "
    "look up their profile or support tickets."
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


def _extract_customer_name_heuristic(query: str) -> str | None:
    """Resolve a customer name from the query using fast offline heuristics."""
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

    return None


def _extract_customer_name(query: str) -> str | None:
    """Resolve a customer name from the query using heuristics and optional LLM."""
    cleaned = query.strip()
    if not cleaned:
        return None

    heuristic = _extract_customer_name_heuristic(cleaned)
    if heuristic:
        return heuristic

    settings = get_settings()
    if not settings.openai_api_key:
        return None

    try:
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        response = (SQL_CUSTOMER_NAME_EXTRACTION_PROMPT | llm).invoke({"query": cleaned})
    except Exception:
        # LLM unavailable: skip extraction and let the caller surface the
        # standard "please provide a customer name" message.
        return None
    content = str(response.content).strip()
    if content.upper() == "NONE" or not content:
        return None
    return content


def _classify_sql_intent(query: str) -> str:
    """Pick which predefined SQL lookup best matches the user query."""
    lowered = query.lower()
    mentions_ticket = "ticket" in lowered or "tickets" in lowered

    if ("high" in lowered or "critical" in lowered) and (
        "priority" in lowered or mentions_ticket
    ):
        return "high_priority_open_tickets"

    if "refund" in lowered and (mentions_ticket or "history" in lowered):
        return "refund_related_tickets"

    if mentions_ticket and ("open" in lowered or "unresolved" in lowered):
        return "open_tickets"

    return "profile_and_tickets"


def _format_sql_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)


def _name_required_payload(lookup_type: str, query: str) -> dict[str, Any]:
    return {
        "lookup_type": lookup_type,
        "query": query,
        "result": {"tickets": [], "count": 0, "message": _NAME_REQUIRED_MESSAGE},
        "message": _NAME_REQUIRED_MESSAGE,
    }


def run_sql_lookup(query: str) -> str:
    """Execute the matching predefined SQL tool and return JSON."""
    cleaned_query = query.strip()
    intent = _classify_sql_intent(cleaned_query)

    if intent == "high_priority_open_tickets":
        # The customer name is optional for this lookup, so use the cheap
        # heuristic only — avoid spending an LLM call when none is needed.
        scoped_name = _extract_customer_name_heuristic(cleaned_query)
        result = call_tool("high_priority_open_ticket_lookup", name=scoped_name)
        payload: dict[str, Any] = {
            "lookup_type": "high_priority_open_tickets",
            "customer_name": scoped_name,
            "result": result,
        }
        return _format_sql_result(payload)

    customer_name = _extract_customer_name(cleaned_query)
    if not customer_name:
        return _format_sql_result(_name_required_payload(intent, cleaned_query))

    if intent == "open_tickets":
        result = call_tool("open_ticket_lookup", name=customer_name)
        payload = {
            "lookup_type": "open_tickets",
            "customer_name": customer_name,
            "result": result,
        }
        return _format_sql_result(payload)

    if intent == "refund_related_tickets":
        result = call_tool("refund_ticket_lookup", name=customer_name)
        payload = {
            "lookup_type": "refund_related_tickets",
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


def _extract_customer_id_from_payload(payload: dict[str, Any]) -> int | str | None:
    """Pull a customer_id out of the SQL agent payload when one is available."""
    customer_id = payload.get("customer_id")
    if customer_id is not None:
        return customer_id

    profile = payload.get("profile")
    if isinstance(profile, dict):
        return profile.get("customer_id")

    result = payload.get("result")
    if isinstance(result, dict):
        if result.get("customer_id") is not None:
            return result["customer_id"]
        tickets = result.get("tickets")
        if isinstance(tickets, list) and tickets:
            first = tickets[0]
            if isinstance(first, dict):
                return first.get("customer_id")

    return None


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

        customer_id = _extract_customer_id_from_payload(parsed)
        if customer_id is not None:
            updates["customer_id"] = str(customer_id)
        return updates

    return sql_node
