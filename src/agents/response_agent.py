"""Response agent that synthesizes SQL and RAG outputs."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.prompts.system_prompts import (
    GENERAL_CAPABILITY_REPLY,
    RESPONSE_SYNTHESIS_PROMPT,
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


def _parse_sql_payload(sql_result: str | None) -> dict[str, Any] | None:
    if not sql_result or not sql_result.strip():
        return None
    try:
        payload = json.loads(sql_result)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _as_ticket_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [ticket for ticket in value if isinstance(ticket, dict)]


def _ticket_sort_key(ticket: dict[str, Any]) -> tuple[int, int, str]:
    priority_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    status_rank = {"open": 0, "in progress": 1, "escalated": 2, "resolved": 3}
    priority = str(ticket.get("priority", "")).strip().lower()
    status = str(ticket.get("status", "")).strip().lower()
    created_at = str(ticket.get("created_at", ""))
    return (
        priority_rank.get(priority, 4),
        status_rank.get(status, 4),
        # ISO dates sort lexically; invert characters so newer dates appear first.
        "".join(chr(255 - ord(char)) for char in created_at),
    )


def _format_count(label: str, count: int) -> str:
    return f"- {label}: {count}"


def _extract_no_customer_name(message: Any) -> str | None:
    text = str(message or "").strip()
    match = re.match(
        r"^No customer found with(?: the)? name ['\"]?(.+?)['\"]?\.$",
        text,
        flags=re.I,
    )
    return match.group(1).strip() if match else None


def _format_no_customer_response(payload: dict[str, Any]) -> str | None:
    candidate_messages = [payload.get("message")]

    for key in ("profile", "tickets", "result"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidate_messages.append(value.get("message"))

    for message in candidate_messages:
        name = _extract_no_customer_name(message)
        if name:
            return (
                f"No customer found with the name {name}. "
                "No support tickets were returned."
            )

    return None


def _ticket_label(ticket: dict[str, Any]) -> str:
    issue_type = str(ticket.get("issue_type") or "Support issue").strip()
    status = str(ticket.get("status") or "Unknown").strip()
    priority = str(ticket.get("priority") or "Unknown").strip()
    description = str(ticket.get("description") or "No description provided.").strip()
    resolution = ticket.get("resolution")
    resolution_text = (
        str(resolution).strip()
        if resolution is not None and str(resolution).strip()
        else "Not yet available."
    )
    return (
        f"* {issue_type}  \n"
        f"  Status: {status} | Priority: {priority}  \n"
        f"  Description: {description}  \n"
        f"  Resolution: {resolution_text}"
    )


def _compact_ticket_description(description: Any) -> str:
    text = str(description or "No description provided.").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+[—-]\s+needs urgent refund review\.?$", "", text, flags=re.I)
    text = re.sub(
        r"\bfor the ([A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)+ replacement)\b",
        r"for \1",
        text,
    )
    text = re.sub(r"\bfor the (.+?) order\.?$", r"for the \1.", text, flags=re.I)
    if text:
        text = text[0].lower() + text[1:]
    return text if text.endswith(".") else f"{text}."


def _compact_ticket_label(ticket: dict[str, Any]) -> str:
    issue_type = str(ticket.get("issue_type") or "Support issue").strip()
    status = str(ticket.get("status") or "Unknown").strip()
    priority = str(ticket.get("priority") or "Unknown").strip()
    description = _compact_ticket_description(ticket.get("description"))
    return f"- {issue_type}: {status}, {priority} priority — {description}"


def _extract_sql_profile_and_tickets(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    lookup_type = str(payload.get("lookup_type") or "")
    message = payload.get("message")
    profile: dict[str, Any] = {}
    tickets: list[dict[str, Any]] = []

    if lookup_type == "profile_and_tickets":
        raw_profile = payload.get("profile")
        if isinstance(raw_profile, dict):
            profile = raw_profile

        raw_tickets = payload.get("tickets")
        if isinstance(raw_tickets, dict):
            tickets = _as_ticket_list(raw_tickets.get("tickets"))
            message = message or raw_tickets.get("message")
        elif isinstance(raw_tickets, list):
            tickets = _as_ticket_list(raw_tickets)
    else:
        result = payload.get("result")
        if isinstance(result, dict):
            tickets = _as_ticket_list(result.get("tickets"))
            message = message or result.get("message")
            raw_profile = result.get("customer") or result.get("profile")
            if isinstance(raw_profile, dict):
                profile = raw_profile

    if not profile and payload.get("customer_name"):
        profile = {"full_name": payload.get("customer_name")}

    return profile, tickets, str(message).strip() if message else None


def _is_refund_relevant_lookup(payload: dict[str, Any]) -> bool:
    return str(payload.get("lookup_type") or "") == "refund_related_tickets"


def _format_customer_overview(profile: dict[str, Any]) -> list[str]:
    if not profile:
        return []

    fields = [
        ("Full name", profile.get("full_name") or profile.get("customer_name")),
        ("Customer tier", profile.get("customer_tier")),
        ("Status", profile.get("status")),
        ("Location", profile.get("location")),
        ("Signup date", profile.get("signup_date")),
    ]
    lines = ["Customer overview:"]
    lines.extend(f"- {label}: {value}" for label, value in fields if value)
    return lines


def _format_sql_support_summary(sql_result: str | None) -> str | None:
    payload = _parse_sql_payload(sql_result)
    if payload is None:
        return None

    no_customer_response = _format_no_customer_response(payload)
    if no_customer_response:
        return no_customer_response

    profile, tickets, message = _extract_sql_profile_and_tickets(payload)
    sorted_tickets = sorted(tickets, key=_ticket_sort_key)
    total_count = len(sorted_tickets)
    open_count = sum(
        1 for ticket in sorted_tickets if str(ticket.get("status", "")).lower() == "open"
    )
    resolved_count = sum(
        1
        for ticket in sorted_tickets
        if str(ticket.get("status", "")).lower() == "resolved"
    )

    sections: list[str] = []

    overview = _format_customer_overview(profile)
    if overview:
        sections.append("\n".join(overview))

    if _is_refund_relevant_lookup(payload):
        if sorted_tickets:
            ticket_lines = ["Relevant support history:"]
            ticket_lines.extend(_compact_ticket_label(ticket) for ticket in sorted_tickets)
            sections.append("\n".join(ticket_lines))
        else:
            sections.append("Relevant support history:\n- No refund-relevant tickets were returned.")
        if message:
            sections.append(f"Note: {message}")
        return "\n\n".join(sections)

    summary_lines = [
        "Ticket summary:",
        _format_count("Total tickets returned", total_count),
        _format_count("Open tickets", open_count),
        _format_count("Resolved tickets", resolved_count),
    ]
    if message:
        summary_lines.append(f"- Note: {message}")
    sections.append("\n".join(summary_lines))

    if sorted_tickets:
        ticket_details = "\n\n".join(
            _ticket_label(ticket) for ticket in sorted_tickets
        )
        sections.append(f"Ticket details:\n\n{ticket_details}")
    else:
        sections.append("Ticket details:\n- No matching tickets were returned.")

    return "\n\n".join(sections)


def _fallback_response(
    query: str,
    route: str | None,
    sql_result: str | None,
    rag_context: str | None,
) -> str:
    if route == "general" and not sql_result and not rag_context:
        return GENERAL_CAPABILITY_REPLY

    source = _source_label(route, sql_result, rag_context)
    heading = (
        "Customer support summary:"
        if source == "customer data"
        else f"Support summary (based on {source}):"
    )
    sections: list[str] = [
        heading,
    ]

    sql_summary = _format_sql_support_summary(sql_result)
    if sql_summary:
        sections.append(sql_summary)
    elif sql_result and sql_result.strip():
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


def _extract_policy_sources(rag_context: str | None) -> str:
    if not rag_context:
        return ""

    marker = "\nSources:"
    marker_index = rag_context.find(marker)
    if marker_index == -1:
        return ""

    sources = rag_context[marker_index + 1 :]
    end_marker = "\n\nSupporting policy excerpts:"
    end_index = sources.find(end_marker)
    if end_index != -1:
        sources = sources[:end_index]

    source_lines = [line.strip() for line in sources.splitlines() if line.strip()]
    if not source_lines:
        return ""
    return "\n".join(source_lines)


def _ensure_policy_sources(answer: str, rag_context: str | None) -> str:
    sources = _extract_policy_sources(rag_context)
    if not sources:
        return answer

    lowered = answer.lower()
    if "sources:" in lowered or "sources used" in lowered:
        return answer

    return f"{answer.rstrip()}\n\n{sources}"


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
        return GENERAL_CAPABILITY_REPLY

    sql_summary = _format_sql_support_summary(sql_result)
    if sql_summary and not rag_context:
        return sql_summary

    settings = get_settings()
    if not settings.openai_api_key:
        return _fallback_response(cleaned_query, route, sql_result, rag_context)

    source = _source_label(route, sql_result, rag_context)
    try:
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.2,
        )
        response = (RESPONSE_SYNTHESIS_PROMPT | llm).invoke(
            {
                "query": cleaned_query or "No question provided.",
                "route": route or "unknown",
                "source": source,
                "sql_result": sql_summary or sql_result or "None",
                "rag_context": rag_context or "None",
            }
        )
    except Exception:
        # LLM unavailable mid-demo: render the deterministic fallback so
        # already-fetched SQL/RAG context still reaches the user.
        return _fallback_response(cleaned_query, route, sql_result, rag_context)
    return _ensure_policy_sources(str(response.content).strip(), rag_context)


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
