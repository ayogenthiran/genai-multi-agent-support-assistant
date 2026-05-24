"""SQL tools for querying the customer support database.

Exposes reusable query functions and LangChain @tool wrappers for customer
lookup and ticket retrieval against the configured SQLite database.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from src.config import get_settings
from src.database import get_session

_CUSTOMER_BY_NAME_SQL = """
SELECT customer_id, full_name, email, phone, customer_tier, signup_date, location, status
FROM customers
WHERE LOWER(TRIM(full_name)) = LOWER(TRIM(:name))
"""

_TICKETS_BY_CUSTOMER_SQL = """
SELECT ticket_id, customer_id, issue_type, status, priority,
       created_at, description, resolution, agent_name
FROM support_tickets
WHERE customer_id = :customer_id
ORDER BY created_at DESC
"""

_OPEN_TICKETS_BY_NAME_SQL = """
SELECT t.ticket_id, t.customer_id, t.issue_type, t.status, t.priority,
       t.created_at, t.description, t.resolution, t.agent_name
FROM support_tickets t
JOIN customers c ON c.customer_id = t.customer_id
WHERE LOWER(TRIM(c.full_name)) = LOWER(TRIM(:name))
  AND LOWER(t.status) = 'open'
ORDER BY t.created_at DESC
"""

_REFUND_TICKETS_BY_NAME_SQL = """
SELECT t.ticket_id, t.customer_id, t.issue_type, t.status, t.priority,
       t.created_at, t.description, t.resolution, t.agent_name
FROM support_tickets t
JOIN customers c ON c.customer_id = t.customer_id
WHERE LOWER(TRIM(c.full_name)) = LOWER(TRIM(:name))
  AND (
       LOWER(t.issue_type) LIKE '%refund%'
    OR LOWER(t.description) LIKE '%refund%'
  )
ORDER BY t.created_at DESC
"""

_HIGH_PRIORITY_OPEN_TICKETS_SQL = """
SELECT t.ticket_id, t.customer_id, c.full_name AS customer_name,
       t.issue_type, t.status, t.priority,
       t.created_at, t.description, t.agent_name
FROM support_tickets t
JOIN customers c ON c.customer_id = t.customer_id
WHERE LOWER(t.status) = 'open'
  AND LOWER(t.priority) IN ('high', 'critical')
ORDER BY t.created_at DESC
"""

_HIGH_PRIORITY_OPEN_TICKETS_FOR_NAME_SQL = """
SELECT t.ticket_id, t.customer_id, c.full_name AS customer_name,
       t.issue_type, t.status, t.priority,
       t.created_at, t.description, t.agent_name
FROM support_tickets t
JOIN customers c ON c.customer_id = t.customer_id
WHERE LOWER(TRIM(c.full_name)) = LOWER(TRIM(:name))
  AND LOWER(t.status) = 'open'
  AND LOWER(t.priority) IN ('high', 'critical')
ORDER BY t.created_at DESC
"""


def _rows_to_dicts(rows: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _empty_name_response(message: str = "Please provide a customer name.") -> dict[str, Any]:
    return {"tickets": [], "count": 0, "message": message}


def _run_named_ticket_query(name: str, query: str, *, label: str) -> dict[str, Any]:
    """Run a parameterized ticket SQL query keyed by customer name.

    Returns a clean dict with ``tickets``, ``count`` and an optional ``message``.
    The query must accept a single ``:name`` parameter and project the
    ticket columns selected by the SQL constants above.
    """
    cleaned_name = name.strip()
    if not cleaned_name:
        return _empty_name_response()

    customer = get_customer_by_name(cleaned_name)
    if "customer_id" not in customer:
        return {
            "tickets": [],
            "count": 0,
            "message": customer.get(
                "message", f"No customer found with name '{cleaned_name}'."
            ),
        }

    settings = get_settings()
    try:
        with get_session(settings.sqlite_db_path) as session:
            result = session.execute(text(query), {"name": cleaned_name})
            tickets = _rows_to_dicts(result.mappings().all())
    except Exception as exc:
        return {
            "tickets": [],
            "count": 0,
            "message": f"Database error while fetching {label} tickets: {exc}",
        }

    if not tickets:
        return {
            "tickets": [],
            "count": 0,
            "message": f"No {label} tickets found for customer '{cleaned_name}'.",
        }

    return {"tickets": tickets, "count": len(tickets)}


def get_customer_by_name(name: str) -> dict[str, Any]:
    """Look up a customer by name (case-insensitive, trimmed).

    Returns the customer row as a dict, or ``{"message": ...}`` when the
    customer is not found or a database error occurs.
    """
    cleaned_name = name.strip()
    if not cleaned_name:
        return {"message": "Please provide a customer name."}

    settings = get_settings()
    try:
        with get_session(settings.sqlite_db_path) as session:
            result = session.execute(
                text(_CUSTOMER_BY_NAME_SQL),
                {"name": cleaned_name},
            )
            row = result.mappings().first()
    except Exception as exc:
        return {"message": f"Database error while looking up customer: {exc}"}

    if row is None:
        return {"message": f"No customer found with name '{cleaned_name}'."}

    return dict(row)


def get_customer_tickets(customer_id: int) -> dict[str, Any]:
    """Return all support tickets for a given customer ID."""
    settings = get_settings()
    try:
        with get_session(settings.sqlite_db_path) as session:
            result = session.execute(
                text(_TICKETS_BY_CUSTOMER_SQL),
                {"customer_id": customer_id},
            )
            tickets = _rows_to_dicts(result.mappings().all())
    except Exception as exc:
        return {
            "tickets": [],
            "count": 0,
            "message": f"Database error while fetching tickets: {exc}",
        }

    if not tickets:
        return {
            "tickets": [],
            "count": 0,
            "message": f"No support tickets found for customer_id {customer_id}.",
        }

    return {"tickets": tickets, "count": len(tickets)}


def get_customer_profile_and_tickets(name: str) -> dict[str, Any]:
    """Return a customer's profile and full support ticket history."""
    customer = get_customer_by_name(name)
    if "customer_id" not in customer:
        return {
            "customer": None,
            "tickets": [],
            "ticket_count": 0,
            "message": customer.get(
                "message", f"No customer found with name '{name.strip()}'."
            ),
        }

    ticket_result = get_customer_tickets(customer["customer_id"])
    response: dict[str, Any] = {
        "customer": customer,
        "tickets": ticket_result.get("tickets", []),
        "ticket_count": ticket_result.get("count", 0),
    }
    if "message" in ticket_result:
        response["message"] = ticket_result["message"]
    return response


def get_open_tickets(name: str) -> dict[str, Any]:
    """Return open support tickets for a customer identified by name."""
    return _run_named_ticket_query(name, _OPEN_TICKETS_BY_NAME_SQL, label="open")


def get_refund_related_tickets(name: str) -> dict[str, Any]:
    """Return refund-related support tickets for a customer identified by name."""
    return _run_named_ticket_query(name, _REFUND_TICKETS_BY_NAME_SQL, label="refund-related")


def get_high_priority_open_tickets(name: str | None = None) -> dict[str, Any]:
    """Return open tickets whose priority is High or Critical.

    If ``name`` is provided, results are limited to that customer; otherwise
    high-priority open tickets are returned across all customers.
    """
    cleaned_name = (name or "").strip()

    if cleaned_name:
        customer = get_customer_by_name(cleaned_name)
        if "customer_id" not in customer:
            return {
                "tickets": [],
                "count": 0,
                "message": customer.get(
                    "message", f"No customer found with name '{cleaned_name}'."
                ),
            }
        query = _HIGH_PRIORITY_OPEN_TICKETS_FOR_NAME_SQL
        params: dict[str, Any] = {"name": cleaned_name}
        scope_label = f"customer '{cleaned_name}'"
    else:
        query = _HIGH_PRIORITY_OPEN_TICKETS_SQL
        params = {}
        scope_label = "any customer"

    settings = get_settings()
    try:
        with get_session(settings.sqlite_db_path) as session:
            result = session.execute(text(query), params)
            tickets = _rows_to_dicts(result.mappings().all())
    except Exception as exc:
        return {
            "tickets": [],
            "count": 0,
            "message": f"Database error while fetching high-priority open tickets: {exc}",
        }

    if not tickets:
        return {
            "tickets": [],
            "count": 0,
            "message": f"No high-priority open tickets found for {scope_label}.",
        }

    return {"tickets": tickets, "count": len(tickets)}


def get_sql_tools() -> list[Any]:
    """Return LangChain ``@tool`` wrappers around the predefined SQL functions."""
    from langchain_core.tools import tool

    @tool
    def lookup_customer_by_name(name: str) -> dict[str, Any]:
        """Look up a customer account by name."""
        return get_customer_by_name(name)

    @tool
    def lookup_customer_tickets(customer_id: int) -> dict[str, Any]:
        """List all support tickets for a customer ID."""
        return get_customer_tickets(customer_id)

    @tool
    def lookup_customer_profile_and_tickets(name: str) -> dict[str, Any]:
        """Get a customer's profile and full ticket history by name."""
        return get_customer_profile_and_tickets(name)

    @tool
    def lookup_open_tickets(name: str) -> dict[str, Any]:
        """List open support tickets for a customer by name."""
        return get_open_tickets(name)

    @tool
    def lookup_refund_related_tickets(name: str) -> dict[str, Any]:
        """List refund-related support tickets for a customer by name."""
        return get_refund_related_tickets(name)

    @tool
    def lookup_high_priority_open_tickets(name: str | None = None) -> dict[str, Any]:
        """List high-priority open tickets, optionally filtered by customer name."""
        return get_high_priority_open_tickets(name)

    return [
        lookup_customer_by_name,
        lookup_customer_tickets,
        lookup_customer_profile_and_tickets,
        lookup_open_tickets,
        lookup_refund_related_tickets,
        lookup_high_priority_open_tickets,
    ]
