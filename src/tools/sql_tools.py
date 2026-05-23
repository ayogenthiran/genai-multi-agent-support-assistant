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
SELECT customer_id, name, email, phone, customer_type, join_date, location
FROM customers
WHERE LOWER(TRIM(name)) = LOWER(TRIM(:name))
"""

_TICKETS_BY_CUSTOMER_SQL = """
SELECT ticket_id, customer_id, issue_type, status, priority,
       created_date, description, resolution
FROM support_tickets
WHERE customer_id = :customer_id
ORDER BY created_date DESC
"""

_OPEN_TICKETS_BY_NAME_SQL = """
SELECT t.ticket_id, t.customer_id, t.issue_type, t.status, t.priority,
       t.created_date, t.description, t.resolution
FROM support_tickets t
JOIN customers c ON c.customer_id = t.customer_id
WHERE LOWER(TRIM(c.name)) = LOWER(TRIM(:name))
  AND LOWER(t.status) = 'open'
ORDER BY t.created_date DESC
"""


def _rows_to_dicts(rows: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def get_customer_by_name(name: str) -> dict[str, Any]:
    """Look up a customer by name (case-insensitive, trimmed)."""
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
    """Return all support tickets for a customer ID."""
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
            "message": f"Database error while fetching tickets: {exc}",
        }

    if not tickets:
        return {
            "tickets": [],
            "message": f"No support tickets found for customer_id {customer_id}.",
        }

    return {"tickets": tickets, "count": len(tickets)}


def get_customer_profile_and_tickets(name: str) -> dict[str, Any]:
    """Return a customer's profile and their full ticket history."""
    customer = get_customer_by_name(name)
    if "customer_id" not in customer:
        return {
            "customer": None,
            "tickets": [],
            "message": customer.get("message", f"No customer found with name '{name.strip()}'."),
        }

    ticket_result = get_customer_tickets(customer["customer_id"])
    response: dict[str, Any] = {
        "customer": customer,
        "tickets": ticket_result["tickets"],
    }
    if "count" in ticket_result:
        response["ticket_count"] = ticket_result["count"]
    if "message" in ticket_result:
        response["message"] = ticket_result["message"]
    return response


def get_open_tickets(name: str) -> dict[str, Any]:
    """Return open support tickets for a customer identified by name."""
    cleaned_name = name.strip()
    if not cleaned_name:
        return {"tickets": [], "message": "Please provide a customer name."}

    customer = get_customer_by_name(cleaned_name)
    if "customer_id" not in customer:
        return {"tickets": [], "message": customer.get("message", f"No customer found with name '{cleaned_name}'.")}

    settings = get_settings()
    try:
        with get_session(settings.sqlite_db_path) as session:
            result = session.execute(
                text(_OPEN_TICKETS_BY_NAME_SQL),
                {"name": cleaned_name},
            )
            tickets = _rows_to_dicts(result.mappings().all())
    except Exception as exc:
        return {
            "tickets": [],
            "message": f"Database error while fetching open tickets: {exc}",
        }

    if not tickets:
        return {
            "tickets": [],
            "message": f"No open tickets found for customer '{cleaned_name}'.",
        }

    return {"tickets": tickets, "count": len(tickets)}


def get_sql_tools() -> list[Any]:
    """Return LangChain tools bound to the configured SQLite database."""
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

    return [
        lookup_customer_by_name,
        lookup_customer_tickets,
        lookup_customer_profile_and_tickets,
        lookup_open_tickets,
    ]
