"""MCP-style local tool registry for support agents."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.tools.document_tools import (
    format_policy_answer,
    pdf_ingestion,
    policy_document_search,
    policy_question_answer,
)
from src.tools.sql_tools import (
    get_customer_by_name,
    get_customer_tickets,
    get_high_priority_open_tickets,
    get_open_tickets,
    get_refund_related_tickets,
)

ToolCallable = Callable[..., Any]

def customer_profile_lookup(name: str) -> dict[str, Any]:
    """Look up a customer profile by name (case-insensitive)."""
    return get_customer_by_name(name)


def customer_ticket_lookup(name: str) -> dict[str, Any]:
    """Return all support tickets for the customer identified by name."""
    customer = get_customer_by_name(name)
    if "customer_id" not in customer:
        return {
            "tickets": [],
            "count": 0,
            "message": customer.get(
                "message",
                f"No customer found with name '{name.strip()}'.",
            ),
        }

    ticket_result = get_customer_tickets(customer["customer_id"])
    response: dict[str, Any] = {
        "customer_id": customer["customer_id"],
        "customer_name": customer.get("full_name"),
        "tickets": ticket_result.get("tickets", []),
        "count": ticket_result.get("count", 0),
    }
    if "message" in ticket_result:
        response["message"] = ticket_result["message"]
    return response


def open_ticket_lookup(name: str) -> dict[str, Any]:
    """Return open support tickets for the customer identified by name."""
    return get_open_tickets(name)


def refund_ticket_lookup(name: str) -> dict[str, Any]:
    """Return refund-related support tickets for the customer identified by name."""
    return get_refund_related_tickets(name)


def high_priority_open_ticket_lookup(name: str | None = None) -> dict[str, Any]:
    """Return high-priority open tickets, optionally filtered by customer name."""
    return get_high_priority_open_tickets(name)


TOOL_REGISTRY: dict[str, ToolCallable] = {
    "customer_profile_lookup": customer_profile_lookup,
    "customer_ticket_lookup": customer_ticket_lookup,
    "open_ticket_lookup": open_ticket_lookup,
    "refund_ticket_lookup": refund_ticket_lookup,
    "high_priority_open_ticket_lookup": high_priority_open_ticket_lookup,
    "policy_document_search": policy_document_search,
    "policy_question_answer": policy_question_answer,
    "pdf_ingestion": pdf_ingestion,
}


def list_tools() -> list[str]:
    """Return registered tool names."""
    return list(TOOL_REGISTRY.keys())


def call_tool(name: str, /, **kwargs: Any) -> Any:
    """Invoke a registered tool by name with keyword arguments."""
    if name not in TOOL_REGISTRY:
        available = ", ".join(sorted(TOOL_REGISTRY))
        raise KeyError(f"Unknown tool '{name}'. Available tools: {available}")
    return TOOL_REGISTRY[name](**kwargs)


def get_langchain_tools() -> list[Any]:
    """Return LangChain ``@tool`` wrappers for all registered MCP-style tools."""
    from langchain_core.tools import tool

    @tool
    def customer_profile_lookup_tool(name: str) -> dict[str, Any]:
        """Look up a customer profile by name."""
        return customer_profile_lookup(name)

    @tool
    def customer_ticket_lookup_tool(name: str) -> dict[str, Any]:
        """List all support tickets for a customer by name."""
        return customer_ticket_lookup(name)

    @tool
    def open_ticket_lookup_tool(name: str) -> dict[str, Any]:
        """List open support tickets for a customer by name."""
        return open_ticket_lookup(name)

    @tool
    def refund_ticket_lookup_tool(name: str) -> dict[str, Any]:
        """List refund-related support tickets for a customer by name."""
        return refund_ticket_lookup(name)

    @tool
    def high_priority_open_ticket_lookup_tool(name: str | None = None) -> dict[str, Any]:
        """List high-priority open tickets, optionally filtered by customer name."""
        return high_priority_open_ticket_lookup(name)

    @tool
    def policy_document_search_tool(query: str) -> str:
        """Search indexed company policy documents for relevant passages."""
        return policy_document_search(query)

    @tool
    def policy_question_answer_tool(query: str) -> str:
        """Answer a customer policy question using indexed policy documents."""
        return format_policy_answer(policy_question_answer(query))

    @tool
    def pdf_ingestion_tool(file_path: str) -> dict[str, Any]:
        """Index a policy PDF so it can be searched and used for answers."""
        return pdf_ingestion(file_path)

    return [
        customer_profile_lookup_tool,
        customer_ticket_lookup_tool,
        open_ticket_lookup_tool,
        refund_ticket_lookup_tool,
        high_priority_open_ticket_lookup_tool,
        policy_document_search_tool,
        policy_question_answer_tool,
        pdf_ingestion_tool,
    ]


def create_mcp_server() -> FastMCP:
    """Build a FastMCP server with all support tools registered."""
    mcp = FastMCP(
        name="customer-support-tools",
        instructions=(
            "Customer support tool layer: predefined, parameterized SQL "
            "lookups for customer profiles and support tickets, plus "
            "RAG-backed policy document search, Q&A, and PDF ingestion."
        ),
    )

    mcp.add_tool(
        customer_profile_lookup,
        name="customer_profile_lookup",
        description="Look up a customer profile by name (case-insensitive).",
    )
    mcp.add_tool(
        customer_ticket_lookup,
        name="customer_ticket_lookup",
        description="List all support tickets for a customer identified by name.",
    )
    mcp.add_tool(
        open_ticket_lookup,
        name="open_ticket_lookup",
        description="List open support tickets for a customer identified by name.",
    )
    mcp.add_tool(
        refund_ticket_lookup,
        name="refund_ticket_lookup",
        description="List refund-related support tickets for a customer identified by name.",
    )
    mcp.add_tool(
        high_priority_open_ticket_lookup,
        name="high_priority_open_ticket_lookup",
        description=(
            "List high-priority (High or Critical) open support tickets, "
            "optionally filtered by customer name."
        ),
    )
    mcp.add_tool(
        policy_document_search,
        name="policy_document_search",
        description="Search indexed policy documents and return relevant passages.",
    )
    mcp.add_tool(
        policy_question_answer,
        name="policy_question_answer",
        description="Answer a policy question using retrieved document context.",
    )
    mcp.add_tool(
        pdf_ingestion,
        name="pdf_ingestion",
        description="Index a policy PDF file path into the vector store for search.",
    )

    return mcp


def main() -> None:
    """Start the MCP server using stdio transport."""
    create_mcp_server().run(transport="stdio")


if __name__ == "__main__":
    main()
