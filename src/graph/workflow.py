"""LangGraph workflow orchestrating the multi-agent support pipeline.

Defines graph state, nodes (supervisor, SQL, RAG, response), conditional edges,
and compilation into an invokable graph used by app.py.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.agents.rag_agent import create_rag_agent
from src.agents.response_agent import create_response_agent
from src.agents.sql_agent import create_sql_agent
from src.agents.supervisor import create_supervisor_agent

RouteType = Literal["sql", "rag", "both", "general"]
NodeName = Literal["sql_agent", "rag_agent", "response_agent"]


class SupportState(TypedDict, total=False):
    """Shared state passed between agent nodes in the graph."""

    user_query: str
    route: RouteType
    sql_result: str | None
    rag_result: dict[str, Any] | None
    final_answer: str | None
    agent_used: str | None
    error: str | None
    # Bridge fields used by existing agent node implementations.
    query: str
    rag_context: str | None
    final_response: str | None
    customer_id: str | None


def _resolve_agent_used(route: RouteType | None) -> str:
    if route in {"sql", "rag", "both", "general"}:
        return route  # type: ignore[return-value]
    return "general"


def _supervisor_node(state: SupportState) -> dict[str, Any]:
    bridged = dict(state)
    if state.get("user_query") and not state.get("query"):
        bridged["query"] = state["user_query"]

    result = create_supervisor_agent()(bridged)
    query = str(result.get("query") or state.get("user_query") or "").strip()
    route = result.get("route", "general")
    return {
        "user_query": query,
        "query": query,
        "route": route,
    }


def _sql_node(state: SupportState) -> dict[str, Any]:
    return create_sql_agent()(dict(state))


def _rag_node(state: SupportState) -> dict[str, Any]:
    result = create_rag_agent()(dict(state))
    rag_context = result.get("rag_context")
    return {"rag_context": rag_context, "rag_result": result.get("rag_result")}


def _response_node(state: SupportState) -> dict[str, Any]:
    bridged = dict(state)

    result = create_response_agent()(bridged)
    route = state.get("route", "general")
    return {
        "final_response": result.get("final_response"),
        "final_answer": result.get("final_response"),
        "agent_used": _resolve_agent_used(route),
    }


def _route_after_supervisor(state: SupportState) -> NodeName:
    route = state.get("route", "general")
    if route == "sql":
        return "sql_agent"
    if route == "rag":
        return "rag_agent"
    if route == "both":
        return "sql_agent"
    return "response_agent"


def _route_after_sql(state: SupportState) -> NodeName:
    if state.get("route") == "both":
        return "rag_agent"
    return "response_agent"


def build_support_graph() -> Any:
    """Construct and compile the multi-agent LangGraph workflow."""
    graph = StateGraph(SupportState)

    graph.add_node("supervisor", _supervisor_node)
    graph.add_node("sql_agent", _sql_node)
    graph.add_node("rag_agent", _rag_node)
    graph.add_node("response_agent", _response_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "sql_agent": "sql_agent",
            "rag_agent": "rag_agent",
            "response_agent": "response_agent",
        },
    )
    graph.add_conditional_edges(
        "sql_agent",
        _route_after_sql,
        {
            "rag_agent": "rag_agent",
            "response_agent": "response_agent",
        },
    )
    graph.add_edge("rag_agent", "response_agent")
    graph.add_edge("response_agent", END)

    return graph.compile()


def run_multi_agent_workflow(user_query: str) -> dict[str, Any]:
    """Run the compiled LangGraph workflow for a single user query.

    Always returns a dict with the same shape regardless of route or failure:
    ``final_answer``, ``agent_used``, ``route``, ``sql_result``, ``rag_result``,
    ``error``. On a failure the ``error`` field carries the exception message
    and ``final_answer`` is left as ``None`` so callers can render a fallback.
    """
    initial_state: SupportState = {
        "user_query": user_query,
        "query": user_query,
    }

    try:
        graph = build_support_graph()
        result = graph.invoke(initial_state)
    except Exception as exc:
        return {
            "final_answer": None,
            "agent_used": "general",
            "route": "general",
            "sql_result": None,
            "rag_result": None,
            "error": str(exc),
        }

    return {
        "final_answer": result.get("final_answer"),
        "agent_used": result.get("agent_used") or _resolve_agent_used(result.get("route")),
        "route": result.get("route") or "general",
        "sql_result": result.get("sql_result"),
        "rag_result": result.get("rag_result"),
        "error": None,
    }
