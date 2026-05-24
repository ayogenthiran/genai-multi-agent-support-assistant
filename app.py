"""Streamlit entry point for the GenAI Multi-Agent Customer Support Assistant.

This module hosts the chat UI, wires up the LangGraph workflow, and displays
agent routing plus final responses.

Run locally (project virtualenv — Python 3.10+ required):
    .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

MIN_PYTHON = (3, 10)
PAGE_TITLE = "Multi-Agent Customer Support Assistant"
DISPLAY_TITLE = f"🤖 {PAGE_TITLE}"


def _ensure_python_version() -> None:
    if sys.version_info >= MIN_PYTHON:
        return

    message = (
        f"This app requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer "
        f"(you are on {sys.version_info.major}.{sys.version_info.minor}). "
        "LangGraph and Pydantic are not compatible with older interpreters.\n\n"
        "Activate the project virtual environment and run:\n\n"
        "```bash\n"
        "source .venv/bin/activate\n"
        "streamlit run app.py\n"
        "```\n\n"
        "Or run directly:\n\n"
        "```bash\n"
        ".venv/bin/streamlit run app.py\n"
        "```"
    )
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🤖", layout="wide")
    st.error("Incompatible Python version")
    st.markdown(message)
    st.stop()


_ensure_python_version()

from src.config import get_settings  # noqa: E402
from src.graph.workflow import run_multi_agent_workflow  # noqa: E402
from src.mcp_server.server import pdf_ingestion  # noqa: E402

AGENT_LABELS: dict[str, str] = {
    "sql": "SQL Agent",
    "rag": "Policy RAG Agent",
    "both": "Both",
    # Legacy value emitted by older workflow versions; kept for back-compat.
    "sql,rag": "Both",
    "general": "General Response",
}

NO_ANSWER_MESSAGE = (
    "I could not generate an answer for that question. Please try again."
)

DEFAULT_ERROR_MESSAGE = (
    "Something went wrong while processing your question. "
    "Please try again or rephrase your request."
)


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: list[dict[str, Any]] = []


def _format_agent_used(result: dict[str, Any]) -> str:
    raw = str(result.get("agent_used") or result.get("route") or "general").lower()
    if raw in AGENT_LABELS:
        return AGENT_LABELS[raw]
    if "sql" in raw and "rag" in raw:
        return AGENT_LABELS["both"]
    return AGENT_LABELS["general"]


def _save_uploaded_pdf(uploaded_file: Any) -> Path:
    policies_dir = get_settings().policies_dir
    policies_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    file_path = policies_dir / safe_name
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path


def _process_uploaded_pdf(uploaded_file: Any) -> None:
    try:
        file_path = _save_uploaded_pdf(uploaded_file)
        with st.spinner("Indexing policy PDF..."):
            result = pdf_ingestion(str(file_path))
    except Exception as exc:
        st.error(f"Failed to process PDF: {exc}")
        return

    message = result.get("message", "PDF processed.")
    chunks_added = result.get("chunks_added", 0)
    if chunks_added:
        st.success(message)
    else:
        st.warning(message)


def _run_workflow(user_query: str) -> tuple[str, str, str | None]:
    try:
        result = run_multi_agent_workflow(user_query)
    except Exception:
        return DEFAULT_ERROR_MESSAGE, AGENT_LABELS["general"], "error"

    agent_used = _format_agent_used(result)
    error = result.get("error")
    final_answer = str(result.get("final_answer") or "").strip()

    if error:
        return DEFAULT_ERROR_MESSAGE, agent_used, "warning"
    if not final_answer:
        return NO_ANSWER_MESSAGE, agent_used, "warning"
    return final_answer, agent_used, None


def _append_assistant_reply(user_query: str) -> None:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            final_answer, agent_used, status = _run_workflow(user_query)
        if status == "error":
            st.error(final_answer)
        elif status == "warning":
            st.warning(final_answer)
        else:
            st.markdown(final_answer)
        st.caption(f"Agent used: {agent_used}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_answer,
            "agent_used": agent_used,
            "status": status,
        }
    )


def _submit_user_query(user_query: str) -> None:
    cleaned = user_query.strip()
    if not cleaned:
        return
    st.session_state.messages.append({"role": "user", "content": cleaned})


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Policy Documents")
        st.caption("Upload a policy PDF to make it searchable.")

        uploaded_file = st.file_uploader(
            "Policy PDF",
            type=["pdf"],
            help="PDFs are saved to data/policies/ and indexed into ChromaDB.",
        )

        if st.button("Process", use_container_width=True):
            if uploaded_file is None:
                st.warning("Please upload a PDF before processing.")
            else:
                _process_uploaded_pdf(uploaded_file)


def _render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            status = message.get("status")
            if status == "error":
                st.error(message["content"])
            elif status == "warning":
                st.warning(message["content"])
            else:
                st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("agent_used"):
                st.caption(f"Agent used: {message['agent_used']}")


def _render_chat_area() -> None:
    _render_chat_history()


def _render_page_header() -> None:
    st.title(DISPLAY_TITLE)
    st.caption(
        "Ask questions about customer profiles, support tickets, and policy documents."
    )


def _maybe_answer_latest_user_message() -> None:
    messages = st.session_state.messages
    if not messages or messages[-1]["role"] != "user":
        return
    _append_assistant_reply(str(messages[-1]["content"]))


def main() -> None:
    """Launch the Streamlit application."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🤖",
        layout="wide",
    )

    _init_session_state()
    _render_sidebar()

    _render_page_header()

    _render_chat_area()
    _maybe_answer_latest_user_message()

    if prompt := st.chat_input("Ask about customers, tickets, or company policies..."):
        _submit_user_query(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
