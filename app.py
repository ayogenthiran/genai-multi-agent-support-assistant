"""Streamlit entry point for the GenAI Multi-Agent Customer Support Assistant.

This module hosts the chat UI, wires up the LangGraph workflow, and displays
agent routing plus final responses.

Run locally (project virtualenv — Python 3.10+ required):
    .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

MIN_PYTHON = (3, 10)
PAGE_TITLE = "GenAI Multi-Agent Customer Support Assistant"


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
    st.set_page_config(page_title=PAGE_TITLE, page_icon="💬", layout="wide")
    st.error("Incompatible Python version")
    st.markdown(message)
    st.stop()


_ensure_python_version()

from src.graph.workflow import run_multi_agent_workflow
from src.mcp_server.server import pdf_ingestion

AGENT_LABELS: dict[str, str] = {
    "sql": "SQL Agent",
    "rag": "Policy RAG Agent",
    "both": "Both",
    "sql,rag": "Both",
    "general": "General Response",
}

EXAMPLE_QUESTIONS: list[str] = [
    "What is the current refund policy?",
    "Give me a quick overview of customer Ema's profile and past support ticket details.",
    "Can Ema get a refund based on her past support ticket and the refund policy?",
    "Show me Ema's open support tickets.",
    "What does the warranty policy say?",
    "Hi, what can you do?",
]

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
    temp_dir = Path(tempfile.gettempdir()) / "genai-support-assistant"
    temp_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    file_path = temp_dir / safe_name
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


def _run_workflow(user_query: str) -> tuple[str, str]:
    try:
        result = run_multi_agent_workflow(user_query)
    except Exception:
        return DEFAULT_ERROR_MESSAGE, AGENT_LABELS["general"]

    final_answer = str(result.get("final_answer") or "").strip()
    agent_used = _format_agent_used(result)
    if not final_answer:
        return (
            "I could not generate an answer for that question. Please try again.",
            agent_used,
        )
    return final_answer, agent_used


def _append_assistant_reply(user_query: str) -> None:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            final_answer, agent_used = _run_workflow(user_query)
        st.markdown(final_answer)
        st.caption(f"**Agent used:** {agent_used}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_answer,
            "agent_used": agent_used,
        }
    )


def _submit_user_query(user_query: str) -> None:
    cleaned = user_query.strip()
    if not cleaned:
        return
    st.session_state.messages.append({"role": "user", "content": cleaned})


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Policy documents")
        st.caption("Upload a company policy PDF to index it for RAG search.")

        uploaded_file = st.file_uploader(
            "Upload policy PDF",
            type=["pdf"],
            help="PDFs are saved temporarily and indexed into the vector store.",
        )

        if st.button("Process Policy PDF", use_container_width=True):
            if uploaded_file is None:
                st.warning("Please upload a PDF before processing.")
            else:
                _process_uploaded_pdf(uploaded_file)

        st.divider()
        st.subheader("Example questions")
        st.caption("Click a question to send it to the assistant.")

        for index, question in enumerate(EXAMPLE_QUESTIONS):
            if st.button(
                question,
                key=f"example_question_{index}",
                use_container_width=True,
            ):
                _submit_user_query(question)
                st.rerun()


def _render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("agent_used"):
                st.caption(f"**Agent used:** {message['agent_used']}")


def _maybe_answer_latest_user_message() -> None:
    messages = st.session_state.messages
    if not messages or messages[-1]["role"] != "user":
        return
    _append_assistant_reply(str(messages[-1]["content"]))


def main() -> None:
    """Launch the Streamlit application."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="💬",
        layout="wide",
    )

    _init_session_state()
    _render_sidebar()

    st.title(PAGE_TITLE)
    st.markdown(
        "Welcome, **John**. Ask questions about customer SQL data (profiles, tickets, "
        "and account history) and uploaded company policy PDFs. The supervisor routes "
        "each question to the SQL agent, the policy RAG agent, both, or a general reply."
    )

    _render_chat_history()
    _maybe_answer_latest_user_message()

    if prompt := st.chat_input("Ask about customers, tickets, or company policies..."):
        _submit_user_query(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
