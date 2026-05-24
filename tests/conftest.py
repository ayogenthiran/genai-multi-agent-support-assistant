"""Shared fixtures and sample queries for the support assistant test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.create_dummy_data import create_dummy_data

EMA_JOHNSON = "Ema Johnson"
DANIEL_SMITH = "Daniel Smith"
PRIYA_PATEL = "Priya Patel"

SQL_QUERY = (
    "Give me a quick overview of customer Ema Johnson's profile and "
    "past support ticket details."
)
RAG_QUERY = "What is the current refund policy?"
MIXED_QUERY = (
    "Can Ema Johnson get a refund based on her support history and "
    "the refund policy?"
)
GENERAL_QUERY = "Hi, what can you do?"


@pytest.fixture
def seeded_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a fresh customers.db at a tmp path and route settings at it."""
    db_path = create_dummy_data(tmp_path / "customers.db")
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))

    from src.config import get_settings

    get_settings.cache_clear()
    yield db_path
    get_settings.cache_clear()
