"""SQLite database access layer for customer support data.

Provides SQLAlchemy engine/session helpers and schema definitions for tables
such as customers and support tickets. Used by the SQL agent and sql_tools to
answer account-specific questions.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import ForeignKey, Text, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for customer support models."""


class Customer(Base):
    """Customer account record."""

    __tablename__ = "customers"

    customer_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    email: Mapped[str] = mapped_column(Text)
    phone: Mapped[str] = mapped_column(Text)
    customer_type: Mapped[str] = mapped_column(Text)
    join_date: Mapped[str] = mapped_column(Text)
    location: Mapped[str] = mapped_column(Text)

    tickets: Mapped[list["SupportTicket"]] = relationship(back_populates="customer")


class SupportTicket(Base):
    """Customer support ticket record."""

    __tablename__ = "support_tickets"

    ticket_id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"))
    issue_type: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    priority: Mapped[str] = mapped_column(Text)
    created_date: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)

    customer: Mapped["Customer"] = relationship(back_populates="tickets")


def get_engine(db_path: Path):
    """Create a SQLAlchemy engine for the given SQLite database path."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def get_session_factory(db_path: Path) -> sessionmaker[Session]:
    """Return a session factory bound to the SQLite database."""
    engine = get_engine(db_path)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def get_session(db_path: Path) -> Generator[Session, None, None]:
    """Yield a database session for read/write operations."""
    session_factory = get_session_factory(db_path)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(db_path: Path) -> None:
    """Create all database tables if they do not exist."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
