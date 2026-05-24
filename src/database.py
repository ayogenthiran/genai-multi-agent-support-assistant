"""SQLite database access layer for customer support data.

Provides SQLAlchemy engine/session helpers and ORM schema definitions for all
five tables: customers, orders, support_tickets, subscriptions, and refunds.
Raw-SQL tools in sql_tools.py use text() queries over the same connection.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Float, ForeignKey, Text, create_engine
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
    full_name: Mapped[str] = mapped_column(Text)
    email: Mapped[str] = mapped_column(Text, unique=True)
    phone: Mapped[str] = mapped_column(Text)
    location: Mapped[str] = mapped_column(Text)
    customer_tier: Mapped[str] = mapped_column(Text)
    signup_date: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)

    orders: Mapped[list["Order"]] = relationship(back_populates="customer")
    tickets: Mapped[list["SupportTicket"]] = relationship(back_populates="customer")
    subscriptions: Mapped[list["Subscription"]] = relationship(back_populates="customer")
    refunds: Mapped[list["Refund"]] = relationship(back_populates="customer")


class Order(Base):
    """Customer order record."""

    __tablename__ = "orders"

    order_id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"))
    order_date: Mapped[str] = mapped_column(Text)
    product_name: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(Text)
    amount: Mapped[float] = mapped_column(Float)
    payment_status: Mapped[str] = mapped_column(Text)
    delivery_status: Mapped[str] = mapped_column(Text)

    customer: Mapped["Customer"] = relationship(back_populates="orders")
    refunds: Mapped[list["Refund"]] = relationship(back_populates="order")


class SupportTicket(Base):
    """Customer support ticket record."""

    __tablename__ = "support_tickets"

    ticket_id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"))
    created_at: Mapped[str] = mapped_column(Text)
    issue_type: Mapped[str] = mapped_column(Text)
    priority: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent_name: Mapped[str | None] = mapped_column(Text, nullable=True)

    customer: Mapped["Customer"] = relationship(back_populates="tickets")


class Subscription(Base):
    """Customer subscription record."""

    __tablename__ = "subscriptions"

    subscription_id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"))
    plan_name: Mapped[str] = mapped_column(Text)
    start_date: Mapped[str] = mapped_column(Text)
    renewal_date: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)

    customer: Mapped["Customer"] = relationship(back_populates="subscriptions")


class Refund(Base):
    """Refund request record."""

    __tablename__ = "refunds"

    refund_id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"))
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.order_id"))
    refund_reason: Mapped[str] = mapped_column(Text)
    refund_status: Mapped[str] = mapped_column(Text)
    requested_at: Mapped[str] = mapped_column(Text)
    processed_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    refund_amount: Mapped[float] = mapped_column(Float)

    customer: Mapped["Customer"] = relationship(back_populates="refunds")
    order: Mapped["Order"] = relationship(back_populates="refunds")


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
