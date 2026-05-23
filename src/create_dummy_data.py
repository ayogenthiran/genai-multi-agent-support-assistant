"""Seed script for demo customer support data.

Populates `data/customers.db` with sample customers and support tickets.
Run once during local setup:

    python -m src.create_dummy_data
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from src.config import get_settings
from src.database import Customer, SupportTicket, get_engine, init_db


CUSTOMERS: list[dict[str, str | int]] = [
    {
        "customer_id": 1,
        "name": "Ema Johnson",
        "email": "ema.johnson@example.com",
        "phone": "+1-555-0101",
        "customer_type": "premium",
        "join_date": "2022-03-14",
        "location": "Seattle, WA",
    },
    {
        "customer_id": 2,
        "name": "Daniel Smith",
        "email": "daniel.smith@example.com",
        "phone": "+1-555-0102",
        "customer_type": "standard",
        "join_date": "2021-11-02",
        "location": "Austin, TX",
    },
    {
        "customer_id": 3,
        "name": "Priya Patel",
        "email": "priya.patel@example.com",
        "phone": "+1-555-0103",
        "customer_type": "premium",
        "join_date": "2023-01-18",
        "location": "Chicago, IL",
    },
    {
        "customer_id": 4,
        "name": "Michael Brown",
        "email": "michael.brown@example.com",
        "phone": "+1-555-0104",
        "customer_type": "standard",
        "join_date": "2020-07-09",
        "location": "Denver, CO",
    },
    {
        "customer_id": 5,
        "name": "Sara Wilson",
        "email": "sara.wilson@example.com",
        "phone": "+1-555-0105",
        "customer_type": "premium",
        "join_date": "2022-09-25",
        "location": "Boston, MA",
    },
]

SUPPORT_TICKETS: list[dict[str, str | int | None]] = [
    {
        "ticket_id": 1,
        "customer_id": 1,
        "issue_type": "refund",
        "status": "resolved",
        "priority": "medium",
        "created_date": "2024-01-12",
        "description": "Requested a refund for a duplicate wireless charger order.",
        "resolution": "Refund of $49.99 issued to the original payment method.",
    },
    {
        "ticket_id": 2,
        "customer_id": 1,
        "issue_type": "delivery delay",
        "status": "open",
        "priority": "high",
        "created_date": "2024-02-03",
        "description": "Smart speaker shipment is five days past the estimated delivery date.",
        "resolution": None,
    },
    {
        "ticket_id": 3,
        "customer_id": 2,
        "issue_type": "warranty",
        "status": "in progress",
        "priority": "medium",
        "created_date": "2024-01-20",
        "description": "Laptop battery no longer holds a charge after 14 months of use.",
        "resolution": "Warranty claim approved; replacement battery is being shipped.",
    },
    {
        "ticket_id": 4,
        "customer_id": 2,
        "issue_type": "account issue",
        "status": "resolved",
        "priority": "low",
        "created_date": "2023-12-05",
        "description": "Unable to update billing address in the customer portal.",
        "resolution": "Account profile reset and billing address updated by support.",
    },
    {
        "ticket_id": 5,
        "customer_id": 3,
        "issue_type": "product replacement",
        "status": "resolved",
        "priority": "high",
        "created_date": "2024-01-08",
        "description": "Received a cracked tablet screen on arrival.",
        "resolution": "Replacement unit shipped with prepaid return label.",
    },
    {
        "ticket_id": 6,
        "customer_id": 3,
        "issue_type": "refund",
        "status": "closed",
        "priority": "medium",
        "created_date": "2023-11-17",
        "description": "Returned unused fitness tracker within the 30-day return window.",
        "resolution": "Full refund processed after warehouse inspection.",
    },
    {
        "ticket_id": 7,
        "customer_id": 4,
        "issue_type": "delivery delay",
        "status": "in progress",
        "priority": "medium",
        "created_date": "2024-02-10",
        "description": "Office chair order stuck in transit with no tracking updates for four days.",
        "resolution": "Carrier investigation opened; expedited shipping credit offered.",
    },
    {
        "ticket_id": 8,
        "customer_id": 4,
        "issue_type": "warranty",
        "status": "open",
        "priority": "low",
        "created_date": "2024-02-15",
        "description": "Coffee maker stopped heating water after the one-year warranty period.",
        "resolution": None,
    },
    {
        "ticket_id": 9,
        "customer_id": 5,
        "issue_type": "account issue",
        "status": "resolved",
        "priority": "high",
        "created_date": "2024-01-25",
        "description": "Two-factor authentication codes are not being delivered to her phone.",
        "resolution": "Phone number re-verified and backup codes issued.",
    },
    {
        "ticket_id": 10,
        "customer_id": 5,
        "issue_type": "product replacement",
        "status": "open",
        "priority": "medium",
        "created_date": "2024-02-18",
        "description": "Bluetooth headphones power off randomly after 20 minutes of use.",
        "resolution": None,
    },
]


def create_dummy_data(db_path: Path | None = None) -> Path:
    """Create tables and insert sample rows into the SQLite database."""
    settings = get_settings()
    target_path = db_path or settings.sqlite_db_path

    if target_path.exists():
        target_path.unlink()

    init_db(target_path)
    engine = get_engine(target_path)

    with Session(engine) as session:
        for customer in CUSTOMERS:
            session.add(Customer(**customer))
        for ticket in SUPPORT_TICKETS:
            session.add(SupportTicket(**ticket))
        session.commit()

    return target_path


def main() -> None:
    """CLI entry point for seeding demo data."""
    db_path = create_dummy_data()
    print(
        f"Successfully created demo database at {db_path} "
        f"with {len(CUSTOMERS)} customers and {len(SUPPORT_TICKETS)} support tickets."
    )


if __name__ == "__main__":
    main()
