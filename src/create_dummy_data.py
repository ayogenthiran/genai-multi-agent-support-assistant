"""Create data/customers.db with synthetic customer support data."""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "customers.db"

N_CUSTOMERS = 50
N_ORDERS = 80
N_TICKETS = 70
N_SUBSCRIPTIONS = 25
N_REFUNDS = 18

random.seed(42)
Faker.seed(42)
fake = Faker()

TIERS = ["Basic", "Premium", "Enterprise"]
# Weighted toward Active so queries for active customers return meaningful results
CUSTOMER_STATUSES = ["Active", "Active", "Active", "Inactive", "Suspended"]

ISSUE_TYPES = [
    "Refund Request",
    "Damaged Product",
    "Warranty Claim",
    "Login Issue",
    "Billing Issue",
    "Subscription Change",
    "Shipping Delay",
    "Technical Support",
    "Account Update",
]
TICKET_STATUSES = ["Open", "In Progress", "Resolved", "Escalated"]
TICKET_PRIORITIES = ["Low", "Medium", "High", "Critical"]

# Weighted toward Paid so most orders look realistic
PAYMENT_STATUSES = ["Paid", "Paid", "Paid", "Pending", "Failed", "Refunded"]
DELIVERY_STATUSES = [
    "Processing",
    "Shipped",
    "Delivered",
    "Delivered",
    "Delayed",
    "Cancelled",
    "Returned",
]
REFUND_STATUSES = ["Requested", "Approved", "Rejected", "Processed", "Escalated"]

PLAN_NAMES = ["Basic Plan", "Premium Plan", "Enterprise Plan"]
SUBSCRIPTION_STATUSES = ["Active", "Active", "Cancelled", "Expired", "Paused"]

AGENT_NAMES = [
    "Sarah Chen",
    "James Park",
    "Amara Osei",
    "Lucas Rivera",
    "Mei Wang",
    "David Kim",
    "Priya Nair",
    "Carlos Mendez",
    "Emma Davis",
    "Ali Hassan",
]

# (product_name, category) pairs
PRODUCTS: list[tuple[str, str]] = [
    ("Wireless Headphones", "Electronics"),
    ("Smart Watch", "Electronics"),
    ("Laptop Stand", "Electronics"),
    ("USB-C Hub", "Electronics"),
    ("Mechanical Keyboard", "Electronics"),
    ("Monitor", "Electronics"),
    ("Webcam", "Electronics"),
    ("Running Shoes", "Clothing"),
    ("Office Chair", "Furniture"),
    ("Standing Desk", "Furniture"),
    ("Coffee Maker", "Appliances"),
    ("Air Purifier", "Appliances"),
    ("Bluetooth Speaker", "Electronics"),
    ("Phone Case", "Accessories"),
    ("Tablet", "Electronics"),
    ("Gaming Mouse", "Electronics"),
    ("LED Desk Lamp", "Home"),
    ("Smart Plug", "Electronics"),
    ("Power Bank", "Electronics"),
    ("Yoga Mat", "Sports"),
    ("Water Bottle", "Sports"),
    ("Backpack", "Accessories"),
    ("Sunglasses", "Accessories"),
    ("Book Set", "Books"),
    ("Puzzle Game", "Toys"),
]

TODAY = date.today()
START_SIGNUP = date(2024, 1, 1)
END_SIGNUP = date(2025, 12, 31)
START_ACTIVITY = date(2024, 6, 1)
SIX_MONTHS_AGO = TODAY - timedelta(days=180)


def _random_date(start: date, end: date) -> str:
    """Return a random ISO date string between start and end (inclusive)."""
    delta = max((end - start).days, 0)
    return (start + timedelta(days=random.randint(0, delta))).isoformat()


def _weighted_recent_date() -> str:
    """Return an activity date weighted 65% toward the last 6 months."""
    if random.random() < 0.65:
        return _random_date(SIX_MONTHS_AGO, TODAY)
    return _random_date(START_ACTIVITY, SIX_MONTHS_AGO)


def _date_after(base_date_str: str, min_days: int = 1, max_days: int = 30) -> str:
    """Return a date that is [min_days, max_days] after base_date_str, capped at today."""
    base = date.fromisoformat(base_date_str)
    offset = timedelta(days=random.randint(min_days, max_days))
    return min(base + offset, TODAY).isoformat()


_DROP_TABLES = """
DROP TABLE IF EXISTS refunds;
DROP TABLE IF EXISTS subscriptions;
DROP TABLE IF EXISTS support_tickets;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;
"""

_CREATE_TABLES = """
CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    full_name     TEXT    NOT NULL,
    email         TEXT    NOT NULL UNIQUE,
    phone         TEXT    NOT NULL,
    location      TEXT    NOT NULL,
    customer_tier TEXT    NOT NULL,
    signup_date   TEXT    NOT NULL,
    status        TEXT    NOT NULL
);

CREATE TABLE orders (
    order_id         INTEGER PRIMARY KEY,
    customer_id      INTEGER NOT NULL,
    order_date       TEXT    NOT NULL,
    product_name     TEXT    NOT NULL,
    category         TEXT    NOT NULL,
    amount           REAL    NOT NULL,
    payment_status   TEXT    NOT NULL,
    delivery_status  TEXT    NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE support_tickets (
    ticket_id   INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    created_at  TEXT    NOT NULL,
    issue_type  TEXT    NOT NULL,
    priority    TEXT    NOT NULL,
    status      TEXT    NOT NULL,
    description TEXT    NOT NULL,
    resolution  TEXT,
    agent_name  TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE subscriptions (
    subscription_id INTEGER PRIMARY KEY,
    customer_id     INTEGER NOT NULL,
    plan_name       TEXT    NOT NULL,
    start_date      TEXT    NOT NULL,
    renewal_date    TEXT    NOT NULL,
    status          TEXT    NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE refunds (
    refund_id      INTEGER PRIMARY KEY,
    customer_id    INTEGER NOT NULL,
    order_id       INTEGER NOT NULL,
    refund_reason  TEXT    NOT NULL,
    refund_status  TEXT    NOT NULL,
    requested_at   TEXT    NOT NULL,
    processed_at   TEXT,
    refund_amount  REAL    NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (order_id)    REFERENCES orders(order_id)
);
"""

# ---------------------------------------------------------------------------
# Fixed demo customers – customer_id 1, 2, 3 (reserved for deterministic demos)
# ---------------------------------------------------------------------------
# Three named customers are pinned so the SQL Customer Agent can reliably
# answer the assignment demo questions. Faker-generated customers start at
# customer_id = 4 and ticket IDs continue after the reserved tickets below.
#
# - Ema Johnson  (id=1): rich profile + orders + multiple tickets + 1 refund
# - Daniel Smith (id=2): two tickets covering delivery-delay and account issues
# - Priya Patel  (id=3): two tickets covering warranty/product-replacement and refund

_EMA_CUSTOMER: tuple = (
    1,
    "Ema Johnson",
    "ema.johnson@example.com",
    "+1-416-555-0101",
    "Toronto, Canada",
    "Premium",
    "2024-03-15",
    "Active",
)

_DANIEL_CUSTOMER: tuple = (
    2,
    "Daniel Smith",
    "daniel.smith@example.com",
    "+1-415-555-0102",
    "San Francisco, USA",
    "Basic",
    "2024-06-20",
    "Active",
)

_PRIYA_CUSTOMER: tuple = (
    3,
    "Priya Patel",
    "priya.patel@example.com",
    "+44-20-7946-0103",
    "London, United Kingdom",
    "Enterprise",
    "2024-02-10",
    "Active",
)

_RESERVED_CUSTOMERS: list[tuple] = [_EMA_CUSTOMER, _DANIEL_CUSTOMER, _PRIYA_CUSTOMER]

_EMA_ORDERS: list[tuple] = [
    # (order_id, customer_id, order_date, product_name, category, amount, payment_status, delivery_status)
    (1, 1, "2024-11-10", "Bluetooth Speaker", "Electronics", 89.99, "Paid", "Delivered"),
    (2, 1, "2025-01-20", "Laptop Stand", "Electronics", 45.00, "Refunded", "Returned"),
]

_RESERVED_ORDERS: list[tuple] = _EMA_ORDERS

_EMA_TICKETS: list[tuple] = [
    # (ticket_id, customer_id, created_at, issue_type, priority, status,
    #  description, resolution, agent_name)
    (
        1, 1, "2025-01-22",
        "Damaged Product", "High", "Resolved",
        "Received a damaged Laptop Stand — the corner was cracked on arrival.",
        "Replacement unit shipped; prepaid return label sent via email.",
        "Sarah Chen",
    ),
    (
        2, 1, "2025-02-05",
        "Refund Request", "Medium", "Open",
        "Requesting a full refund for the damaged Laptop Stand order.",
        None,  # resolution NULL — ticket is still open
        "James Park",
    ),
    (
        3, 1, "2025-03-10",
        "Shipping Delay", "Low", "Open",
        "Replacement Bluetooth Speaker has not arrived two weeks after dispatch.",
        None,  # resolution NULL — ticket is open
        None,  # agent_name NULL — open and unassigned
    ),
    (
        4, 1, "2025-04-02",
        "Billing Issue", "High", "Open",
        "Double-charged for the Bluetooth Speaker replacement — needs urgent refund review.",
        None,  # resolution NULL — ticket is open
        "Priya Nair",
    ),
]

_DANIEL_TICKETS: list[tuple] = [
    # (ticket_id, customer_id, created_at, issue_type, priority, status,
    #  description, resolution, agent_name)
    (
        5, 2, "2025-03-18",
        "Shipping Delay", "Medium", "Open",
        "Mechanical Keyboard order is three weeks past the promised delivery date.",
        None,  # resolution NULL — ticket is open
        "Lucas Rivera",
    ),
    (
        6, 2, "2025-02-12",
        "Account Update", "Low", "Resolved",
        "Need to update the email address on the account for login verification.",
        "Verified identity via phone; new email address applied and confirmed.",
        "Amara Osei",
    ),
]

_PRIYA_TICKETS: list[tuple] = [
    # (ticket_id, customer_id, created_at, issue_type, priority, status,
    #  description, resolution, agent_name)
    (
        7, 3, "2025-04-08",
        "Warranty Claim", "High", "Open",
        "Smart Watch arrived with a cracked screen — requesting a warranty "
        "replacement under the standard product warranty.",
        None,  # resolution NULL — ticket is open
        "Mei Wang",
    ),
    (
        8, 3, "2025-01-30",
        "Refund Request", "Medium", "Resolved",
        "Returned a Bluetooth Speaker within the 14-day refund window after "
        "discovering audio distortion at high volume.",
        "Refund of $74.50 processed back to the original payment method.",
        "David Kim",
    ),
]

_RESERVED_TICKETS: list[tuple] = _EMA_TICKETS + _DANIEL_TICKETS + _PRIYA_TICKETS

_EMA_REFUNDS: list[tuple] = [
    # (refund_id, customer_id, order_id, refund_reason, refund_status,
    #  requested_at, processed_at, refund_amount)
    (1, 1, 2, "Damaged Product", "Approved", "2025-02-05", "2025-02-10", 45.00),
]

_RESERVED_REFUNDS: list[tuple] = _EMA_REFUNDS


def _generate_customers() -> list[tuple]:
    """Generate random customers after the reserved demo records."""
    rows: list[tuple] = []
    first_id = len(_RESERVED_CUSTOMERS) + 1
    for cid in range(first_id, N_CUSTOMERS + 1):
        rows.append((
            cid,
            fake.name(),
            fake.unique.email(),
            fake.phone_number()[:20],
            f"{fake.city()}, {fake.country()}",
            random.choice(TIERS),
            _random_date(START_SIGNUP, END_SIGNUP),
            random.choice(CUSTOMER_STATUSES),
        ))
    return rows


def _generate_orders(cust_ids: list[int]) -> list[tuple]:
    """Generate random orders after the reserved demo orders."""
    rows: list[tuple] = []
    first_id = len(_RESERVED_ORDERS) + 1
    for oid in range(first_id, N_ORDERS + 1):
        product_name, category = random.choice(PRODUCTS)
        rows.append((
            oid,
            random.choice(cust_ids),
            _weighted_recent_date(),
            product_name,
            category,
            round(random.uniform(9.99, 499.99), 2),
            random.choice(PAYMENT_STATUSES),
            random.choice(DELIVERY_STATUSES),
        ))
    return rows


def _generate_tickets(cust_ids: list[int]) -> list[tuple]:
    """Generate random tickets after the reserved demo tickets."""
    rows: list[tuple] = []
    first_id = len(_RESERVED_TICKETS) + 1
    for tid in range(first_id, N_TICKETS + 1):
        status = random.choice(TICKET_STATUSES)
        if status == "Resolved":
            resolution: str | None = fake.sentence(nb_words=12)
            agent: str | None = random.choice(AGENT_NAMES)
        elif status in ("In Progress", "Escalated"):
            resolution = None
            agent = random.choice(AGENT_NAMES)
        else:  # Open
            resolution = None
            agent = random.choice(AGENT_NAMES) if random.random() < 0.5 else None

        rows.append((
            tid,
            random.choice(cust_ids),
            _weighted_recent_date(),
            random.choice(ISSUE_TYPES),
            random.choice(TICKET_PRIORITIES),
            status,
            fake.sentence(nb_words=18),
            resolution,
            agent,
        ))
    return rows


def _generate_subscriptions(all_cust_ids: list[int]) -> list[tuple]:
    """Generate N_SUBSCRIPTIONS subscriptions for a random subset of customers."""
    chosen = random.sample(all_cust_ids, N_SUBSCRIPTIONS)
    rows: list[tuple] = []
    for sid, cid in enumerate(chosen, start=1):
        start = _random_date(START_SIGNUP, TODAY - timedelta(days=30))
        renewal = _date_after(start, min_days=30, max_days=365)
        rows.append((
            sid,
            cid,
            random.choice(PLAN_NAMES),
            start,
            renewal,
            random.choice(SUBSCRIPTION_STATUSES),
        ))
    return rows


def _generate_refunds(extra_orders: list[tuple]) -> list[tuple]:
    """Generate refunds for randomly sampled non-reserved orders."""
    # extra_orders tuple layout: (order_id[0], customer_id[1], order_date[2],
    #   product_name[3], category[4], amount[5], payment_status[6], delivery_status[7])
    pool = [(row[0], row[1], row[5]) for row in extra_orders]  # (order_id, cust_id, amount)
    target_count = max(N_REFUNDS - len(_RESERVED_REFUNDS), 0)
    selected = random.sample(pool, min(target_count, len(pool)))

    first_id = len(_RESERVED_REFUNDS) + 1
    rows: list[tuple] = []
    for rid, (oid, cid, order_amount) in enumerate(selected, start=first_id):
        status = random.choice(REFUND_STATUSES)
        requested = _weighted_recent_date()
        processed = _date_after(requested, 1, 14) if status in ("Processed", "Approved") else None
        rows.append((
            rid,
            cid,
            oid,
            random.choice(ISSUE_TYPES),
            status,
            requested,
            processed,
            round(random.uniform(5.0, order_amount), 2),
        ))
    return rows


_INDEXES = [
    "CREATE INDEX idx_customers_full_name ON customers(full_name)",
    "CREATE INDEX idx_support_tickets_customer_id ON support_tickets(customer_id)",
    "CREATE INDEX idx_support_tickets_status ON support_tickets(status)",
    "CREATE INDEX idx_support_tickets_priority ON support_tickets(priority)",
    "CREATE INDEX idx_support_tickets_created_at ON support_tickets(created_at)",
    "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
    "CREATE INDEX idx_orders_delivery_status ON orders(delivery_status)",
    "CREATE INDEX idx_subscriptions_customer_id ON subscriptions(customer_id)",
    "CREATE INDEX idx_subscriptions_plan_name ON subscriptions(plan_name)",
    "CREATE INDEX idx_subscriptions_renewal_date ON subscriptions(renewal_date)",
    "CREATE INDEX idx_refunds_customer_id ON refunds(customer_id)",
    "CREATE INDEX idx_refunds_order_id ON refunds(order_id)",
]

def _print_reserved_customer(cur: sqlite3.Cursor, customer_id: int, name: str) -> None:
    """Print the profile, orders, tickets, and refunds for one reserved customer."""
    print(f"\n=== {name} (customer_id={customer_id}) – sanity check ===")

    cur.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
    profile = cur.fetchone()
    if profile:
        print("\n[Profile]")
        for key in profile.keys():
            print(f"  {key}: {profile[key]}")
    else:
        print(f"  ERROR: {name} not found in customers table!")
        return

    cur.execute(
        "SELECT * FROM orders WHERE customer_id = ? ORDER BY order_date",
        (customer_id,),
    )
    orders = cur.fetchall()
    if orders:
        print("\n[Orders]")
        for row in orders:
            print(
                f"  order_id={row['order_id']} | {row['product_name']} "
                f"| ${row['amount']:.2f} | {row['payment_status']} | {row['delivery_status']}"
            )

    cur.execute(
        "SELECT * FROM support_tickets WHERE customer_id = ? ORDER BY created_at",
        (customer_id,),
    )
    tickets = cur.fetchall()
    if tickets:
        print("\n[Support Tickets]")
        for row in tickets:
            print(
                f"  ticket_id={row['ticket_id']} | {row['issue_type']} "
                f"| {row['status']} | {row['priority']} | {row['created_at']}"
            )
            if row["agent_name"]:
                print(f"    agent: {row['agent_name']}")
            if row["resolution"]:
                print(f"    resolution: {row['resolution']}")

    cur.execute(
        """
        SELECT rf.*, o.product_name
        FROM refunds rf
        JOIN orders o ON o.order_id = rf.order_id
        WHERE rf.customer_id = ?
        """,
        (customer_id,),
    )
    refunds = cur.fetchall()
    if refunds:
        print("\n[Refunds]")
        for row in refunds:
            print(
                f"  refund_id={row['refund_id']} | order_id={row['order_id']} "
                f"| {row['product_name']} | ${row['refund_amount']:.2f} "
                f"| {row['refund_status']} | requested: {row['requested_at']}"
            )


def _run_sanity_check(db_path: Path) -> None:
    """Print profiles, orders, tickets, and refunds for all reserved demo customers."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    for reserved in _RESERVED_CUSTOMERS:
        _print_reserved_customer(cur, customer_id=reserved[0], name=reserved[1])

    conn.close()



def create_dummy_data(db_path: Path = DB_PATH) -> Path:
    """Create the SQLite demo database and return its path."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")

    conn.executescript(_DROP_TABLES + _CREATE_TABLES)

    extra_customers = _generate_customers()
    all_customers = list(_RESERVED_CUSTOMERS) + extra_customers

    non_reserved_cids = [row[0] for row in extra_customers]
    extra_orders = _generate_orders(non_reserved_cids)
    all_orders = list(_RESERVED_ORDERS) + extra_orders

    extra_tickets = _generate_tickets(non_reserved_cids)
    all_tickets = list(_RESERVED_TICKETS) + extra_tickets

    all_cust_ids = [row[0] for row in all_customers]
    all_subscriptions = _generate_subscriptions(all_cust_ids)

    extra_refunds = _generate_refunds(extra_orders)
    all_refunds = list(_RESERVED_REFUNDS) + extra_refunds

    # Insert in foreign-key dependency order (parents before children)
    with conn:
        conn.executemany(
            "INSERT INTO customers VALUES (?,?,?,?,?,?,?,?)",
            all_customers,
        )
        conn.executemany(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)",
            all_orders,
        )
        conn.executemany(
            "INSERT INTO support_tickets VALUES (?,?,?,?,?,?,?,?,?)",
            all_tickets,
        )
        conn.executemany(
            "INSERT INTO subscriptions VALUES (?,?,?,?,?,?)",
            all_subscriptions,
        )
        conn.executemany(
            "INSERT INTO refunds VALUES (?,?,?,?,?,?,?,?)",
            all_refunds,
        )
        for idx_sql in _INDEXES:
            conn.execute(idx_sql)

    conn.close()

    print("\n=== Database created ===")
    print(f"  Path             : {db_path}")
    print(f"  customers        : {len(all_customers)}")
    print(f"  orders           : {len(all_orders)}")
    print(f"  support_tickets  : {len(all_tickets)}")
    print(f"  subscriptions    : {len(all_subscriptions)}")
    print(f"  refunds          : {len(all_refunds)}")

    _run_sanity_check(db_path)

    return db_path


# Backwards-compatible alias retained for any external scripts that still
# reference the old function name.
create_database = create_dummy_data


def main() -> None:
    """CLI entry point."""
    create_dummy_data()


if __name__ == "__main__":
    main()
