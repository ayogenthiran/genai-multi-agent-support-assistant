# Customer Support Database – Schema Reference

This file describes every table and column in `data/customers.db` in plain English.
It is intended to provide additional schema context to the LangChain SQL agent so it
can plan accurate JOIN queries.

---

## Table: `customers`

Stores one row per customer account.

| Column | Type | Description |
|---|---|---|
| `customer_id` | INTEGER PK | Unique identifier for the customer. |
| `full_name` | TEXT | Customer's full name (e.g. "Ema Johnson"). |
| `email` | TEXT UNIQUE | Customer's email address; must be unique across all accounts. |
| `phone` | TEXT | Customer's phone number in local or international format. |
| `location` | TEXT | City and country of the customer (e.g. "Toronto, Canada"). |
| `customer_tier` | TEXT | Account tier: one of `Basic`, `Premium`, or `Enterprise`. |
| `signup_date` | TEXT | ISO 8601 date on which the customer registered (e.g. "2024-03-15"). |
| `status` | TEXT | Current account status: one of `Active`, `Inactive`, or `Suspended`. |

---

## Table: `orders`

Stores one row per customer order.
`orders.customer_id` links to `customers.customer_id`.

| Column | Type | Description |
|---|---|---|
| `order_id` | INTEGER PK | Unique identifier for the order. |
| `customer_id` | INTEGER FK | References `customers.customer_id` — the customer who placed the order. |
| `order_date` | TEXT | ISO 8601 date on which the order was placed. |
| `product_name` | TEXT | Name of the product ordered (e.g. "Bluetooth Speaker"). |
| `category` | TEXT | Product category (e.g. "Electronics", "Clothing"). |
| `amount` | REAL | Total order value in USD. |
| `payment_status` | TEXT | Payment outcome: one of `Paid`, `Pending`, `Failed`, or `Refunded`. |
| `delivery_status` | TEXT | Current delivery state: one of `Processing`, `Shipped`, `Delivered`, `Delayed`, `Cancelled`, or `Returned`. |

---

## Table: `support_tickets`

Stores one row per customer support ticket.
`support_tickets.customer_id` links to `customers.customer_id`.

| Column | Type | Description |
|---|---|---|
| `ticket_id` | INTEGER PK | Unique identifier for the support ticket. |
| `customer_id` | INTEGER FK | References `customers.customer_id` — the customer who raised the ticket. |
| `created_at` | TEXT | ISO 8601 date on which the ticket was created. |
| `issue_type` | TEXT | Category of the issue: one of `Refund Request`, `Damaged Product`, `Login Issue`, `Billing Issue`, `Subscription Change`, `Shipping Delay`, `Technical Support`, or `Account Update`. |
| `priority` | TEXT | Urgency level: one of `Low`, `Medium`, `High`, or `Critical`. |
| `status` | TEXT | Current ticket state: one of `Open`, `In Progress`, `Resolved`, or `Escalated`. |
| `description` | TEXT | Free-text description of the customer's problem. |
| `resolution` | TEXT (nullable) | How the issue was resolved; NULL when `status` is not `Resolved`. |
| `agent_name` | TEXT (nullable) | Name of the support agent assigned to the ticket; NULL only when `status` is `Open` and the ticket is unassigned. |

---

## Table: `subscriptions`

Stores one row per customer subscription plan.
`subscriptions.customer_id` links to `customers.customer_id`.

| Column | Type | Description |
|---|---|---|
| `subscription_id` | INTEGER PK | Unique identifier for the subscription. |
| `customer_id` | INTEGER FK | References `customers.customer_id` — the subscriber. |
| `plan_name` | TEXT | Subscription plan: one of `Basic Plan`, `Premium Plan`, or `Enterprise Plan`. |
| `start_date` | TEXT | ISO 8601 date on which the subscription started. |
| `renewal_date` | TEXT | ISO 8601 date of the next scheduled renewal. |
| `status` | TEXT | Subscription state: one of `Active`, `Cancelled`, `Expired`, or `Paused`. |

---

## Table: `refunds`

Stores one row per refund request, linked to both the customer and the original order.
`refunds.customer_id` links to `customers.customer_id`.
`refunds.order_id` links to `orders.order_id`.

| Column | Type | Description |
|---|---|---|
| `refund_id` | INTEGER PK | Unique identifier for the refund request. |
| `customer_id` | INTEGER FK | References `customers.customer_id` — the customer requesting the refund. |
| `order_id` | INTEGER FK | References `orders.order_id` — the order being refunded. |
| `refund_reason` | TEXT | Reason for the refund (uses the same values as `support_tickets.issue_type`). |
| `refund_status` | TEXT | Current status: one of `Requested`, `Approved`, `Rejected`, `Processed`, or `Escalated`. |
| `requested_at` | TEXT | ISO 8601 date on which the refund was requested. |
| `processed_at` | TEXT (nullable) | ISO 8601 date on which the refund was processed; NULL when `refund_status` is `Requested` or `Rejected`. |
| `refund_amount` | REAL | Amount refunded in USD; never exceeds the linked order's `amount`. |

---

## Key Relationships

```
customers  ──< orders           (customers.customer_id → orders.customer_id)
customers  ──< support_tickets  (customers.customer_id → support_tickets.customer_id)
customers  ──< subscriptions    (customers.customer_id → subscriptions.customer_id)
customers  ──< refunds          (customers.customer_id → refunds.customer_id)
orders     ──< refunds          (orders.order_id       → refunds.order_id)
```

---

## Useful SQL Patterns for the SQL Agent

The SQL Customer Agent uses **predefined, parameterized** versions of the
patterns below — it does not generate SQL from natural language. These
snippets are provided only as a human-readable reference for the schema.

```sql
-- Customer profile with all tickets
SELECT c.*, t.*
FROM customers c
JOIN support_tickets t ON t.customer_id = c.customer_id
WHERE LOWER(TRIM(c.full_name)) = 'ema johnson';

-- Open support tickets for a customer
SELECT t.*
FROM support_tickets t
JOIN customers c ON c.customer_id = t.customer_id
WHERE LOWER(TRIM(c.full_name)) = 'ema johnson'
  AND LOWER(t.status) = 'open';

-- Refund-related support tickets for a customer
SELECT t.*
FROM support_tickets t
JOIN customers c ON c.customer_id = t.customer_id
WHERE LOWER(TRIM(c.full_name)) = 'ema johnson'
  AND (
       LOWER(t.issue_type)  LIKE '%refund%'
    OR LOWER(t.description) LIKE '%refund%'
  );

-- High-priority open tickets (optionally scoped to a customer)
SELECT t.*, c.full_name AS customer_name
FROM support_tickets t
JOIN customers c ON c.customer_id = t.customer_id
WHERE LOWER(t.status) = 'open'
  AND LOWER(t.priority) IN ('high', 'critical');
```
