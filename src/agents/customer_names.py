"""Shared helpers for resolving demo customer names from natural-language queries."""

from __future__ import annotations

import re

KNOWN_CUSTOMER_NAMES: tuple[str, ...] = (
    "Ema Johnson",
    "Daniel Smith",
    "Priya Patel",
)

_POSSESSIVE_FIRST_NAME = re.compile(
    r"\b([A-Za-z]+)(?:'s|'s|s)\b",
    re.IGNORECASE,
)


def match_known_customer(query: str) -> str | None:
    """Match a known demo customer from a query (full name or first name)."""
    cleaned = query.strip()
    if not cleaned:
        return None

    lowered = cleaned.lower()
    for name in KNOWN_CUSTOMER_NAMES:
        if name.lower() in lowered:
            return name

    for name in KNOWN_CUSTOMER_NAMES:
        first_name = name.split()[0].lower()
        if re.search(rf"\b{re.escape(first_name)}\b", lowered):
            return name

    for match in _POSSESSIVE_FIRST_NAME.finditer(cleaned):
        token = match.group(1).lower()
        for name in KNOWN_CUSTOMER_NAMES:
            if name.split()[0].lower() == token:
                return name

    return None
