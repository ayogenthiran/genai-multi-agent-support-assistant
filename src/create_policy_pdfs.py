"""Generate sample policy PDFs for the RAG pipeline.

Writes three demo PDFs (refund policy, shipping policy, warranty policy) to
`data/policies/`. Run once during local setup, before indexing documents:

    python -m src.create_policy_pdfs

These files are gitignored and regenerated on demand — same pattern as
`create_dummy_data.py` for `data/customers.db`.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from src.config import get_settings

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "policies"

POLICY_CONTENT: dict[str, str] = {
    "refund_policy.pdf": """Refund Policy

Customers may request a full refund within 30 days of delivery for unused items
in original packaging. Premium customers receive an extended 45-day refund window.

Refunds are not available for digital downloads after access has been granted.
Duplicate orders are eligible for a full refund once the duplicate shipment is
confirmed by support.

Approved refunds are issued to the original payment method within 5-7 business days.
Partial refunds may apply when items are returned damaged or missing accessories.
""",
    "shipping_policy.pdf": """Shipping Policy

Standard shipping delivers within 5-7 business days. Premium customers receive
free expedited shipping on orders over $50.

If a shipment is more than 3 business days late, support may offer a shipping
credit or priority reshipment at no additional cost.

International orders may require 10-14 business days and are subject to customs
fees paid by the customer.
""",
    "warranty_policy.pdf": """Warranty Policy

Electronics include a 12-month limited warranty covering manufacturing defects.
Premium customers receive an additional 6 months of warranty coverage.

Battery replacements are covered when capacity drops below 50% within the warranty
period. Cosmetic damage, liquid damage, and unauthorized repairs are excluded.

To start a warranty claim, contact support with the order number and a brief
description of the issue. Approved claims receive repair, replacement, or store
credit at the company's discretion.
""",
}


def _write_pdf(output_path: Path, text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(50, 50, 545, 792),
        text.strip(),
        fontsize=11,
        fontname="helv",
    )
    document.save(output_path)
    document.close()


def create_policy_pdfs(output_dir: Path | None = None) -> list[Path]:
    """Create sample policy PDF files in the given directory."""
    settings = get_settings()
    target_dir = output_dir or settings.policies_dir
    created: list[Path] = []

    for filename, content in POLICY_CONTENT.items():
        path = target_dir / filename
        _write_pdf(path, content)
        created.append(path)

    return created


def main() -> None:
    """CLI entry point for generating demo policy PDFs."""
    paths = create_policy_pdfs()
    for path in paths:
        print(f"Created {path}")


if __name__ == "__main__":
    main()
