"""Generate synthetic demo policy PDFs for the RAG pipeline.

Writes the synthetic refund, shipping, and warranty PDFs to `data/policies/`.
Run once during local setup, before indexing documents:

    python -m src.create_policy_pdfs
"""

from __future__ import annotations

from pathlib import Path

import fitz

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "policies"

POLICY_CONTENT: dict[str, str] = {
    "refund_policy.pdf": """Return and Refund Policy

Sample/Demo Policy Notice
This return and refund policy is synthetic content created for retrieval testing
and customer support demonstrations.

Cancellation Window
Customers may cancel or request a return within 14 days of receiving the goods.

Return Eligibility
To be eligible for a return:
- The goods must have been purchased within the last 14 days.
- The goods must be unused and in their original packaging.
- The customer must provide order details or proof of purchase.

Non-Returnable Items
The following items are not eligible for return:
- Personalized or custom-made products.
- Items that deteriorate quickly or are past their expiry date.
- Unsealed health, hygiene, or safety-related products.
- Items that are inseparably mixed with other goods after delivery.

Refund Processing
Approved refunds will be reimbursed no later than 14 days after the returned
goods are received. Refunds should be issued to the original payment method when
possible.

Return Shipping
Customers are responsible for the cost and risk of returning goods unless the
return is due to an error, damaged product, duplicate charge, or confirmed
support exception.

Damaged or Incorrect Items
If the item arrived damaged, incorrect, or defective, support should review the
customer's ticket, request photos if needed, and determine whether a replacement,
refund, or escalation is appropriate.

Support Executive Next Step
Verify the delivery date, return request date, item condition, proof of purchase,
and reason for return. If the request appears eligible, proceed with refund
review or replacement handling.
""",
    "warranty_policy.pdf": """Warranty Policy

Sample/Demo Policy Notice
This warranty policy is synthetic content created for retrieval testing and
customer support demonstrations.

Coverage Period
Products are covered for 12 months from the delivery date. The warranty window is
based on the confirmed delivery date associated with the customer's order.

Covered Items
The warranty covers manufacturing defects, hardware failures that occur under
normal use, and defective accessories included with the original shipment.

Items Not Covered
The warranty does not cover accidental damage, misuse, water damage,
unauthorized repairs, or normal wear and tear from regular product use.

Required Proof
Customers must provide the order number, purchase date, and product photos if the
item is damaged or visibly defective. Support may request additional photos when
the product condition is unclear.

Resolution Options
After inspection, eligible claims may be resolved with a repair, replacement, or
store credit. The final resolution depends on the inspection result, available
inventory, and the nature of the defect.

Support Process
Customers should contact support and provide their order details, a description
of the issue, and any available evidence. Support should confirm that the request
matches the warranty policy before offering a resolution.

Support Executive Next Step
Verify the purchase date, warranty window, product condition, and submitted
evidence. If the claim appears eligible, route it for inspection and prepare the
appropriate repair, replacement, or store credit option.
""",
    "shipping_policy.pdf": """Shipping and Delivery Policy

Sample/Demo Policy Notice
This shipping and delivery policy is synthetic content created for retrieval
testing and customer support demonstrations.

Standard Delivery Timeline
Standard delivery typically takes 5-7 business days after the order has shipped.
Delivery estimates may vary during holidays, severe weather, or carrier service
interruptions.

Delayed Delivery
If an order is more than 7 business days late, support should check the tracking
status and review the most recent carrier scan. Customers should be informed of
the current status and expected next action.

Lost Package
A package is considered lost if there has been no tracking update for 10
business days. Support should verify that the shipping address is correct and
confirm whether the carrier has opened an investigation.

Damaged Shipment
Customers reporting damaged shipments should provide photos of the outer
packaging, shipping label, and damaged item. Photos help support determine
whether the damage occurred during transit.

Resolution Options
Depending on the order status and carrier findings, support may offer a
replacement shipment, begin a refund review, or open a carrier investigation.

Support Process
Support should verify the order status, tracking number, delivery address, and
delay duration before deciding on a resolution. Notes should include the latest
carrier scan and any evidence provided by the customer.

Support Executive Next Step
Confirm the tracking status and decide whether to escalate to the carrier, send a
replacement, or review refund eligibility based on the delay or damage evidence.
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
    """Create synthetic demo policy PDF files in the given directory."""
    target_dir = output_dir or DEFAULT_OUTPUT_DIR
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
