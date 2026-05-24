"""Load policy PDFs for RAG indexing (implementation layer).

Extracts page-level text with PyMuPDF. Chunking and embedding happen in
vector_store.py. MCP/agents call tools/document_tools.py, not this module.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz
from langchain_core.documents import Document

from src.config import get_settings


def infer_policy_type(filename: str) -> str:
    """Infer policy category from the PDF filename."""
    name = filename.lower()
    if "refund" in name:
        return "refund"
    if "warranty" in name:
        return "warranty"
    if "shipping" in name or "delivery" in name:
        return "shipping"
    if "cancellation" in name:
        return "cancellation"
    return "general"


def clean_text(text: str) -> str:
    """Normalize excessive whitespace while preserving paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_pages(file_path: str) -> list[Document]:
    """Extract page-level text from a PDF as LangChain Document objects."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {file_path}")

    policy_type = infer_policy_type(path.name)
    documents: list[Document] = []

    with fitz.open(path) as pdf:
        for page_index, page in enumerate(pdf, start=1):
            page_text = clean_text(page.get_text("text"))
            if not page_text:
                continue
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": path.name,
                        "page": page_index,
                        "policy_type": policy_type,
                    },
                )
            )

    if not documents:
        raise ValueError(f"No text could be extracted from PDF: {file_path}")

    return documents


def load_documents(policies_dir: Path | None = None) -> list[Document]:
    """Load page-level documents from all PDFs in the policies directory."""
    settings = get_settings()
    target_dir = policies_dir or settings.policies_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    documents: list[Document] = []
    for pdf_path in sorted(target_dir.glob("*.pdf")):
        documents.extend(load_pdf_pages(str(pdf_path)))
    return documents
