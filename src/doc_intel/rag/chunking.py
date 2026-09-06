"""Structure-aware chunks.

An invoice is not prose, so chunks follow its structure: one header chunk (parties,
number, dates, currency), one chunk per line item, one totals chunk, and the raw OCR
text in page blocks for anything the schema did not capture. Every chunk names its
document, page, kind and field path, which is what a citation needs.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from doc_intel.models import Extracted, Invoice, ProcessResult

ChunkKind = Literal["header", "line_item", "totals", "ocr"]
OCR_BLOCK_LINES = 12


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    page: int = 1
    kind: ChunkKind
    field_path: str | None = None
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


def _shown(field: Extracted[Any]) -> str:
    """Prefer the printed quote; fall back to the normalised value."""
    if field.quote:
        return field.quote
    return "" if field.value is None else str(field.value)


def _party(
    label: str, name: Extracted[Any], address: Extracted[Any], tax_id: Extracted[Any]
) -> str:
    bits = [b for b in (_shown(name), _shown(address)) if b]
    if _shown(tax_id):
        bits.append(f"tax id {_shown(tax_id)}")
    return f"{label} {', '.join(bits)}" if bits else ""


def header_text(invoice: Invoice) -> str:
    supplier, buyer = invoice.supplier, invoice.buyer
    parts = [
        f"Invoice {_shown(invoice.number)}",
        f"issued {_shown(invoice.issue_date)}" if _shown(invoice.issue_date) else "",
        f"due {_shown(invoice.due_date)}" if _shown(invoice.due_date) else "",
        f"currency {_shown(invoice.currency)}" if _shown(invoice.currency) else "",
        _party("supplier", supplier.name, supplier.address, supplier.tax_id),
        _party("buyer", buyer.name, buyer.address, buyer.tax_id),
    ]
    return ". ".join(part for part in parts if part)


def line_item_text(invoice: Invoice, index: int) -> str:
    item = invoice.line_items[index]
    return (
        f"Invoice {_shown(invoice.number)}, line {index + 1}: {_shown(item.description)}; "
        f"quantity {_shown(item.quantity)} {_shown(item.unit)}; "
        f"unit price {_shown(item.unit_price)}; line total {_shown(item.total)}"
    )


def totals_text(invoice: Invoice) -> str:
    taxes = "; ".join(
        f"{_shown(t.name)} {_shown(t.rate)} = {_shown(t.amount)}" for t in invoice.taxes
    )
    return (
        f"Invoice {_shown(invoice.number)} totals: subtotal {_shown(invoice.totals.subtotal)}; "
        f"tax {taxes or _shown(invoice.totals.tax_total)}; "
        f"grand total {_shown(invoice.totals.grand_total)} {_shown(invoice.currency)}"
    )


def chunk_document(result: ProcessResult) -> list[Chunk]:
    invoice = result.invoice
    doc = result.document_id
    number = _shown(invoice.number)
    chunks = [
        Chunk(
            document_id=doc,
            kind="header",
            field_path="number",
            text=header_text(invoice),
            metadata={"number": number},
        )
    ]
    for index in range(len(invoice.line_items)):
        chunks.append(
            Chunk(
                document_id=doc,
                kind="line_item",
                field_path=f"line_items[{index}]",
                text=line_item_text(invoice, index),
                metadata={"number": number},
            )
        )
    chunks.append(
        Chunk(
            document_id=doc,
            kind="totals",
            field_path="totals",
            text=totals_text(invoice),
            metadata={"number": number},
        )
    )
    for page_index, page_text in enumerate(result.ocr_pages, start=1):
        lines = [line for line in page_text.splitlines() if line.strip()]
        for start in range(0, len(lines), OCR_BLOCK_LINES):
            block = "\n".join(lines[start : start + OCR_BLOCK_LINES])
            chunks.append(
                Chunk(
                    document_id=doc,
                    page=page_index,
                    kind="ocr",
                    field_path=f"ocr[{page_index}][{start // OCR_BLOCK_LINES}]",
                    text=block,
                    metadata={"number": number},
                )
            )
    return chunks
