from decimal import Decimal

from doc_intel.models import ProcessResult, Timings
from doc_intel.rag.chunking import chunk_document
from tests.models.factories import invoice


def _result(ocr_pages: list[str] | None = None) -> ProcessResult:
    return ProcessResult(
        document_id="doc-1",
        invoice=invoice(),
        issues=[],
        cost_usd=Decimal(0),
        timings=Timings(total_ms=1),
        ocr_pages=ocr_pages or [],
    )


def test_one_chunk_per_structural_unit() -> None:
    chunks = chunk_document(_result())
    kinds = [c.kind for c in chunks]
    assert kinds == ["header", "line_item", "line_item", "totals"]
    assert chunks[0].text.startswith("Invoice INV-1042")
    assert "Beauty Supplies Ltd" in chunks[0].text
    assert chunks[1].field_path == "line_items[0]" and "Shampoo 1L" in chunks[1].text
    assert "grand total 78.60 EUR" in chunks[3].text
    assert all(c.metadata["number"] == "INV-1042" for c in chunks)


def test_ocr_text_becomes_page_blocks() -> None:
    page = "\n".join(f"line {i}" for i in range(30))
    chunks = chunk_document(_result(ocr_pages=[page, "second page"]))
    ocr = [c for c in chunks if c.kind == "ocr"]
    assert [(c.page, c.field_path) for c in ocr] == [
        (1, "ocr[1][0]"),
        (1, "ocr[1][1]"),
        (1, "ocr[1][2]"),
        (2, "ocr[2][0]"),
    ]
    assert ocr[0].text.startswith("line 0") and ocr[2].text.endswith("line 29")
