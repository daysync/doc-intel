"""Narrow a question to the document it names.

Most questions name an invoice number. Once that number is recognised, restricting search
to that document removes every sibling chunk from other invoices, which is what hurt the
header questions in the 3a baseline. If nothing matches, search stays global.
"""

import re

from doc_intel.rag.index import Retriever

# INV-2026-0917, F-2026-3165, SF 2026/4609, №12345, "invoice 4587"
_LABELS = r"invoice|inv|№|no\.?|number|рахунок|счёт|счет|ინვოისი"
_NUMBER = re.compile(
    r"\b([A-Z]{1,4}[-/ ]?\d{2,4}[-/ ]\d{2,6}|[A-Z]{1,4}-\d{3,8})\b"  # INV-2026-0917, INV-7001
    rf"|(?:{_LABELS})\s*[#:№]?\s*([A-Z0-9][A-Z0-9-/]{{3,}})",  # "invoice 4587", "№ 12345"
    re.IGNORECASE,
)


def invoice_numbers_in(question: str) -> list[str]:
    found: list[str] = []
    for match in _NUMBER.finditer(question):
        token = (match.group(1) or match.group(2) or "").strip(" .,?!:;")
        if token and any(ch.isdigit() for ch in token) and token not in found:
            found.append(token)
    return found


async def resolve_scope(question: str, retriever: Retriever) -> list[str] | None:
    """Document ids the question refers to, or None for a global search."""
    for number in invoice_numbers_in(question):
        documents = await retriever.documents_with_number(number)
        if documents:
            return documents
    return None
