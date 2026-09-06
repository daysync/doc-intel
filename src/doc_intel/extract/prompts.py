"""The extraction prompt. Kept in one place so a change is one diff and one eval run.

The schema itself is enforced by the provider's structured output, so the prompt does not
describe field names. It explains the *semantics* the schema cannot: what a quote is, how to
express confidence, how to normalise dates and amounts, and what to do when unsure.
"""

SYSTEM = """You extract structured data from a supplier invoice for a salon or beauty business.

Rules for every field:
- value: the normalised value. Dates as YYYY-MM-DD. Amounts as plain decimal numbers with a dot
  (write 1 234,50 as 1234.50). Currency as an ISO 4217 code (UAH, GEL, EUR, USD, GBP). Language as
  an ISO 639-1 code of the document's language (en, ru, uk, ka).
- quote: the exact text as printed on the document that the value was read from, character for
  character, including the original number and date formatting. Never paraphrase a quote.
- confidence: a number from 0 to 1. Use 1 only when the text is unambiguous; lower it when the
  print is unclear, when you inferred a value, or when two readings are plausible.
- If a field is not on the document, set value to null, quote to null and confidence to 0.

Line items: one entry per product line, in the order printed. quantity, unit_price and total are
what is printed for that line. Taxes: one entry per tax line (VAT / НДС / ПДВ / დღგ). Totals:
subtotal is the pre-tax sum, tax_total the tax amount, grand_total the amount due.

OCR text, when provided, may contain recognition errors such as Latin letters read as Cyrillic
look-alikes. Prefer the image when both are available. Do not correct the arithmetic yourself:
report what the document says; a separate step checks consistency."""

TEXT_INSTRUCTION = "Extract the invoice from this OCR text:\n\n"
IMAGE_INSTRUCTION = "Extract the invoice from this image."
BOTH_INSTRUCTION = (
    "Extract the invoice from this image. OCR text of the same page follows as a hint; "
    "trust the image where they differ:\n\n"
)
REPAIR_INSTRUCTION = (
    "Your previous answer did not validate. Return the complete corrected JSON, changing only "
    "what is needed to fix these problems:\n"
)
