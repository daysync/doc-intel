"""Three invoice layout families rendered with ReportLab.

Every string drawn on the page is a ``quote`` from the ground-truth Invoice or a label, so
the rendered document and the truth agree character for character. Variation within a
family (font, accent colour, column widths) comes from the seeded rng.
"""

import random
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from doc_intel.dataset.content import InvoiceContent
from doc_intel.models import Extracted

FONT_DIR = Path(__file__).resolve().parents[3] / "data" / "fonts"
FONTS = {
    "sans": "DejaVuSans.ttf",
    "sans-bold": "DejaVuSans-Bold.ttf",
    "serif": "DejaVuSerif.ttf",
    "mono": "DejaVuSansMono.ttf",
}
LAYOUTS = ("classic", "compact", "receipt")

_registered = False


def register_fonts() -> None:
    global _registered
    if _registered:
        return
    for name, file in FONTS.items():
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
    _registered = True


def q(field: Extracted[Any]) -> str:
    """The printed form of a field. Ground truth always has a quote."""
    assert field.quote is not None
    return field.quote


def render_pdf(content: InvoiceContent, layout: str, rng: random.Random) -> bytes:
    register_fonts()
    buffer = BytesIO()
    LAYOUT_FUNCTIONS[layout](content, rng, buffer)
    return buffer.getvalue()


# --- classic: A4, title top-right, supplier/buyer blocks, ruled table, totals bottom-right


def _classic(content: InvoiceContent, rng: random.Random, buffer: BytesIO) -> None:
    inv, labels = content.invoice, content.labels
    body = rng.choice(("sans", "serif"))
    accent = rng.choice(
        (colors.HexColor("#1f3a5f"), colors.HexColor("#333333"), colors.HexColor("#7a1f1f"))
    )
    c = Canvas(buffer, pagesize=A4, invariant=1)
    width, height = A4
    left, right, top = 20 * mm, width - 20 * mm, height - 20 * mm

    c.setFont("sans-bold", 22)
    c.setFillColor(accent)
    c.drawRightString(right, top, labels["title"])
    c.setFillColor(colors.black)

    c.setFont("sans-bold", 11)
    c.drawString(left, top, q(inv.supplier.name))
    c.setFont(body, 9)
    c.drawString(left, top - 5 * mm, q(inv.supplier.address))
    c.drawString(left, top - 10 * mm, f"{labels['tax_id']}: {q(inv.supplier.tax_id)}")

    y = top - 22 * mm
    c.setFont("sans-bold", 9)
    c.drawString(left, y, f"{labels['buyer']}:")
    c.setFont(body, 9)
    c.drawString(left, y - 5 * mm, q(inv.buyer.name))
    c.drawString(left, y - 10 * mm, q(inv.buyer.address))
    c.drawString(left, y - 15 * mm, f"{labels['tax_id']}: {q(inv.buyer.tax_id)}")

    meta_x = right - 60 * mm
    for i, (label, value) in enumerate(
        (
            (labels["number"], q(inv.number)),
            (labels["date"], q(inv.issue_date)),
            (labels["due"], q(inv.due_date)),
        )
    ):
        c.setFont("sans-bold", 9)
        c.drawString(meta_x, y - i * 5 * mm, f"{label}:")
        c.setFont(body, 9)
        c.drawRightString(right, y - i * 5 * mm, value)

    y -= 28 * mm
    cols = (left, left + 90 * mm, left + 110 * mm, left + 125 * mm, right)
    c.setFont("sans-bold", 9)
    c.setFillColor(accent)
    c.drawString(cols[0], y, labels["item"])
    c.drawRightString(cols[1] + 10 * mm, y, labels["qty"])
    c.drawString(cols[2], y, labels["unit"])
    c.drawRightString(cols[3] + 22 * mm, y, labels["price"])
    c.drawRightString(cols[4], y, labels["total"])
    c.setFillColor(colors.black)
    c.setStrokeColor(accent)
    c.line(left, y - 2 * mm, right, y - 2 * mm)

    c.setFont(body, 9)
    for item in inv.line_items:
        y -= 7 * mm
        c.drawString(cols[0], y, q(item.description))
        c.drawRightString(cols[1] + 10 * mm, y, q(item.quantity))
        c.drawString(cols[2], y, q(item.unit))
        c.drawRightString(cols[3] + 22 * mm, y, q(item.unit_price))
        c.drawRightString(cols[4], y, q(item.total))

    y -= 12 * mm
    c.setStrokeColor(colors.grey)
    c.line(right - 70 * mm, y + 5 * mm, right, y + 5 * mm)
    tax = inv.taxes[0]
    rows = (
        (labels["subtotal"], q(inv.totals.subtotal)),
        (f"{q(tax.name)} {q(tax.rate)}", q(tax.amount)),
        (f"{labels['grand_total']} {q(inv.currency)}", q(inv.totals.grand_total)),
    )
    for i, (label, value) in enumerate(rows):
        font = "sans-bold" if i == len(rows) - 1 else body
        c.setFont(font, 10 if i == len(rows) - 1 else 9)
        c.drawString(right - 70 * mm, y - i * 6 * mm, label)
        c.drawRightString(right, y - i * 6 * mm, value)
    c.showPage()
    c.save()


# --- compact: A4, two-column header, one-line meta strip, zebra table, boxed total


def _compact(content: InvoiceContent, rng: random.Random, buffer: BytesIO) -> None:
    inv, labels = content.invoice, content.labels
    body = rng.choice(("sans", "serif"))
    shade = rng.choice(
        (colors.HexColor("#eef2f7"), colors.HexColor("#f4f4f4"), colors.HexColor("#fdf3e7"))
    )
    c = Canvas(buffer, pagesize=A4, invariant=1)
    width, height = A4
    left, right, top = 15 * mm, width - 15 * mm, height - 15 * mm

    c.setFont("sans-bold", 16)
    c.drawString(left, top, f"{labels['title']} {q(inv.number)}")
    c.setFont(body, 9)
    c.drawRightString(
        right, top, f"{labels['date']}: {q(inv.issue_date)}   {labels['due']}: {q(inv.due_date)}"
    )

    y = top - 12 * mm
    c.setFillColor(shade)
    c.rect(left, y - 22 * mm, right - left, 24 * mm, fill=1, stroke=0)
    c.setFillColor(colors.black)
    mid = (left + right) / 2
    for x, title, party in (
        (left + 3 * mm, labels["supplier"], inv.supplier),
        (mid, labels["buyer"], inv.buyer),
    ):
        c.setFont("sans-bold", 8)
        c.drawString(x, y - 2 * mm, title.upper())
        c.setFont(body, 9)
        c.drawString(x, y - 8 * mm, q(party.name))
        c.drawString(x, y - 13 * mm, q(party.address))
        c.drawString(x, y - 18 * mm, f"{labels['tax_id']} {q(party.tax_id)}")

    y -= 34 * mm
    c.setFont("sans-bold", 8)
    xs = (left + 2 * mm, left + 95 * mm, left + 112 * mm, left + 140 * mm, right - 2 * mm)
    for x, label, align_right in (
        (xs[0], labels["item"], False),
        (xs[1], labels["qty"], True),
        (xs[2], labels["unit"], False),
        (xs[3], labels["price"], True),
        (xs[4], labels["total"], True),
    ):
        (c.drawRightString if align_right else c.drawString)(x, y, label)
    c.setFont(body, 9)
    for i, item in enumerate(inv.line_items):
        y -= 6.5 * mm
        if i % 2 == 0:
            c.setFillColor(shade)
            c.rect(left, y - 2 * mm, right - left, 6.5 * mm, fill=1, stroke=0)
            c.setFillColor(colors.black)
        c.drawString(xs[0], y, q(item.description))
        c.drawRightString(xs[1], y, q(item.quantity))
        c.drawString(xs[2], y, q(item.unit))
        c.drawRightString(xs[3], y, q(item.unit_price))
        c.drawRightString(xs[4], y, q(item.total))

    y -= 14 * mm
    tax = inv.taxes[0]
    c.setFont(body, 9)
    c.drawRightString(right - 30 * mm, y, f"{labels['subtotal']}:")
    c.drawRightString(xs[4], y, q(inv.totals.subtotal))
    c.drawRightString(right - 30 * mm, y - 5 * mm, f"{q(tax.name)} {q(tax.rate)}:")
    c.drawRightString(xs[4], y - 5 * mm, q(tax.amount))
    c.setStrokeColor(colors.black)
    c.rect(right - 105 * mm, y - 15 * mm, 105 * mm, 8 * mm, fill=0, stroke=1)
    c.setFont("sans-bold", 9)
    c.drawString(right - 103 * mm, y - 12.5 * mm, f"{labels['grand_total']} ({q(inv.currency)})")
    c.drawRightString(xs[4], y - 12.5 * mm, q(inv.totals.grand_total))
    c.showPage()
    c.save()


# --- receipt: 80 mm thermal-printer style, monospace, dashed rules


def _receipt(content: InvoiceContent, rng: random.Random, buffer: BytesIO) -> None:
    inv, labels = content.invoice, content.labels
    width = 80 * mm
    line_h = 4.2 * mm
    lines = 22 + 2 * len(inv.line_items)
    height = lines * line_h + 20 * mm
    c = Canvas(buffer, pagesize=(width, height), invariant=1)
    x, y = 4 * mm, height - 8 * mm
    dash = "-" * 38

    def out(text: str, bold: bool = False, center: bool = False) -> None:
        nonlocal y
        c.setFont("sans-bold" if bold else "mono", 8.5 if bold else 8)
        (c.drawCentredString(width / 2, y, text) if center else c.drawString(x, y, text))
        y -= line_h

    out(q(inv.supplier.name), bold=True, center=True)
    out(q(inv.supplier.address), center=True)
    out(f"{labels['tax_id']} {q(inv.supplier.tax_id)}", center=True)
    out(dash)
    out(f"{labels['title']} {q(inv.number)}", bold=True)
    out(f"{labels['date']}: {q(inv.issue_date)}")
    out(f"{labels['due']}: {q(inv.due_date)}")
    out(f"{labels['buyer']}: {q(inv.buyer.name)}")
    out(f"{labels['tax_id']} {q(inv.buyer.tax_id)}")
    out(dash)
    for item in inv.line_items:
        out(q(item.description))
        out(
            f"  {q(item.quantity)} {q(item.unit)} x {q(item.unit_price)}".ljust(26)
            + q(item.total).rjust(12)
        )
    out(dash)
    tax = inv.taxes[0]
    out(f"{labels['subtotal']}".ljust(26) + q(inv.totals.subtotal).rjust(12))
    out(f"{q(tax.name)} {q(tax.rate)}".ljust(26) + q(tax.amount).rjust(12))
    out(
        f"{labels['grand_total']} {q(inv.currency)}".ljust(22)
        + q(inv.totals.grand_total).rjust(16),
        bold=True,
    )
    out(dash)
    c.showPage()
    c.save()


LAYOUT_FUNCTIONS: dict[str, Callable[[InvoiceContent, random.Random, BytesIO], None]] = {
    "classic": _classic,
    "compact": _compact,
    "receipt": _receipt,
}
