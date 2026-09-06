"""Field-level accuracy of an extracted Invoice against ground truth.

Exact match for identifiers, dates, currency and language; tolerance for amounts; names
compared after normalising case and punctuation. Line items are compared by position.
The result is a flat ``{field: bool}`` map so aggregation over documents is a mean.
"""

import re
from decimal import Decimal
from typing import Any

from doc_intel.models import Invoice

AMOUNT_TOLERANCE = Decimal("0.01")
# digits, Latin, Cyrillic (with the extra Ukrainian and Russian letters) and Georgian survive
_NOT_LETTER = re.compile(r"[^0-9a-z\u0430-\u044f\u0451\u0456\u0457\u0454\u0491\u10d0-\u10f0]+")
KEY_FIELDS = (
    "number",
    "issue_date",
    "due_date",
    "currency",
    "language",
    "supplier.name",
    "supplier.tax_id",
    "buyer.name",
    "totals.subtotal",
    "totals.tax_total",
    "totals.grand_total",
)


def _norm_text(value: object) -> str:
    return re.sub(
        r"[^0-9a-z\u0430-\u044f\u0451\u0456\u0457\u0454\u0491\u10d0-\u10f0]+",
        "",
        str(value).casefold(),
    )


def _same(field: str, truth: object, got: object) -> bool:
    if truth is None and got is None:
        return True
    if truth is None or got is None:
        return False
    if isinstance(truth, Decimal) and isinstance(got, Decimal):
        return abs(truth - got) <= AMOUNT_TOLERANCE
    if field.endswith((".name", ".address", ".description")):
        return _norm_text(truth) == _norm_text(got)
    if field.endswith(("number", "tax_id")):
        return _norm_text(truth) == _norm_text(got)
    return truth == got


def _get(invoice: Invoice, path: str) -> Any:
    node: Any = invoice
    for part in path.split("."):
        node = getattr(node, part)
    return node.value


def compare(truth: Invoice, got: Invoice) -> dict[str, bool]:
    """Per-field correctness. Keys are stable so results can be averaged across documents."""
    scores = {field: _same(field, _get(truth, field), _get(got, field)) for field in KEY_FIELDS}
    scores["line_items.count"] = len(truth.line_items) == len(got.line_items)
    for index, item in enumerate(truth.line_items):
        other = got.line_items[index] if index < len(got.line_items) else None
        for name in ("description", "quantity", "unit_price", "total"):
            key = f"line_items.{name}"
            truth_value = getattr(item, name).value
            got_value = getattr(other, name).value if other else None
            scores.setdefault(key, True)
            scores[key] = scores[key] and _same(f".{name}", truth_value, got_value)
    return scores


def accuracy(score_maps: list[dict[str, bool]]) -> dict[str, float]:
    """Mean correctness per field over documents, plus an overall mean over all fields."""
    if not score_maps:
        return {}
    fields = sorted({key for scores in score_maps for key in scores})
    per_field = {f: sum(1 for s in score_maps if s.get(f, False)) / len(score_maps) for f in fields}
    per_field["all_fields"] = sum(per_field[f] for f in fields) / len(fields)
    return per_field
