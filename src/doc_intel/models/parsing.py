"""Lenient parsers for values models print in many ways.

Models and OCR produce "1 234,50", "1,731.49", "9 pcs", "03 Aug 2026", "30.08.2026". The
schema wants a Decimal and a date. Rejecting the whole document for a thousands separator
would be silly, so these run before validation. They never guess: an unparseable value
still fails, and a separate repair round asks the model to fix it.
"""

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any

from pydantic import BeforeValidator

_AMOUNT = re.compile(r"[-+]?\d[\d\s.,'\u00a0]*\d|\d")
_MONTHS = {
    m: i
    for i, names in enumerate(
        (
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for m in names
}


def parse_amount(value: Any) -> Any:
    """'1 234,50' -> 1234.50; '1,731.49' -> 1731.49; '9 pcs' -> 9.

    Unparseable input is returned unchanged so pydantic reports it.
    """
    if value is None or isinstance(value, Decimal | int | float):
        return value
    if not isinstance(value, str):
        return value
    match = _AMOUNT.search(value.replace("\u00a0", " "))
    if not match:
        return value
    token = match.group(0).replace(" ", "").replace("'", "")
    if "," in token and "." in token:
        # whichever separator comes last is the decimal point
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "," in token:
        head, _, tail = token.rpartition(",")
        token = f"{head.replace(',', '')}.{tail}" if len(tail) in (1, 2) else token.replace(",", "")
    elif token.count(".") > 1:
        head, _, tail = token.rpartition(".")
        token = f"{head.replace('.', '')}.{tail}"
    try:
        return Decimal(token)
    except InvalidOperation:
        return value


def parse_date(value: Any) -> Any:
    """ISO first, then 30.08.2026 / 30/08/2026, then '03 Aug 2026' / 'Aug 3, 2026'."""
    if value is None or isinstance(value, date):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    for pattern in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y.%m.%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    words = re.findall(r"[A-Za-z]+|\d+", text)
    if len(words) == 3:
        month = next((_MONTHS[w.lower()] for w in words if w.lower() in _MONTHS), None)
        numbers = [int(w) for w in words if w.isdigit()]
        if month and len(numbers) == 2:
            day, year = sorted(numbers)
            if year > 31 and 1 <= day <= 31:
                try:
                    return date(year if year > 99 else 2000 + year, month, day)
                except ValueError:
                    return value
    return value


Money = Annotated[Decimal, BeforeValidator(parse_amount)]
Day = Annotated[date, BeforeValidator(parse_date)]
