"""One extracted field: the value, the text it came from, and how sure we are.

Every leaf field of an extracted document is wrapped in ``Extracted[T]`` rather than
stored as a bare value with a parallel confidence dict. Three reasons:

* ``invoice.number.value`` / ``.quote`` / ``.confidence`` read naturally and nest into
  line items without inventing a path syntax.
* The JSON schema derived from the model demands a source quote for every field, which
  is what makes an LLM's extraction auditable.
* A review screen can walk the tree once (``iter_extracted``) and get a flat path -> score map.
"""

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Extracted[T](BaseModel):
    """A single extracted value with its provenance.

    ``value`` is ``None`` when the field was not found in the document. ``quote`` is the
    verbatim source text the value was read from; it may be present even when parsing the
    value failed, which is itself useful to a reviewer. Fields have no defaults on purpose:
    strict structured-output modes require every property to be listed as required.
    """

    model_config = ConfigDict(extra="forbid")

    value: T | None
    quote: str | None
    confidence: float = Field(ge=0.0, le=1.0)


def iter_extracted(model: BaseModel, prefix: str = "") -> Iterator[tuple[str, Extracted[Any]]]:
    """Yield ``(dotted_path, field)`` for every ``Extracted`` leaf inside ``model``.

    Paths look like ``"supplier.name"`` or ``"line_items[2].unit_price"``.
    """
    for name in type(model).model_fields:
        path = f"{prefix}.{name}" if prefix else name
        yield from _walk(getattr(model, name), path)


def _walk(obj: object, path: str) -> Iterator[tuple[str, Extracted[Any]]]:
    if isinstance(obj, Extracted):
        yield path, obj
    elif isinstance(obj, BaseModel):
        yield from iter_extracted(obj, path)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            yield from _walk(item, f"{path}[{index}]")
