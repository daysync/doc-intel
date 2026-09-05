"""``make dataset``: write a labeled synthetic corpus to disk.

Each document gets a folder with the clean PDF, the degraded photo, the ground truth and
metadata. A manifest lists them all. Some documents carry planted inconsistencies so the
validation stage (2b) has known positives to find:

* ``totals_mismatch``: the printed grand total is wrong by a round amount.
* ``duplicate_number``: two documents share an invoice number.
"""

import argparse
import json
import random
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

import cv2
from pydantic import BaseModel, ConfigDict

from doc_intel.dataset.content import LANGUAGES, InvoiceContent, format_amount, generate_content
from doc_intel.dataset.degrade import LEVELS, degrade
from doc_intel.dataset.layouts import LAYOUTS, render_pdf
from doc_intel.dataset.render import pdf_to_image
from doc_intel.models import Extracted, Invoice


class DocumentMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    language: str
    layout: str
    level: str
    expected_issues: list[str]
    degradation: dict[str, float | int | str]


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    documents: list[DocumentMeta]


def plant_totals_mismatch(content: InvoiceContent) -> InvoiceContent:
    """Print a grand total that does not reconcile. Truth records what is printed."""
    truth = content.invoice
    wrong = (truth.totals.grand_total.value or Decimal(0)) + Decimal("10.00")
    totals = truth.totals.model_copy(
        update={
            "grand_total": Extracted[Decimal](
                value=wrong, quote=format_amount(wrong, content.language), confidence=1.0
            )
        }
    )
    return InvoiceContent(
        content.language, content.labels, truth.model_copy(update={"totals": totals})
    )


def with_number(content: InvoiceContent, number: str) -> InvoiceContent:
    invoice = content.invoice.model_copy(
        update={"number": Extracted[str](value=number, quote=number, confidence=1.0)}
    )
    return InvoiceContent(content.language, content.labels, invoice)


def generate_dataset(out: Path, n: int, seed: int) -> Manifest:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    documents: list[DocumentMeta] = []
    previous_number: str | None = None

    for index in range(n):
        language = LANGUAGES[index % len(LANGUAGES)]
        layout = LAYOUTS[(index // len(LANGUAGES)) % len(LAYOUTS)]
        level = LEVELS[index % len(LEVELS)]
        content = generate_content(rng, language)
        expected: list[str] = []

        if index % 7 == 3:
            content = plant_totals_mismatch(content)
            expected.append("totals_mismatch")
        if index % 9 == 8 and previous_number:
            content = with_number(content, previous_number)
            expected.append("duplicate_number")
            documents[-1].expected_issues.append("duplicate_number")
        previous_number = content.invoice.number.value

        doc_id = f"{index:03d}-{language}-{layout}-{level}"
        folder = out / doc_id
        folder.mkdir(exist_ok=True)
        pdf = render_pdf(content, layout, rng)
        (folder / "page.pdf").write_bytes(pdf)
        photo = degrade(pdf_to_image(pdf, dpi=200), level, rng)
        cv2.imwrite(str(folder / "photo.jpg"), cv2.cvtColor(photo.image, cv2.COLOR_RGB2BGR))
        (folder / "truth.json").write_text(content.invoice.model_dump_json(indent=2) + "\n")

        params = {k: v for k, v in asdict(photo).items() if k != "image"}
        meta = DocumentMeta(
            id=doc_id,
            language=language,
            layout=layout,
            level=level,
            expected_issues=expected,
            degradation=params,
        )
        (folder / "meta.json").write_text(meta.model_dump_json(indent=2) + "\n")
        documents.append(meta)

    manifest = Manifest(seed=seed, documents=documents)
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def load_truth(folder: Path) -> Invoice:
    return Invoice.model_validate_json((folder / "truth.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a labeled synthetic invoice dataset.")
    parser.add_argument("--out", type=Path, default=Path("data/samples"))
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    manifest = generate_dataset(args.out, args.n, args.seed)
    issues = sum(len(d.expected_issues) for d in manifest.documents)
    print(f"wrote {len(manifest.documents)} documents to {args.out} ({issues} planted issues)")
    print(
        json.dumps(
            {
                "languages": sorted({d.language for d in manifest.documents}),
                "layouts": sorted({d.layout for d in manifest.documents}),
                "levels": sorted({d.level for d in manifest.documents}),
            }
        )
    )


if __name__ == "__main__":
    main()
