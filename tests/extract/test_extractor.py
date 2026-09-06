from decimal import Decimal

import numpy as np
import pytest

from doc_intel.dataset.content import InvoiceContent
from doc_intel.extract.extractor import Extractor
from doc_intel.llm.errors import FixtureMissingError, StructuredOutputError
from doc_intel.llm.types import ImagePart, TextPart
from doc_intel.ocr.preprocess import preprocess
from doc_intel.ocr.tesseract import ocr_page
from tests.extract.conftest import MODEL, recorded_llm
from tests.llm.fakes import FakeLLM
from tests.models.factories import invoice


async def test_repair_round_feeds_validation_errors_back() -> None:
    good = invoice().model_dump_json()
    bad = good.replace('"value":"EUR"', '"value":"euro-ish"')  # currency validator rejects this
    assert bad != good
    llm = FakeLLM([bad, good])
    result = await Extractor(llm, "m", "text", max_repairs=1).extract(None, "some text")
    assert result.repairs == 1
    assert len(llm.calls) == 2
    repair_request = llm.calls[1][0]
    roles = [m.role for m in repair_request.messages]
    assert roles == ["user", "assistant", "user"]
    assert "currency" in repair_request.messages[2].parts[0].text  # type: ignore[union-attr]
    assert result.invoice.currency.value == "EUR"


async def test_gives_up_after_max_repairs() -> None:
    bad = invoice().model_dump_json().replace('"value":"EUR"', '"value":"euro-ish"')
    with pytest.raises(StructuredOutputError):
        await Extractor(FakeLLM([bad, bad]), "m", "text", max_repairs=1).extract(None, "t")


def test_request_shapes_per_strategy() -> None:
    image = np.full((20, 20, 3), 255, np.uint8)
    text_only = Extractor(FakeLLM(""), "m", "text").build_request(None, "hello")
    assert [type(p) for p in text_only.messages[0].parts] == [TextPart]
    image_only = Extractor(FakeLLM(""), "m", "image").build_request(image, None)
    assert [type(p) for p in image_only.messages[0].parts] == [ImagePart, TextPart]
    both = Extractor(FakeLLM(""), "m", "both").build_request(image, "hello")
    assert [type(p) for p in both.messages[0].parts] == [ImagePart, TextPart]
    assert "hint" in both.messages[0].parts[1].text  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        Extractor(FakeLLM(""), "m", "image").build_request(None, "x")


async def test_extracts_the_receipt_with_the_local_model(
    en_receipt: tuple[InvoiceContent, np.ndarray],
) -> None:
    content, image = en_receipt
    ocr_text = ocr_page(preprocess(image).binary, languages="eng").text
    extractor = Extractor(recorded_llm(), MODEL, "both")
    try:
        result = await extractor.extract(image, ocr_text)
    except FixtureMissingError as error:
        pytest.skip(f"no fixture yet: {error}")
    truth = content.invoice
    got = result.invoice
    assert got.number.value == truth.number.value
    assert got.totals.grand_total.value == truth.totals.grand_total.value
    assert got.currency.value == truth.currency.value
    # the 3B model invents dates (recorded: year 308 for "03 Aug 2026"); the validation
    # rules flag that, the extractor does not. Its quote is right, though:
    assert got.issue_date.quote and "03 Aug 2026" in got.issue_date.quote
    assert len(got.line_items) == len(truth.line_items)
    assert got.line_items[0].total.value == truth.line_items[0].total.value
    assert isinstance(got.totals.grand_total.value, Decimal)
    assert result.record.cost_usd == 0
    assert (
        result.repairs == 0
    )  # lenient parsing accepted "9 pcs" and "1,731.49" without a repair round
