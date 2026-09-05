from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from doc_intel.api.app import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    from tests.api.fakes import FakeProcessor

    with TestClient(create_app(FakeProcessor())) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def en_receipt() -> tuple[object, object]:
    """Re-exported for tests outside tests/extract; see tests/extract/conftest.py."""
    import random

    from doc_intel.dataset.content import generate_content
    from doc_intel.dataset.layouts import render_pdf
    from doc_intel.dataset.render import pdf_to_image

    rng = random.Random(21)
    content = generate_content(rng, "en")
    return content, pdf_to_image(render_pdf(content, "receipt", rng), dpi=110)
