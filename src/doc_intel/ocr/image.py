"""Bytes in, RGB pages out. The only place that knows about file formats.

Phone photos are the primary input: EXIF orientation is honoured and HEIC (the iPhone
default) is decoded via pillow-heif. PDFs become one image per page.
"""

from io import BytesIO

import numpy as np
import pillow_heif
from numpy.typing import NDArray
from PIL import Image, ImageOps

from doc_intel.dataset.render import pdf_to_image

pillow_heif.register_heif_opener()

Rgb = NDArray[np.uint8]

IMAGE_MIMES = frozenset({"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"})
PDF_MIME = "application/pdf"


def load_pages(data: bytes, mime: str, dpi: int = 200, max_pages: int = 10) -> list[Rgb]:
    if mime == PDF_MIME:
        return _pdf_pages(data, dpi, max_pages)
    if mime in IMAGE_MIMES:
        with Image.open(BytesIO(data)) as image:
            upright = ImageOps.exif_transpose(image) or image
            return [np.asarray(upright.convert("RGB"), dtype=np.uint8)]
    raise ValueError(f"unsupported document type {mime!r}")


def _pdf_pages(data: bytes, dpi: int, max_pages: int) -> list[Rgb]:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(data)
    try:
        count = min(len(document), max_pages)
    finally:
        document.close()
    return [pdf_to_image(data, dpi=dpi, page_index=index) for index in range(count)]


def to_jpeg(image: Rgb, quality: int = 90) -> bytes:
    """Encode for sending to a vision model. JPEG keeps payloads small."""
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
