"""PDF page to raster image, the step between "a document" and "a photo of a document"."""

import numpy as np
import pypdfium2 as pdfium
from numpy.typing import NDArray


def pdf_to_image(pdf: bytes, dpi: int = 200, page_index: int = 0) -> NDArray[np.uint8]:
    """Render one page to an RGB uint8 array (H, W, 3)."""
    document = pdfium.PdfDocument(pdf)
    try:
        page = document[page_index]
        bitmap = page.render(scale=dpi / 72)
        image = bitmap.to_pil().convert("RGB")
    finally:
        document.close()
    return np.asarray(image, dtype=np.uint8)  # pypdfium2 is untyped; the PIL round-trip pins dtype
