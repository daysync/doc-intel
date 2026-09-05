"""Classical cleanup before OCR: grayscale, denoise, deskew, binarise.

No model involved. Deskew estimates the page angle from text lines: words are smeared
horizontally into line blobs, each blob's minimum-area rectangle gives an angle, and the
median of the long thin ones is the page skew. Robust to ±10° and to page borders.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

Gray = NDArray[np.uint8]


@dataclass
class Preprocessed:
    gray: Gray
    """Deskewed grayscale page, for engines that prefer tone over binary."""
    binary: Gray
    """Deskewed, adaptively thresholded page: black text on white."""
    angle_deg: float
    """Correction that was applied, in ``rotate()`` convention. Zero means the page was straight."""


def _u8(array: object) -> Gray:
    return np.asarray(array, dtype=np.uint8)


def to_gray(image: NDArray[np.uint8]) -> Gray:
    if image.ndim == 2:
        return _u8(image)
    return _u8(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))


def binarise(gray: Gray) -> Gray:
    """Adaptive threshold copes with uneven lighting where a global threshold cannot."""
    blurred = cv2.medianBlur(gray, 3)
    return _u8(
        cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )
    )


def estimate_skew(gray: Gray) -> float:
    """The rotation (``rotate()`` convention) that makes the text lines horizontal.

    If a page was rotated by ``rotate(page, a)``, this returns approximately ``-a``.
    """
    inverted = cv2.bitwise_not(binarise(gray))
    _, width = inverted.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 40), 3))
    smeared = cv2.dilate(inverted, kernel, iterations=1)
    contours, _ = cv2.findContours(smeared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles: list[float] = []
    for contour in contours:
        (_, _), (w, h), angle = cv2.minAreaRect(contour)
        if w < h:
            w, h = h, w
            angle += 90.0
        if w < width * 0.15 or h == 0 or w / h < 4:
            continue
        angle = ((angle + 45.0) % 90.0) - 45.0
        angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def rotate(gray: Gray, angle_deg: float) -> Gray:
    if abs(angle_deg) < 0.05:
        return gray
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1.0)
    return _u8(
        cv2.warpAffine(
            gray, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
    )


def preprocess(image: NDArray[np.uint8]) -> Preprocessed:
    gray = to_gray(image)
    angle = estimate_skew(gray)
    deskewed = rotate(gray, angle)
    return Preprocessed(gray=deskewed, binary=binarise(deskewed), angle_deg=angle)
