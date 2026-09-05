import random

import numpy as np
import pytest

from doc_intel.dataset.degrade import degrade
from doc_intel.ocr.preprocess import binarise, estimate_skew, preprocess, rotate, to_gray


def test_clean_page_has_near_zero_skew(clean_en_page: np.ndarray) -> None:
    assert abs(estimate_skew(to_gray(clean_en_page))) < 0.3


@pytest.mark.parametrize("angle", [-4.0, -1.5, 2.0, 5.0])
def test_deskew_recovers_a_known_rotation(clean_en_page: np.ndarray, angle: float) -> None:
    skewed = rotate(to_gray(clean_en_page), angle)
    assert abs(estimate_skew(skewed) + angle) < 0.6


def test_preprocess_on_a_hard_photo_removes_most_of_the_skew(clean_en_page: np.ndarray) -> None:
    photo = degrade(clean_en_page, "hard", random.Random(4))
    result = preprocess(photo.image)
    # degrade rotated by photo.angle_deg (cv2 convention); the correction is its negative
    assert abs(result.angle_deg + photo.angle_deg) < 1.0
    assert result.binary.dtype == np.uint8 and set(np.unique(result.binary)) <= {0, 255}


def test_binarise_is_black_text_on_white(clean_en_page: np.ndarray) -> None:
    binary = binarise(to_gray(clean_en_page))
    assert (binary == 255).mean() > 0.8
