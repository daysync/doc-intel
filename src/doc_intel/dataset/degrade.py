"""Make a clean render look like a phone photo.

Each level applies a fixed recipe with seeded randomness: rotation, perspective, blur,
uneven lighting, sensor noise, JPEG compression. The parameters used are returned so the
preprocessing stage can be scored against them (did deskew recover the angle?).
"""

import random
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

LEVELS = ("clean", "mild", "hard")
Image = NDArray[np.uint8]


def _u8(array: object) -> Image:
    """OpenCV returns loosely typed arrays; pin them to uint8 for the type checker and ourselves."""
    return np.asarray(array, dtype=np.uint8)


@dataclass
class Degradation:
    level: str
    angle_deg: float = 0.0
    perspective: float = 0.0
    blur_px: int = 0
    noise_sigma: float = 0.0
    lighting: float = 0.0
    jpeg_quality: int = 100
    image: Image = field(default_factory=lambda: np.zeros((1, 1, 3), np.uint8), repr=False)


def degrade(image: Image, level: str, rng: random.Random) -> Degradation:
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}")
    if level == "clean":
        return Degradation(level=level, image=_jpeg(image, 95))

    hard = level == "hard"
    d = Degradation(
        level=level,
        angle_deg=rng.uniform(-5.0, 5.0) if hard else rng.uniform(-2.0, 2.0),
        perspective=rng.uniform(0.02, 0.05) if hard else rng.uniform(0.0, 0.015),
        blur_px=rng.choice((3, 5)) if hard else rng.choice((0, 3)),
        noise_sigma=rng.uniform(8, 14) if hard else rng.uniform(3, 6),
        lighting=rng.uniform(0.35, 0.55) if hard else rng.uniform(0.1, 0.25),
        jpeg_quality=rng.randint(50, 65) if hard else rng.randint(75, 88),
    )
    out = _pad(image, 0.06)
    out = _rotate(out, d.angle_deg)
    out = _perspective(out, d.perspective, rng)
    if d.blur_px:
        out = _u8(cv2.GaussianBlur(out, (d.blur_px, d.blur_px), 0))
    out = _lighting(out, d.lighting, rng)
    out = _noise(out, d.noise_sigma, rng)
    d.image = _jpeg(out, d.jpeg_quality)
    return d


def _pad(image: Image, fraction: float) -> Image:
    h, w = image.shape[:2]
    pad_h, pad_w = int(h * fraction), int(w * fraction)
    border = cv2.copyMakeBorder(
        image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(235, 232, 226)
    )
    return _u8(border)


def _rotate(image: Image, angle: float) -> Image:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return _u8(cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE))


def _perspective(image: Image, strength: float, rng: random.Random) -> Image:
    if strength <= 0:
        return image
    h, w = image.shape[:2]

    def jitter() -> float:
        return rng.uniform(-strength, strength)

    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = np.array(
        [
            [w * jitter(), h * jitter()],
            [w * (1 + jitter()), h * jitter()],
            [w * (1 + jitter()), h * (1 + jitter())],
            [w * jitter(), h * (1 + jitter())],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return _u8(cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE))


def _lighting(image: Image, strength: float, rng: random.Random) -> Image:
    """Multiply by a linear gradient in a random direction: one side lit, one in shadow."""
    h, w = image.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    direction = rng.uniform(0, 2 * np.pi)
    ramp = (xs / w) * np.cos(direction) + (ys / h) * np.sin(direction)
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min() + 1e-6)
    gain = 1.0 - strength * ramp
    return _u8(np.clip(image.astype(np.float32) * gain[..., None], 0, 255))


def _noise(image: Image, sigma: float, rng: random.Random) -> Image:
    generator = np.random.default_rng(rng.getrandbits(32))
    noise = generator.normal(0, sigma, image.shape).astype(np.float32)
    return _u8(np.clip(image.astype(np.float32) + noise, 0, 255))


def _jpeg(image: Image, quality: int) -> Image:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert decoded is not None
    return _u8(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
