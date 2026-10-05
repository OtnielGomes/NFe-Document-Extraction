"""Simulated photos: one degraded JPEG per DANFE page, fixed by a seed.

The pipeline follows the rendition definition: perspective, blur, shadow,
JPEG compression, then noise. The same PDF and seed always yield the same bytes.
"""

from __future__ import annotations

import io
import random

import pypdfium2 as pdfium  # type: ignore[import-untyped]
from PIL import Image, ImageChops, ImageFilter

PHOTO_RENDER_SCALE = 2
_PERSPECTIVE_MARGIN = 0.07
_BLUR_RADIUS = (1.0, 1.8)
_SHADOW_STRENGTH = (0.22, 0.38)
_JPEG_QUALITY = (40, 55)
_FINAL_JPEG_QUALITY = 80
_NOISE_AMOUNT = 14
_NOISE_SCALE = 16


def simulate_photo_pages(pdf: bytes, *, seed: int) -> tuple[bytes, ...]:
    """Degrade every page of a digital DANFE into one Simulated-photo JPEG."""
    rng = random.Random(seed)
    document = pdfium.PdfDocument(pdf)
    pages: list[bytes] = []
    try:
        for index in range(len(document)):
            page_rng = random.Random(rng.randrange(2**32))
            image = _render(document[index])
            image = _perspective(image, page_rng)
            image = image.filter(
                ImageFilter.GaussianBlur(radius=page_rng.uniform(*_BLUR_RADIUS))
            )
            image = _shadow(image, page_rng)
            image = _jpeg_roundtrip(image, page_rng.randint(*_JPEG_QUALITY))
            image = _noise(image, page_rng)
            pages.append(_encode_jpeg(image))
    finally:
        document.close()
    return tuple(pages)


def _render(page: object) -> Image.Image:
    bitmap = page.render(scale=PHOTO_RENDER_SCALE)  # type: ignore[attr-defined]
    try:
        return bitmap.to_pil().convert("RGB")
    finally:
        bitmap.close()
        page.close()  # type: ignore[attr-defined]


def _perspective(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    source = ((0, 0), (width, 0), (width, height), (0, height))
    target = tuple(_jitter(x, y, width, height, rng) for x, y in source)
    return image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        _perspective_coefficients(source, target),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )


def _jitter(
    x: float,
    y: float,
    width: int,
    height: int,
    rng: random.Random,
) -> tuple[float, float]:
    moved_x = x + rng.uniform(-_PERSPECTIVE_MARGIN, _PERSPECTIVE_MARGIN) * width
    moved_y = y + rng.uniform(-_PERSPECTIVE_MARGIN, _PERSPECTIVE_MARGIN) * height
    return (
        min(max(moved_x, 0.0), width - 1.0),
        min(max(moved_y, 0.0), height - 1.0),
    )


def _perspective_coefficients(
    source: tuple[tuple[float, float], ...],
    target: tuple[tuple[float, float], ...],
) -> tuple[float, ...]:
    """Map each output corner (target) back to the input corner (source)."""
    matrix: list[list[float]] = []
    ordinates: list[float] = []
    for (source_x, source_y), (target_x, target_y) in zip(source, target, strict=True):
        matrix.append(
            [
                target_x,
                target_y,
                1.0,
                0.0,
                0.0,
                0.0,
                -source_x * target_x,
                -source_x * target_y,
            ]
        )
        ordinates.append(source_x)
        matrix.append(
            [
                0.0,
                0.0,
                0.0,
                target_x,
                target_y,
                1.0,
                -source_y * target_x,
                -source_y * target_y,
            ]
        )
        ordinates.append(source_y)
    return tuple(_solve(matrix, ordinates))


def _solve(matrix: list[list[float]], ordinates: list[float]) -> list[float]:
    size = len(ordinates)
    rows = [row[:] + [ordinates[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        if abs(scale) < 1e-12:
            raise RuntimeError("perspective transform is singular")
        for index in range(column, size + 1):
            rows[column][index] /= scale
        for row in range(size):
            if row == column:
                continue
            factor = rows[row][column]
            for index in range(column, size + 1):
                rows[row][index] -= factor * rows[column][index]
    return [rows[index][size] for index in range(size)]


def _shadow(image: Image.Image, rng: random.Random) -> Image.Image:
    strength = rng.uniform(*_SHADOW_STRENGTH)
    gradient = Image.linear_gradient("L").resize(image.size)
    gradient = gradient.rotate(
        rng.uniform(0, 360),
        resample=Image.Resampling.BICUBIC,
        fillcolor=128,
    )
    table = [round(255 - strength * (255 - level)) for level in range(256)]
    shade = gradient.point(table)
    overlay = Image.merge("RGB", (shade, shade, shade))
    return ImageChops.multiply(image, overlay)


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    _save_jpeg(image, buffer, quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def _noise(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    sample_size = (max(1, width // _NOISE_SCALE), max(1, height // _NOISE_SCALE))
    raw = bytes(rng.randrange(256) for _ in range(sample_size[0] * sample_size[1]))
    grain = Image.frombytes("L", sample_size, raw).resize(
        image.size,
        Image.Resampling.BILINEAR,
    )
    table = [128 + ((level - 128) * _NOISE_AMOUNT) // 128 for level in range(256)]
    centered = grain.point(table)
    noise = Image.merge("RGB", (centered, centered, centered))
    return ImageChops.add(image, noise, scale=1.0, offset=-128)


def _encode_jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    _save_jpeg(image, buffer, _FINAL_JPEG_QUALITY)
    return buffer.getvalue()


def _save_jpeg(image: Image.Image, buffer: io.BytesIO, quality: int) -> None:
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=2,
        optimize=False,
        progressive=False,
    )
