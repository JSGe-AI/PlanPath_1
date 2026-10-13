"""Read fixed physical fields from a WSI pyramid."""

from __future__ import annotations

import math
from typing import Any

from PIL import Image


def field_size_level0(magnification: int) -> int:
    if magnification not in {5, 10, 20, 40}:
        raise ValueError(f"unsupported nominal magnification: {magnification}x")
    return max(1, int(4096 * 5 / magnification))


def read_field(
    slide: Any,
    x: int,
    y: int,
    magnification: int,
    output_pixels: int = 512,
) -> Image.Image:
    field_level0 = field_size_level0(magnification)
    target_downsample = max(1.0, 40.0 / magnification)
    level = slide.get_best_level_for_downsample(target_downsample)
    native_downsample = float(slide.level_downsamples[level])
    native_size = max(1, int(math.ceil(field_level0 / native_downsample)))
    return (
        slide.read_region((int(x), int(y)), level, (native_size, native_size))
        .convert("RGB")
        .resize((output_pixels, output_pixels), Image.Resampling.BICUBIC)
    )
