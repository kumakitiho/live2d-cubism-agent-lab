from __future__ import annotations

from time import perf_counter

import pytest
from PIL import Image

from tools.asset_quality_evaluator import count_white_halo
from tools.hidden_region_completer import transparency_fill


@pytest.mark.benchmark
def test_transparency_fill_2048_wall_clock() -> None:
    size = (2048, 2048)
    part = Image.new("RGBA", size, (0, 0, 0, 0))
    part.putpixel((1023, 1024), (40, 80, 120, 255))
    inpaint = Image.new("L", size, 0)
    inpaint.paste(255, (1024, 1016, 1040, 1032))
    protect = Image.new("L", size, 0)

    started = perf_counter()
    result = transparency_fill(part, inpaint, protect, iterations=1)
    elapsed = perf_counter() - started

    assert result.getpixel((1024, 1024))[3] == 255
    assert elapsed < 3.0


@pytest.mark.benchmark
def test_white_halo_quality_2048_wall_clock() -> None:
    size = (2048, 2048)
    source = Image.new("RGBA", size, (0, 0, 0, 255))
    part = Image.new("RGBA", size, (255, 255, 255, 0))
    alpha = Image.new("L", size, 0)
    alpha.paste(255, (1, 1, 2047, 2047))
    part.putalpha(alpha)

    started = perf_counter()
    results = [count_white_halo(part, source) for _ in range(4)]
    elapsed = perf_counter() - started

    assert results == [8180] * 4
    assert elapsed < 3.0
