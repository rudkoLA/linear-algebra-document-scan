import numpy as np
import pytest


def _build_document_like_image(height, width, seed):
    rng = np.random.default_rng(seed)

    y, x = np.mgrid[0:height, 0:width]
    base = np.zeros((height, width, 3), dtype=np.float64)
    base[..., 0] = 150 + 60 * (x / max(width - 1, 1))
    base[..., 1] = 155 + 55 * (y / max(height - 1, 1))
    base[..., 2] = 145 + 45 * (x + y) / max(height + width - 2, 1)

    # Simulate text lines and margins to create strong benchmark features.
    margin = max(int(0.12 * width), 20)
    top = max(int(0.14 * height), 20)
    for row in range(top, height - top, max(int(0.06 * height), 10)):
        line_len = width - 2 * margin - int(rng.integers(10, max(12, int(0.2 * width))))
        line_len = max(line_len, int(0.4 * width))
        col_start = margin + int(rng.integers(0, max(1, int(0.05 * width))))
        col_end = min(width - margin, col_start + line_len)
        base[row : row + 2, col_start:col_end, :] = 50

    # Dark border to mimic paper edges.
    border = max(int(0.01 * min(height, width)), 2)
    base[:border, :, :] = 30
    base[-border:, :, :] = 30
    base[:, :border, :] = 30
    base[:, -border:, :] = 30

    noise = rng.normal(0.0, 2.5, size=base.shape)
    image = np.clip(base + noise, 0, 255).astype(np.uint8)
    return image


@pytest.fixture(scope="session")
def benchmark_document_images():
    specs = [
        (320, 460, 1),
        (380, 540, 2),
        (440, 660, 3),
    ]

    cases = []
    for height, width, seed in specs:
        img = _build_document_like_image(height, width, seed)
        pts = np.array(
            [
                [0.17 * width, 0.11 * height],
                [0.83 * width, 0.18 * height],
                [0.89 * width, 0.86 * height],
                [0.10 * width, 0.79 * height],
            ],
            dtype=np.float64,
        )
        cases.append(
            {
                "name": f"synthetic-{height}x{width}",
                "image": img,
                "points": pts,
                "dsize": (360, 260),
            }
        )

    return cases
