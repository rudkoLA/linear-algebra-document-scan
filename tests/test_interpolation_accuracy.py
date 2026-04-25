import numpy as np

from src.transformations import apply_transformation_matrix
from src.utils import interpolate_bilinear


def _manual_bilinear_pixel(img, x, y):
    h, w = img.shape[:2]

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x - x0
    dy = y - y0

    x0c = int(np.clip(x0, 0, w - 1))
    x1c = int(np.clip(x1, 0, w - 1))
    y0c = int(np.clip(y0, 0, h - 1))
    y1c = int(np.clip(y1, 0, h - 1))

    Ia = img[y0c, x0c].astype(np.float64)
    Ib = img[y0c, x1c].astype(np.float64)
    Ic = img[y1c, x0c].astype(np.float64)
    Id = img[y1c, x1c].astype(np.float64)

    value = (
        (1 - dx) * (1 - dy) * Ia
        + dx * (1 - dy) * Ib
        + (1 - dx) * dy * Ic
        + dx * dy * Id
    )
    return value.astype(img.dtype)


def test_interpolate_bilinear_known_scalar_value():
    img = np.array(
        [
            [[0, 0, 0], [100, 100, 100]],
            [[200, 200, 200], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    src_x = np.array([[0.25]], dtype=np.float64)
    src_y = np.array([[0.75]], dtype=np.float64)

    out = interpolate_bilinear(img, src_x, src_y)
    expected = _manual_bilinear_pixel(img, x=0.25, y=0.75)

    assert out.shape == (1, 1, 3)
    assert np.array_equal(out[0, 0], expected)


def test_apply_transformation_matrix_matches_manual_interpolation():
    img = np.array(
        [
            [[10, 15, 20], [30, 35, 40], [50, 55, 60]],
            [[70, 75, 80], [90, 95, 100], [110, 115, 120]],
            [[130, 135, 140], [150, 155, 160], [170, 175, 180]],
        ],
        dtype=np.uint8,
    )

    # Translation by (+0.5, +0.5) in destination space.
    H = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    out = apply_transformation_matrix(
        img,
        H,
        dsize=(2, 2),
        interpolation="bilinear",
    )

    expected = np.zeros((2, 2, 3), dtype=np.uint8)
    H_inv = np.linalg.inv(H)
    for y in range(2):
        for x in range(2):
            src = H_inv @ np.array([x, y, 1.0], dtype=np.float64)
            src_x = src[0] / src[2]
            src_y = src[1] / src[2]
            expected[y, x] = _manual_bilinear_pixel(img, src_x, src_y)

    assert np.array_equal(out, expected)
