import numpy as np
import pytest

from src.homograph import compute_homography_matrix, homography


def test_fewer_than_four_points_raises_value_error():
    points = np.array(
        [
            [10.0, 10.0],
            [40.0, 10.0],
            [10.0, 40.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="At least 4 points"):
        compute_homography_matrix(points, width=120, height=80)


def test_all_points_on_straight_line_raise_value_error():
    points = np.array(
        [
            [0.0, 0.0],
            [25.0, 25.0],
            [50.0, 50.0],
            [100.0, 100.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="collinear"):
        compute_homography_matrix(points, width=200, height=120)


def test_badly_shaped_quad_raises_value_error():
    points = np.array(
        [
            [0.0, 0.0],
            [500.0, 1.0],
            [520.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="collinear|near-zero area|skewed"):
        compute_homography_matrix(points, width=300, height=180)


def test_extreme_perspective_angles_produce_finite_result():
    h, w = 300, 420
    gradient = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    img = np.stack([gradient, gradient, gradient], axis=-1)

    points = np.array(
        [
            [180.0, 15.0],
            [220.0, 15.0],
            [390.0, 285.0],
            [10.0, 285.0],
        ],
        dtype=np.float64,
    )

    warped, diagnostics = homography(
        img,
        points,
        width=320,
        height=420,
        return_diagnostics=True,
    )

    assert warped.shape == (420, 320, 3)
    assert np.isfinite(warped).all()
    assert np.isfinite(diagnostics["homography_matrix"]).all()
    assert diagnostics["rho"] > 0
