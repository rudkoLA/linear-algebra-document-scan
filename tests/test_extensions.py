import numpy as np
import pytest

from src.extensions import (
    auto_detect_document_corners,
    compute_svd_confidence_for_points,
    compute_svd_confidence_from_design_matrix,
    detect_and_warp_document,
)
from src.homograph import sort_points

cv2 = pytest.importorskip("cv2")


def _make_detectable_document_image():
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    quad = np.array([[90, 70], [390, 45], [430, 305], [65, 325]], dtype=np.int32)

    cv2.fillConvexPoly(img, quad, (245, 245, 245))
    cv2.polylines(img, [quad], True, (20, 20, 20), 3)
    for y in range(105, 290, 24):
        cv2.line(img, (115, y), (380, y - 14), (80, 80, 80), 2)

    return img, quad.astype(np.float64)


def test_svd_confidence_metric_matches_known_values():
    expected_singulars = np.array([11.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 2.0, 1.0])
    A = np.diag(expected_singulars)

    rho, singular_values = compute_svd_confidence_from_design_matrix(A)
    assert np.allclose(singular_values, expected_singulars)
    assert np.isclose(rho, 2.0)


def test_svd_confidence_metric_for_homography_points():
    points = np.array(
        [
            [120.0, 70.0],
            [530.0, 95.0],
            [575.0, 365.0],
            [75.0, 330.0],
            [327.0, 69.0],
            [560.0, 225.0],
            [325.0, 365.0],
            [92.0, 210.0],
        ],
        dtype=np.float64,
    )

    rho, singular_values = compute_svd_confidence_for_points(points, width=600, height=400)
    assert rho > 0
    assert singular_values.ndim == 1
    assert singular_values.size >= 2


def test_auto_detect_document_corners_returns_expected_quad():
    image, true_quad = _make_detectable_document_image()
    detected = auto_detect_document_corners(image)

    detected = sort_points(detected)
    expected = sort_points(true_quad)

    point_error = np.linalg.norm(detected - expected, axis=1)
    assert point_error.mean() < 20.0
    assert point_error.max() < 35.0


def test_detect_and_warp_document_integration():
    image, _ = _make_detectable_document_image()

    warped, diagnostics = detect_and_warp_document(image, width=280, height=200)
    assert warped.shape == (200, 280, 3)
    assert diagnostics["detected_corners"].shape == (4, 2)
    assert diagnostics["homography_matrix"].shape == (3, 3)
