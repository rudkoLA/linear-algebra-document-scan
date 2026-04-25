import numpy as np
import pytest

from src.homograph import compute_homography_matrix
from src.transformations import apply_transformation_matrix

cv2 = pytest.importorskip("cv2")


def test_warp_matches_opencv_on_benchmark_set(benchmark_document_images):
    for case in benchmark_document_images:
        image = case["image"]
        points = case["points"]
        out_width, out_height = case["dsize"]

        H = compute_homography_matrix(points, width=out_width, height=out_height)

        ours = apply_transformation_matrix(
            image,
            H,
            dsize=(out_width, out_height),
            interpolation="bilinear",
        )
        opencv = cv2.warpPerspective(
            image,
            H.astype(np.float64),
            dsize=(out_width, out_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        diff = np.abs(ours.astype(np.int16) - opencv.astype(np.int16))
        assert diff.max() <= 3, f"{case['name']} max pixel diff too high: {diff.max()}"
        assert (
            diff.mean() <= 0.6
        ), f"{case['name']} mean pixel diff too high: {diff.mean():.4f}"
