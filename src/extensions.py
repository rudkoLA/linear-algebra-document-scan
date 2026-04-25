import numpy as np

from .homograph import (
    compute_homography_matrix,
    homography,
    sort_points,
    svd_confidence_metric,
)

try:
    import cv2
except ImportError:  # pragma: no cover - handled in runtime by explicit error
    cv2 = None


def compute_svd_confidence_from_design_matrix(A):
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"Design matrix must be 2D, got shape {A.shape}")

    singular_values = np.linalg.svd(A, compute_uv=False)
    rho = svd_confidence_metric(singular_values)
    return rho, singular_values


def compute_svd_confidence_for_points(input_points, width, height):
    _, diagnostics = compute_homography_matrix(
        input_points,
        width,
        height,
        return_diagnostics=True,
    )
    return diagnostics["rho"], diagnostics["singular_values"]


def auto_detect_document_corners(
    image,
    *,
    canny_threshold1=75,
    canny_threshold2=200,
    epsilon_ratio=0.02,
    min_area_ratio=0.08,
    return_edges=False,
):
    if cv2 is None:
        raise ImportError(
            "opencv-python-headless is required for edge-based corner detection"
        )

    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must have 2 or 3 dimensions, got {image.ndim}")
    if image.size == 0:
        raise ValueError("Image is empty")

    if image.ndim == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError(f"Unsupported channel count: {image.shape[2]}")
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in edge map")

    image_area = float(gray.shape[0] * gray.shape[1])
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_quad = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        area = cv2.contourArea(approx)
        if area < min_area_ratio * image_area:
            continue

        best_quad = approx.reshape(4, 2).astype(np.float64)
        break

    if best_quad is None:
        largest = contours[0]
        if cv2.contourArea(largest) < min_area_ratio * image_area:
            raise ValueError("Could not find a document-sized contour")

        rect = cv2.minAreaRect(largest)
        best_quad = cv2.boxPoints(rect).astype(np.float64)

    corners = sort_points(best_quad)
    if return_edges:
        return corners, edges
    return corners


def detect_and_warp_document(image, width, height, **detector_kwargs):
    corners = auto_detect_document_corners(image, **detector_kwargs)
    warped, diagnostics = homography(
        np.asarray(image),
        corners,
        width,
        height,
        return_diagnostics=True,
    )
    diagnostics = dict(diagnostics)
    diagnostics["detected_corners"] = corners
    return warped, diagnostics
