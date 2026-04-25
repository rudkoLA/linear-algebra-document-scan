import math
import numpy as np
from .transformations import apply_transformation_matrix


def _validate_point_set(points, min_points=4, min_rank=2):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"Points must be an array of shape (N, 2), got {points.shape}"
        )

    if points.shape[0] < min_points:
        raise ValueError(f"At least {min_points} points are required, got {points.shape[0]}")

    if not np.isfinite(points).all():
        raise ValueError("Points contain NaN or infinite values")

    centered = points - np.mean(points, axis=0)
    if np.linalg.matrix_rank(centered) < min_rank:
        raise ValueError("Points are collinear or nearly collinear")

    return points


def _polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(p1, q1, p2, q2):
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    return (o1 * o2 < 0) and (o3 * o4 < 0)


def _is_self_intersecting_quad(quad):
    return _segments_intersect(quad[0], quad[1], quad[2], quad[3]) or _segments_intersect(
        quad[1], quad[2], quad[3], quad[0]
    )


def _min_quad_angle_degrees(quad):
    min_angle = 180.0
    for i in range(4):
        prev_pt = quad[(i - 1) % 4]
        curr_pt = quad[i]
        next_pt = quad[(i + 1) % 4]

        v1 = prev_pt - curr_pt
        v2 = next_pt - curr_pt
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0

        cos_theta = np.dot(v1, v2) / (n1 * n2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        min_angle = min(min_angle, angle)

    return float(min_angle)

def generate_dst_points(src_pts, width, height):
    src_pts = _validate_point_set(src_pts, min_points=4)

    center = np.mean(src_pts, axis=0)
    angles = np.array([math.atan2(p[1] - center[1], p[0] - center[0]) for p in src_pts])
    order = np.argsort(angles)
    sorted_pts = src_pts[order]

    n = len(sorted_pts)
    quarter = n / 4
    corners = []
    corner_src_indices = []
    for i in range(4):
        start = int(round(i * quarter))
        end = int(round((i + 1) * quarter))
        if end <= start:
            end = start + 1
        group = sorted_pts[start:end]
        group_indices = order[start:end]
        dists = np.linalg.norm(group - center, axis=1)
        best = np.argmax(dists)
        corners.append(group[best])
        corner_src_indices.append(group_indices[best])

    corners = np.array(corners)

    tl = corners[np.argmin(corners[:, 0] + corners[:, 1])]
    tr = corners[np.argmax(corners[:, 0] - corners[:, 1])]
    br = corners[np.argmax(corners[:, 0] + corners[:, 1])]
    bl = corners[np.argmin(corners[:, 0] - corners[:, 1])]
    quad = np.array([tl, tr, br, bl], dtype=np.float64)

    if _polygon_area(quad) < 1e-6:
        raise ValueError("Detected quadrilateral has near-zero area")
    if _is_self_intersecting_quad(quad):
        raise ValueError("Detected quadrilateral is self-intersecting")
    if _min_quad_angle_degrees(quad) < 3.0:
        raise ValueError("Detected quadrilateral is too skewed for stable homography")

    quad_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    rect_edges = [
        (np.array([0.0, 0.0]), np.array([width, 0.0])),
        (np.array([width, 0.0]), np.array([width, height])),
        (np.array([width, height]), np.array([0.0, height])),
        (np.array([0.0, height]), np.array([0.0, 0.0])),
    ]

    dst_pts = []
    for pt in src_pts:
        best_t = None
        best_edge = None
        best_dist = np.inf

        for edge_i, (i, j) in enumerate(quad_edges):
            a, b_pt = quad[i], quad[j]
            ab = b_pt - a
            ab_len_sq = np.dot(ab, ab)

            t = np.dot(pt - a, ab) / ab_len_sq
            t = np.clip(t, 0.0, 1.0)
            closest = a + t * ab
            dist = np.linalg.norm(pt - closest)

            if dist < best_dist:
                best_dist = dist
                best_t = t
                best_edge = edge_i

        r_start, r_end = rect_edges[best_edge]
        dst = r_start + best_t * (r_end - r_start)
        dst_pts.append(dst)

    return np.array(dst_pts, dtype=np.float64)


def sort_points(points):
    points = _validate_point_set(points, min_points=4)

    center = np.mean(points, axis=0)

    def get_angle(p):
        return math.atan2(p[1] - center[1], p[0] - center[0])

    sorted_points = np.array(sorted(points.tolist(), key=get_angle))

    return sorted_points


def get_transformation_matrix(points):
    points = _validate_point_set(points, min_points=2, min_rank=1)

    x_mean = np.mean(points[:, 0])
    y_mean = np.mean(points[:, 1])

    davg = np.mean(np.sqrt((points[:, 0] - x_mean) ** 2 + (points[:, 1] - y_mean) ** 2))

    if davg <= 1e-12:
        raise ValueError("Point normalization failed due to near-zero spread")

    s = np.sqrt(2) / davg

    return np.array([[s, 0, -s * x_mean], [0, s, -s * y_mean], [0, 0, 1]])


def transform_coords(points):
    points = _validate_point_set(points, min_points=2, min_rank=1)

    h = get_transformation_matrix(points)

    h_points = np.column_stack((points, np.full((len(points), 1), 1)))

    return h_points @ h.T, h


def build_dlt_system(src_points_h, dst_points_h):
    src_points_h = np.asarray(src_points_h, dtype=np.float64)
    dst_points_h = np.asarray(dst_points_h, dtype=np.float64)

    if src_points_h.shape != dst_points_h.shape:
        raise ValueError(
            f"Source and destination points must have equal shape, got {src_points_h.shape} and {dst_points_h.shape}"
        )

    if src_points_h.ndim != 2 or src_points_h.shape[1] != 3:
        raise ValueError(
            f"Homogeneous points must have shape (N, 3), got {src_points_h.shape}"
        )

    A = []
    for i in range(len(src_points_h)):
        x, y, _ = src_points_h[i]
        u, v, _ = dst_points_h[i]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    return np.asarray(A, dtype=np.float64)


def svd_confidence_metric(singular_values):
    singular_values = np.asarray(singular_values, dtype=np.float64)
    if singular_values.size < 2:
        return np.inf

    sigma_9 = singular_values[-1]
    sigma_8 = singular_values[-2]

    if abs(sigma_9) <= 1e-15:
        return np.inf

    return float(sigma_8 / sigma_9)


def solve_normalized_dlt(src_points, dst_points):
    src_points = _validate_point_set(src_points, min_points=4)
    dst_points = _validate_point_set(dst_points, min_points=4)

    if src_points.shape != dst_points.shape:
        raise ValueError(
            f"Source and destination points must have equal shape, got {src_points.shape} and {dst_points.shape}"
        )

    h_in_pts, t_in = transform_coords(src_points)
    h_out_pts, t_out = transform_coords(dst_points)

    A = build_dlt_system(h_in_pts, h_out_pts)

    _, singular_values, vh = np.linalg.svd(A)
    h_vector = vh[-1, :]
    h_norm = h_vector.reshape(3, 3)

    h_final = np.linalg.inv(t_out) @ h_norm @ t_in

    if abs(h_final[2, 2]) <= 1e-15:
        raise ValueError("Homography normalization failed due to zero scale term")

    h_final /= h_final[2, 2]

    diagnostics = {
        "A": A,
        "h_vector": h_vector,
        "h_normalized": h_norm,
        "singular_values": singular_values,
        "rho": svd_confidence_metric(singular_values),
        "t_in": t_in,
        "t_out": t_out,
    }

    return h_final, diagnostics


def compute_homography_matrix(
    input_points,
    width,
    height,
    *,
    max_svd_confidence_ratio=None,
    return_diagnostics=False,
):
    input_points = _validate_point_set(input_points, min_points=4)
    output_points = generate_dst_points(input_points, width, height)

    h_final, diagnostics = solve_normalized_dlt(input_points, output_points)
    diagnostics["output_points"] = output_points

    if (
        max_svd_confidence_ratio is not None
        and np.isfinite(diagnostics["rho"])
        and diagnostics["rho"] > max_svd_confidence_ratio
    ):
        raise ValueError(
            "Homography is ill-conditioned: SVD confidence ratio exceeds threshold"
        )

    if return_diagnostics:
        return h_final, diagnostics

    return h_final


def homography(
    img,
    input_points,
    width,
    height,
    *,
    max_svd_confidence_ratio=None,
    return_diagnostics=False,
):
    h_result = compute_homography_matrix(
        input_points,
        width,
        height,
        max_svd_confidence_ratio=max_svd_confidence_ratio,
        return_diagnostics=return_diagnostics,
    )

    if return_diagnostics:
        h_final, diagnostics = h_result
    else:
        h_final = h_result
        diagnostics = None

    warped = apply_transformation_matrix(
        img, h_final, interpolation="bilinear", dsize=(width, height)
    )

    if return_diagnostics:
        diagnostics = dict(diagnostics)
        diagnostics["homography_matrix"] = h_final
        return warped, diagnostics

    return warped
