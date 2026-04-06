import math
import numpy as np
from .transformations import apply_transformation_matrix

def generate_dst_points(src_pts, width, height):
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

    center = np.mean(points, axis=0)

    def get_angle(p):
        return math.atan2(p[1] - center[1], p[0] - center[0])

    sorted_points = np.array(sorted(points.tolist(), key=get_angle))

    return sorted_points


def get_transformation_matrix(points):
    x_mean = np.mean(points[:, 0])
    y_mean = np.mean(points[:, 1])

    davg = np.mean(np.sqrt((points[:, 0] - x_mean) ** 2 + (points[:, 1] - y_mean) ** 2))

    s = np.sqrt(2) / davg

    return np.array([[s, 0, -s * x_mean], [0, s, -s * y_mean], [0, 0, 1]])


def transform_coords(points):
    h = get_transformation_matrix(points)

    h_points = np.column_stack((points, np.full((len(points), 1), 1)))

    return h_points @ h.T, h


def homography(img, input_points, width, height):
    output_points = generate_dst_points(input_points, width, height)

    h_in_pts, t_in = transform_coords(input_points)
    h_out_pts, t_out = transform_coords(output_points)

    A = []
    for i in range(len(h_in_pts)):
        x, y, _ = h_in_pts[i]
        u, v, _ = h_out_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    _, _, vh = np.linalg.svd(A)
    h_norm = vh[-1, :].reshape(3, 3)

    h_final = np.linalg.inv(t_out) @ h_norm @ t_in

    h_final /= h_final[2, 2]

    return apply_transformation_matrix(
        img, h_final, interpolation="bilinear", dsize=(width, height)
    )
