import numpy as np
from .utils import interpolate_nearest, interpolate_bilinear


def apply_transformation_matrix(
    img, transformation_matrix, dsize=None, interpolation="nearest"
):
    old_height, old_width = img.shape[:2]

    if dsize is None:
        corners = np.array(
            [
                [0, 0, 1],
                [old_width, 0, 1],
                [old_width, old_height, 1],
                [0, old_height, 1],
            ]
        )
        transformed_corners = corners @ transformation_matrix.T
        transformed_corners /= transformed_corners[:, 2:3]

        min_x, min_y = np.min(transformed_corners[:, :2], axis=0)
        max_x, max_y = np.max(transformed_corners[:, :2], axis=0)
        width, height = int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y))
    else:
        width, height = dsize

    H_inv = np.linalg.inv(transformation_matrix)
    dest_y, dest_x = np.mgrid[0:height, 0:width]
    dest_coords = np.stack([dest_x, dest_y, np.ones_like(dest_x)], axis=-1)

    source_coords = dest_coords @ H_inv.T

    w = source_coords[..., 2]
    w = np.where(w == 0, 1e-10, w)
    src_x = source_coords[..., 0] / w
    src_y = source_coords[..., 1] / w

    if interpolation == "nearest":
        return interpolate_nearest(img, src_x, src_y)
    elif interpolation == "bilinear":
        return interpolate_bilinear(img, src_x, src_y)
    else:
        raise ValueError("Interpolation must be 'nearest' or 'bilinear'")
