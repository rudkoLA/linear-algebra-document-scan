import numpy as np


def interpolate_nearest(img, src_x, src_y):
    old_height, old_width = img.shape[:2]

    map_x = np.clip(np.round(src_x), 0, old_width - 1).astype(int)
    map_y = np.clip(np.round(src_y), 0, old_height - 1).astype(int)

    return img[map_y, map_x]


def interpolate_bilinear(img, src_x, src_y):
    old_height, old_width = img.shape[:2]

    x0 = np.floor(src_x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(src_y).astype(int)
    y1 = y0 + 1

    dx = src_x - x0
    dy = src_y - y0

    x0 = np.clip(x0, 0, old_width - 1)
    x1 = np.clip(x1, 0, old_width - 1)
    y0 = np.clip(y0, 0, old_height - 1)
    y1 = np.clip(y1, 0, old_height - 1)

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    wa = (1 - dx) * (1 - dy)
    wb = dx * (1 - dy)
    wc = (1 - dx) * dy
    wd = dx * dy

    return (
        wa[..., None] * Ia
        + wb[..., None] * Ib
        + wc[..., None] * Ic
        + wd[..., None] * Id
    ).astype(img.dtype)
