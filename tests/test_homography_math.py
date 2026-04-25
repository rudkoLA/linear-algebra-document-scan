import numpy as np

from src.homograph import compute_homography_matrix, solve_normalized_dlt


def test_h_times_h_inverse_is_identity():
    input_points = np.array(
        [
            [120.0, 70.0],
            [530.0, 90.0],
            [575.0, 360.0],
            [80.0, 330.0],
            [325.0, 65.0],
            [555.0, 220.0],
            [325.0, 365.0],
            [95.0, 210.0],
        ],
        dtype=np.float64,
    )

    H, diagnostics = compute_homography_matrix(
        input_points,
        width=600,
        height=400,
        return_diagnostics=True,
    )

    round_trip = H @ np.linalg.inv(H)
    assert np.allclose(round_trip, np.eye(3), atol=1e-8)
    assert np.isfinite(H).all()
    assert diagnostics["A"].shape == (2 * len(input_points), 9)


def test_dlt_null_space_property_Ah_is_near_zero():
    input_points = np.array(
        [
            [130.0, 80.0],
            [550.0, 120.0],
            [590.0, 360.0],
            [70.0, 350.0],
            [340.0, 76.0],
            [570.0, 230.0],
            [320.0, 360.0],
            [84.0, 216.0],
        ],
        dtype=np.float64,
    )

    true_h = np.array(
        [
            [1.07, 0.06, 32.0],
            [0.03, 1.12, 18.0],
            [7.5e-4, 4.0e-4, 1.0],
        ],
        dtype=np.float64,
    )

    src_h = np.column_stack((input_points, np.ones((len(input_points), 1), dtype=np.float64)))
    dst_h = src_h @ true_h.T
    dst_points = dst_h[:, :2] / dst_h[:, 2:3]

    estimated_h, diagnostics = solve_normalized_dlt(input_points, dst_points)

    A = diagnostics["A"]
    h = diagnostics["h_vector"]
    residual_norm = np.linalg.norm(A @ h)

    assert residual_norm < 1e-6
    estimated_h /= estimated_h[2, 2]
    true_h /= true_h[2, 2]
    assert np.allclose(estimated_h, true_h, atol=1e-6)
    assert diagnostics["rho"] > 0
