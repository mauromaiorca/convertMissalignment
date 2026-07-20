from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from geometry.quarter_turn import (  # noqa: E402
    apply_quarter_turn_numpy,
    factor_affines,
    quarter_turn_matrix,
    transform_shape_xy,
)
from imod_affine import forward_points_pixels  # noqa: E402


def rotation(deg: float) -> np.ndarray:
    angle = np.deg2rad(deg)
    return np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=float,
    )


def test_selects_minus_90_degree_quarter_turn_for_minus_84_degree_affine():
    matrices = np.stack([rotation(-84.65), rotation(-84.55), rotation(-84.35)])
    shifts = np.array([[10.0, -3.0], [2.0, 1.0], [-5.0, 4.0]])
    factored = factor_affines(matrices, shifts)
    assert factored.k == 1
    assert np.allclose(factored.quarter_turn_matrix, quarter_turn_matrix(1))
    assert factored.residual_rotation_max_abs_deg < 6.0
    assert factored.max_recomposition_error < 1e-12
    assert np.array_equal(factored.residual_shifts, shifts)


def test_factorization_preserves_centered_coordinate_mapping_with_swapped_shape():
    raw_shape = (8, 6)
    rotated_shape = transform_shape_xy(raw_shape, 1)
    matrix = rotation(-84.5) @ np.array([[1.003, 0.002], [0.0, 1.001]])
    shift = np.array([12.25, -7.5])
    factored = factor_affines(matrix, shift, k=1)
    q = factored.quarter_turn_matrix

    raw_points = np.array(
        [
            [0.0, 0.0],
            [raw_shape[0] - 1.0, 0.0],
            [0.0, raw_shape[1] - 1.0],
            [raw_shape[0] - 1.0, raw_shape[1] - 1.0],
            [(raw_shape[0] - 1.0) / 2.0, (raw_shape[1] - 1.0) / 2.0],
        ]
    )
    direct = forward_points_pixels(raw_points, matrix, shift, raw_shape, raw_shape)
    quarter_points = forward_points_pixels(
        raw_points,
        q,
        np.zeros(2),
        raw_shape,
        rotated_shape,
    )
    residual = forward_points_pixels(
        quarter_points,
        factored.residual_matrices[0],
        factored.residual_shifts[0],
        rotated_shape,
        raw_shape,
    )
    assert np.allclose(direct, residual, atol=1e-10)


def test_numpy_quarter_turn_is_exact_pixel_permutation():
    stack = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    rotated = apply_quarter_turn_numpy(stack, 1)
    assert rotated.shape == (2, 5, 3)
    assert np.array_equal(np.sort(rotated.reshape(-1)), np.sort(stack.reshape(-1)))
    restored = apply_quarter_turn_numpy(rotated, 3)
    assert np.array_equal(restored, stack)
