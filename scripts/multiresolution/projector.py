#!/usr/bin/env python3
"""Independent analytic parallel-beam projector for geometric verification.

Cryo-ET tilt series are parallel (orthographic) projections. With the tilt axis
along Y and rotation by ``theta`` about Y, a physical voxel ``[X, Y, Z]``
projects to physical detector ``[u, v]`` with::

    u = X*cos(theta) + Z*sin(theta)
    v = Y

An optional in-plane tilt-axis angle ``phi`` rotates ``(u, v)`` in the detector
plane. This module builds the source projection matrix ``P_source`` (3x4,
source voxel -> source detector pixel, homogeneous) from explicit grids; the
multiresolution transfer ``P_working = inv(G_d) @ P_source @ G_v`` is then an
algebraic identity that ``test_multires_projection`` checks against an
INDEPENDENT physical projection of the same physical point (not the matrix
algebra), so the test is not circular.

This is a labelled limitation: no IMOD forward projector is used for the
*matrix* derivation (real ``tilt``/``xyzproj`` are used separately for
reconstruction geometry). The analytic projector is exact for parallel beams.
"""
from __future__ import annotations

import numpy as np

from .grid2d import Grid2D
from .grid3d import Grid3D


def physical_projection_matrix(theta_deg: float, tilt_axis_angle_deg: float = 0.0) -> np.ndarray:
    """3x4 physical-voxel-homogeneous -> physical-detector-homogeneous."""
    t = np.deg2rad(theta_deg)
    P = np.array([
        [np.cos(t), 0.0, np.sin(t), 0.0],
        [0.0,       1.0, 0.0,       0.0],
        [0.0,       0.0, 0.0,       1.0],
    ], dtype=float)
    phi = np.deg2rad(tilt_axis_angle_deg)
    if phi != 0.0:
        R = np.array([[np.cos(phi), -np.sin(phi), 0.0],
                      [np.sin(phi),  np.cos(phi), 0.0],
                      [0.0,          0.0,         1.0]])
        P = R @ P
    return P


def source_projection_matrix(volume: Grid3D, detector: Grid2D, theta_deg: float,
                             tilt_axis_angle_deg: float = 0.0) -> np.ndarray:
    """``P_source = D @ P_phys @ Q_volume`` (3x4): source voxel -> source detector pixel.

    ``D = inv(Q_detector)`` maps physical detector -> detector pixels.
    """
    P_phys = physical_projection_matrix(theta_deg, tilt_axis_angle_deg)
    D = detector.Q_inv  # 3x3 physical -> pixel
    return D @ P_phys @ volume.Q  # 3x3 @ 3x4 @ 4x4 = 3x4


def project_physical_point(physical_xyz, detector: Grid2D, theta_deg: float,
                           tilt_axis_angle_deg: float = 0.0) -> np.ndarray:
    """Independent route: project a PHYSICAL point and return its DETECTOR pixel.

    Used as the non-circular ground truth for projection invariance.
    """
    X = np.asarray(physical_xyz, float).reshape(3)
    P_phys = physical_projection_matrix(theta_deg, tilt_axis_angle_deg)
    d_phys_h = P_phys @ np.array([X[0], X[1], X[2], 1.0])
    d_phys = d_phys_h[:2] / d_phys_h[2]
    return detector.physical_to_pixel(d_phys)
