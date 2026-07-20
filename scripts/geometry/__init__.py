"""Geometry utilities for explicit detector-frame and volume-frame transforms."""

from .quarter_turn import (
    QuarterTurnFactorization,
    apply_quarter_turn_numpy,
    choose_quarter_turn,
    factor_affines,
    quarter_turn_matrix,
    transform_shape_xy,
)
from .volume_frames import (
    BASE_AXIS_PERMUTATION,
    DETECTOR_FRAME,
    IMOD_MRC_FRAME,
    VOLUME_FRAME_CONTRACT_VERSION,
    WARP_FRAME,
    apply_projection_quarter_turn_to_detector_shape_xy,
    apply_projection_quarter_turn_to_warp_shape,
    imod_mrc_shape_to_current_warp_xyz,
    imod_mrc_shape_to_warp_xyz,
    volume_frame_manifest,
)

__all__ = [
    "QuarterTurnFactorization",
    "apply_quarter_turn_numpy",
    "choose_quarter_turn",
    "factor_affines",
    "quarter_turn_matrix",
    "transform_shape_xy",
    "BASE_AXIS_PERMUTATION",
    "DETECTOR_FRAME",
    "IMOD_MRC_FRAME",
    "VOLUME_FRAME_CONTRACT_VERSION",
    "WARP_FRAME",
    "apply_projection_quarter_turn_to_detector_shape_xy",
    "apply_projection_quarter_turn_to_warp_shape",
    "imod_mrc_shape_to_current_warp_xyz",
    "imod_mrc_shape_to_warp_xyz",
    "volume_frame_manifest",
]
