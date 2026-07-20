from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from geometry.volume_frames import (  # noqa: E402
    apply_projection_quarter_turn_to_detector_shape_xy,
    apply_projection_quarter_turn_to_warp_shape,
    imod_mrc_shape_to_current_warp_xyz,
    imod_mrc_shape_to_warp_xyz,
    volume_frame_manifest,
)


def test_imod_reconstruction_storage_maps_thickness_y_to_warp_z():
    # Real testABC geometry: IMOD MRC X=2046, Y(thickness)=494,
    # Z(detector vertical)=2880.
    assert imod_mrc_shape_to_warp_xyz((2046, 494, 2880)) == (2046, 2880, 494)


def test_odd_quarter_turn_swaps_detector_xy_only():
    assert apply_projection_quarter_turn_to_detector_shape_xy((2046, 2880), 1) == (
        2880,
        2046,
    )
    assert apply_projection_quarter_turn_to_detector_shape_xy((2046, 2880), 3) == (
        2880,
        2046,
    )
    assert apply_projection_quarter_turn_to_detector_shape_xy((2046, 2880), 2) == (
        2046,
        2880,
    )


def test_detector_quarter_turn_does_not_swap_warp_volume_xy():
    base = (2046, 2880, 494)
    assert apply_projection_quarter_turn_to_warp_shape(base, 1) == base
    assert apply_projection_quarter_turn_to_warp_shape(base, 3) == base
    assert imod_mrc_shape_to_current_warp_xyz(
        (2046, 494, 2880), quarter_turn_k=1
    ) == base


def test_combined_mapping_records_detector_only_scope():
    manifest = volume_frame_manifest((2046, 494, 2880), quarter_turn_k=1)
    assert manifest["contract_version"] == 2
    assert manifest["base_axis_permutation_imod_mrc_to_warp"] == [0, 2, 1]
    assert manifest["base_shape_warp_xyz"] == [2046, 2880, 494]
    assert manifest["reconstruction_shape_warp_xyz"] == [2046, 2880, 494]
    assert manifest["current_shape_warp_xyz"] == [2046, 2880, 494]
    assert manifest["projection_quarter_turn_scope"] == "detector_frame_only"
    assert manifest["volume_shape_invariant_under_detector_quarter_turn"] is True
