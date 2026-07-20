from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import mrcfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class FakeCubicGrid:
    def __init__(self, shape, values=None):
        self.shape = tuple(shape)
        self.values = values


class FakeTiltSeries:
    def __init__(self, path: str, n_tilts: int):
        self.path = path
        self.n_tilts = n_tilts

    def save_meta(self, path: str):
        Path(path).write_text('<TiltSeries VolumeDimensionsAngstrom="1,1,1"/>\n')


def _load_converter():
    warpylib = types.ModuleType("warpylib")
    warpylib.CubicGrid = FakeCubicGrid
    warpylib.TiltSeries = FakeTiltSeries
    ops = types.ModuleType("warpylib.ops")
    rescale_mod = types.ModuleType("warpylib.ops.rescale")

    def fake_rescale(images, size):
        return torch.nn.functional.interpolate(
            images[:, None], size=size, mode="bilinear", align_corners=False
        )[:, 0]

    rescale_mod.rescale = fake_rescale
    sys.modules["warpylib"] = warpylib
    sys.modules["warpylib.ops"] = ops
    sys.modules["warpylib.ops.rescale"] = rescale_mod
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "etomo_to_warp_v7_test", ROOT / "scripts" / "etomo_to_warp.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rotation(deg: float) -> np.ndarray:
    angle = np.deg2rad(deg)
    return np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )


def test_quarter_turn_mode_materializes_stack_and_writes_residual_xf(tmp_path):
    converter = _load_converter()
    series = "TS_demo_raw_xf_affine_fixed"
    staged = tmp_path / series
    staged.mkdir()
    data = np.arange(2 * 6 * 8, dtype=np.float32).reshape(2, 6, 8)
    with mrcfile.new(staged / f"{series}.st", overwrite=True) as handle:
        handle.set_data(data)
        handle.voxel_size = 2.0
    (staged / f"{series}.rawtlt").write_text("-10\n10\n")
    matrices = np.stack([_rotation(-84.6), _rotation(-84.4)])
    shifts = np.array([[3.0, -2.0], [-4.0, 1.0]])
    converter.write_xf(staged / f"{series}.xf", matrices, shifts)
    converter.write_xf(staged / f"{series}.source.xf", matrices, shifts)

    out = tmp_path / "warp"
    ts, _ = converter.process_tilt_series(
        staged,
        out,
        tilt_axis_angle=84.0,
        volume_shape=(6, 4, 8),
        output_pixel_size=None,
        alignment_mode="quarter-turn-affine",
        axis_frame="aligned",
        grid_shape_xy=(3, 3),
    )

    # IMOD MRC (6,4,8) maps to Warp volume (6,8,4). The k=1
    # quarter turn rotates only the 2-D detector frame, so the requested
    # reconstruction-volume extents remain (6,8,4).
    assert tuple(float(value) for value in ts.volume_dimensions_physical) == (
        12.0,
        16.0,
        8.0,
    )

    output_stack = out / "tiltstack" / series / f"{series}.st"
    with mrcfile.open(output_stack, permissive=True) as handle:
        result = np.asarray(handle.data)
        assert result.shape == (2, 8, 6)
        assert float(handle.voxel_size.x) == 2.0
    assert np.array_equal(result, np.rot90(data, k=1, axes=(-2, -1)))

    manifest = json.loads((out / f"{series}.conversion.json").read_text())
    assert manifest["alignment_mode"] == "quarter-turn-affine"
    assert manifest["quarter_turn"]["np_rot90_k"] == 1
    assert manifest["quarter_turn"]["residual_rotation_max_abs_deg"] < 6.0
    assert manifest["quarter_turn"]["max_recomposition_error"] < 1e-12
    # Input volume shape is IMOD MRC storage order: X, Y(thickness), Z.
    # Base Warp order is X,Z,Y. The detector rot90 does not swap volume X/Y.
    assert manifest["volume_frame"]["contract_version"] == 2
    assert manifest["volume_frame"]["base_shape_warp_xyz"] == [6, 8, 4]
    assert manifest["volume_frame"]["projection_quarter_turn_scope"] == "detector_frame_only"
    assert manifest["warp_volume_shape_xyz"] == [6, 8, 4]
    assert manifest["movement_grid_range_A"]["x"][1] < 20.0
    assert Path(manifest["residual_xf_file"]).is_file()
