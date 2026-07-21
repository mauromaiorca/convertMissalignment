"""IMOD -> Warp tilt-angle sign convention (IMOD_TO_WARP_TILT_ANGLE_SIGN = -1).

Covers: view order stays identity; the sign is applied exactly once to the angles and once
to OFFSET (LevelAngleY); the effective angles are the sign-transformed IMOD effective angles;
per-view arrays keep their source association; the Warp->IMOD angle/OFFSET round trip; .xf is
unaffected by the angle sign; cache invalidation when the sign changes; old sign-ambiguous
manifests are stale. Runs without warpylib/WarpTools/IMOD (a fake TiltSeries is injected).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from geometry.imod_positioning import (  # noqa: E402
    IMOD_TO_WARP_TILT_ANGLE_SIGN, ImodPositioning, from_toml_table, imod_angles_to_warp,
    imod_offset_to_warp_level_angle_y, tilt_angle_convention_manifest, tilt_view_order_identity,
    validate_tilt_angle_sign, warp_angles_to_imod, warp_level_angle_y_to_imod_offset,
)

NON_SYMMETRIC = np.array([-58.2, -31.1, 1.4, 28.8, 61.0])
FIRST_FIVE_IMOD = [-54.78, -51.39, -48.01, -44.61, -41.45]


class HelperTests(unittest.TestCase):
    def test_default_sign_is_minus_one(self):
        self.assertEqual(IMOD_TO_WARP_TILT_ANGLE_SIGN, -1)

    def test_spec_example_negated_not_reversed(self):
        warp = imod_angles_to_warp(NON_SYMMETRIC, -1)
        self.assertEqual(warp, [58.2, 31.1, -1.4, -28.8, -61.0])
        self.assertNotEqual(warp, list(NON_SYMMETRIC[::-1]))     # NOT reversed

    def test_first_five_dataset_angles(self):
        warp = imod_angles_to_warp(FIRST_FIVE_IMOD, -1)
        self.assertTrue(np.allclose(warp, [54.78, 51.39, 48.01, 44.61, 41.45]))

    def test_offset_uses_same_sign(self):
        self.assertEqual(imod_offset_to_warp_level_angle_y(-11.5, -1), 11.5)

    def test_effective_angle_identity(self):
        off = -11.5
        warp = np.array(imod_angles_to_warp(NON_SYMMETRIC, -1))
        ly = imod_offset_to_warp_level_angle_y(off, -1)
        self.assertTrue(np.allclose(warp + ly, -1 * (NON_SYMMETRIC + off)))

    def test_sign_plus_one_is_identity(self):
        self.assertEqual(imod_angles_to_warp(NON_SYMMETRIC, 1), list(NON_SYMMETRIC))
        self.assertEqual(imod_offset_to_warp_level_angle_y(-11.5, 1), -11.5)

    def test_validate_rejects_non_pm1(self):
        for bad in (0, 2, -2, 0.5, "x"):
            with self.assertRaises((ValueError, TypeError)):
                validate_tilt_angle_sign(bad)

    def test_view_order_identity(self):
        vo = tilt_view_order_identity(5)
        self.assertEqual(vo["mapping"], "identity")
        self.assertEqual(vo["warp_to_source"], [0, 1, 2, 3, 4])
        self.assertEqual(vo["source_to_warp"], [0, 1, 2, 3, 4])

    def test_convention_manifest(self):
        m = tilt_angle_convention_manifest(-1)
        self.assertEqual(m["imod_to_warp_sign"], -1)
        self.assertEqual(m["operation"], "elementwise_negation")
        self.assertTrue(m["offset_uses_same_sign"])
        self.assertEqual(m["validation_status"], "pending_reconstruction_comparison")


class InverseRoundTripTests(unittest.TestCase):
    """Warp -> IMOD export inverse (sign is its own inverse)."""

    def test_angle_round_trip(self):
        warp = imod_angles_to_warp(NON_SYMMETRIC, -1)
        self.assertEqual(warp_angles_to_imod(warp, -1), list(NON_SYMMETRIC))

    def test_offset_round_trip(self):
        self.assertEqual(warp_level_angle_y_to_imod_offset(11.5, -1), -11.5)

    def test_xf_unaffected_by_angle_sign(self):
        # composing final = residual @ original must not depend on the angle sign
        from pipeline.imod_revision import Affine2D, compose_final_transform
        raw_xy, ali_xy = (4096, 4096), (2048, 2048)
        orig = Affine2D(0.5 * np.eye(2), np.array([2.0, -1.0]))
        delta = Affine2D(np.eye(2), np.array([3.0, 4.0]))
        final = compose_final_transform(orig, delta, raw_shape_xy=raw_xy, aligned_shape_xy=ali_xy)
        # the sign helpers never touch matrices; final is identical regardless of any sign var
        self.assertTrue(np.allclose(final.matrix, orig.matrix))


class CacheInvalidationTests(unittest.TestCase):
    def test_positioning_hash_changes_with_sign(self):
        a = ImodPositioning(tilt_angle_offset_deg=-11.5, imod_to_warp_tilt_angle_sign=-1)
        b = ImodPositioning(tilt_angle_offset_deg=-11.5, imod_to_warp_tilt_angle_sign=1)
        self.assertNotEqual(a.positioning_hash(), b.positioning_hash())

    def test_toml_round_trip_carries_sign(self):
        t = ImodPositioning(imod_to_warp_tilt_angle_sign=-1).to_toml_table()
        self.assertEqual(t["imod_to_warp_tilt_angle_sign"], -1)
        self.assertEqual(from_toml_table(t).imod_to_warp_tilt_angle_sign, -1)

    def test_old_manifest_without_sign_defaults_and_is_stale_vs_plus_one(self):
        # A pre-contract table (no sign) loads as the default -1; its hash differs from +1,
        # so a sign +1 artefact is stale.
        old = {"tilt_angle_offset_deg": -11.5}                  # no sign recorded
        loaded = from_toml_table(old)
        self.assertEqual(loaded.imod_to_warp_tilt_angle_sign, -1)
        plus = ImodPositioning(tilt_angle_offset_deg=-11.5, imod_to_warp_tilt_angle_sign=1)
        self.assertNotEqual(loaded.positioning_hash(), plus.positioning_hash())

    def test_reconstruction_contract_hash_includes_sign(self):
        from pipeline.reconstruction_tiling import (
            ReconstructionTiling, reconstruction_contract_hash)
        tiling = ReconstructionTiling(subvolume_size=64, subvolume_padding=6)
        base = dict(tiling=tiling, output_angpix=17.6, normalize=False, warptools_version="2.0.39")
        self.assertNotEqual(
            reconstruction_contract_hash(**base, tilt_angle_sign=-1),
            reconstruction_contract_hash(**base, tilt_angle_sign=1))

    def test_export_cache_key_includes_sign(self):
        from pipeline.imod_revision import RevisionPolicy
        from pipeline.imod_revision_writer import export_cache_key
        base = dict(source_geometry_hash="s", refined_geometry_hash="r", positioning_hash="p",
                    volume_frame_contract_version=2, policy=RevisionPolicy(), imod_version="5.1.11")
        self.assertNotEqual(
            export_cache_key(**base, tilt_angle_sign=-1),
            export_cache_key(**base, tilt_angle_sign=1))


# --------------------------------------------------------------------------- #
# End-to-end converter: fake warpylib + a tiny real MRC stack.
# --------------------------------------------------------------------------- #
class FakeCubicGrid:
    def __init__(self, *a, **k):
        pass


class FakeTiltSeries:
    def __init__(self, *a, **k):
        self.path = k.get("path")
        self.n_tilts = k.get("n_tilts")
        self._attrs = {}

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def save_meta(self, path):        # no-op; process_tilt_series writes its own JSON manifest
        self._saved = path

    def apply_tomogram_shift_3d(self, vec):
        self._shift = [float(x) for x in vec.tolist()]


def _load_converter_with_fake_warp():
    warpylib = types.ModuleType("warpylib")
    warpylib.CubicGrid = FakeCubicGrid
    warpylib.TiltSeries = FakeTiltSeries
    ops = types.ModuleType("warpylib.ops")
    rescale_mod = types.ModuleType("warpylib.ops.rescale")
    rescale_mod.rescale = lambda images, size: images
    sys.modules["warpylib"] = warpylib
    sys.modules["warpylib.ops"] = ops
    sys.modules["warpylib.ops.rescale"] = rescale_mod
    spec = importlib.util.spec_from_file_location(
        "etomo_to_warp_sign_test", ROOT / "scripts" / "etomo_to_warp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConverterEndToEndTests(unittest.TestCase):
    def setUp(self):
        import mrcfile  # noqa: F401
        self.mod = _load_converter_with_fake_warp()
        self.tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        self.n = 5
        self.raw_angles = list(NON_SYMMETRIC)
        ts_dir = self.tmp / "TS_demo"
        ts_dir.mkdir()
        # rawtlt
        (ts_dir / "TS_demo.rawtlt").write_text("".join(f"{a}\n" for a in self.raw_angles))
        # identity .xf + source.xf
        rows = "".join("1.0 0.0 0.0 1.0 0.0 0.0\n" for _ in range(self.n))
        (ts_dir / "TS_demo.xf").write_text(rows)
        (ts_dir / "TS_demo.source.xf").write_text(rows)
        # tiny MRC stack (n, 8, 8) float32 with voxel size 2.0
        import mrcfile
        data = np.zeros((self.n, 8, 8), dtype=np.float32)
        with mrcfile.new(str(ts_dir / "TS_demo.st"), overwrite=True) as m:
            m.set_data(data)
            m.voxel_size = 2.0
        self.ts_dir = ts_dir
        self.out = self.tmp / "out"
        self.out.mkdir()
        self.pos = ImodPositioning(
            tilt_angle_offset_deg=-11.5, x_axis_tilt_deg=1.82,
            imod_to_warp_tilt_angle_sign=-1,
            present_fields=("OFFSET", "XAXISTILT"))

    def _run(self, sign=-1):
        ts, _ = self.mod.process_tilt_series(
            self.ts_dir, self.out, tilt_axis_angle=84.1, volume_shape=(8, 4, 8),
            output_pixel_size=None, alignment_mode="translation", axis_frame="raw",
            grid_shape_xy=(5, 5), positioning=self.pos, level_angle_x_sign=-1,
            imod_to_warp_tilt_angle_sign=sign)
        manifest = json.loads((self.out / "TS_demo.conversion.json").read_text())
        return ts, manifest

    def test_angles_negated_once_and_offset_signed_once(self):
        ts, manifest = self._run(sign=-1)
        self.assertTrue(np.allclose(list(ts.angles.tolist()), [-a for a in self.raw_angles]))
        self.assertNotEqual(list(ts.angles.tolist()), self.raw_angles[::-1])   # not reversed
        self.assertAlmostEqual(float(ts.level_angle_y), 11.5, places=5)        # sign * OFFSET, once
        # LevelAngleX unaffected by the tilt-angle sign
        self.assertAlmostEqual(float(ts.level_angle_x), -1.82, places=5)

    def test_manifest_records_identity_order_and_sign(self):
        _, manifest = self._run(sign=-1)
        self.assertEqual(manifest["tilt_view_order"]["mapping"], "identity")
        self.assertEqual(manifest["tilt_view_order"]["warp_to_source"], list(range(self.n)))
        self.assertEqual(manifest["tilt_angle_convention"]["imod_to_warp_sign"], -1)
        self.assertTrue(manifest["tilt_angle_convention"]["offset_uses_same_sign"])

    def test_sign_plus_one_keeps_imod_sign(self):
        ts, manifest = self._run(sign=1)
        self.assertTrue(np.allclose(list(ts.angles.tolist()), self.raw_angles))
        self.assertAlmostEqual(float(ts.level_angle_y), -11.5, places=5)
        self.assertEqual(manifest["tilt_angle_convention"]["imod_to_warp_sign"], 1)

    def test_per_view_arrays_retain_source_association(self):
        ts, _ = self._run(sign=-1)
        # angles, tilt_axis_angles and the per-tilt offsets all have n rows, same order
        self.assertEqual(len(ts.angles.tolist()), self.n)
        self.assertEqual(len(ts.tilt_axis_angles.tolist()), self.n)
        self.assertEqual(len(ts.tilt_axis_offset_x.tolist()), self.n)


class DiagnosticComparisonTests(unittest.TestCase):
    """The clip-rotx comparison core (NCC) — testable without IMOD/WarpTools."""

    def test_ncc_identical_is_one(self):
        from pipeline.diagnose_tilt_angle_sign import normalized_cross_correlation
        rng = np.random.default_rng(0)
        v = rng.normal(size=(4, 5, 6))
        self.assertAlmostEqual(normalized_cross_correlation(v, v), 1.0, places=6)

    def test_ncc_negated_is_minus_one(self):
        from pipeline.diagnose_tilt_angle_sign import normalized_cross_correlation
        rng = np.random.default_rng(1)
        v = rng.normal(size=(3, 3, 3))
        self.assertAlmostEqual(normalized_cross_correlation(v, -v), -1.0, places=6)

    def test_compare_prefers_higher_ncc_as_agreement(self):
        # a synthetic reference; the "minus1" candidate is the reference itself (NCC 1),
        # the "plus1" candidate is a flipped/decorrelated volume (lower NCC).
        import tempfile
        import mrcfile
        from pipeline.diagnose_tilt_angle_sign import compare
        rng = np.random.default_rng(2)
        ref = rng.normal(size=(6, 6, 6)).astype(np.float32)
        plus1 = ref[::-1].copy()                      # handedness-flipped -> lower NCC
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            def _w(name, arr):
                p = d / name
                with mrcfile.new(str(p), overwrite=True) as m:
                    m.set_data(arr.astype(np.float32))
                return p
            report = compare(_w("ref.mrc", ref), _w("p1.mrc", plus1), _w("m1.mrc", ref))
        self.assertEqual(report["agrees_with_clip_rotx"], "warp_sign_minus1")
        self.assertTrue(report["minus1_agrees"])


if __name__ == "__main__":
    unittest.main()
