"""IMOD positioning propagation: parse -> Geometry -> resolved TOML -> staging manifest
-> cluster conversion rehydration -> cache/marker contract, plus the legacy path.

Every check here is pure (config / dataclass / TOML / command construction / synthetic
tilt.com). Nothing imports warpylib, torch, WarpTools or IMOD, so the propagation is
validated off-cluster. The Warp *application* of the rehydrated object is covered by
test_converter_positioning_fake_warp.py; its geometric sign is a cluster concern.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

try:                        # py311+: stdlib; else the pip 'tomli' backport
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
# Only scripts/ on the path: 'geometry' (scripts/geometry package) and 'pipeline'
# (scripts/pipeline package) both resolve from here. Adding scripts/pipeline/ too would
# shadow the geometry package with scripts/pipeline/geometry.py.
sys.path.insert(0, str(ROOT / "scripts"))

from geometry.imod_positioning import (  # noqa: E402
    ImodPositioning, from_toml_table, parse_imod_positioning,
)
from pipeline import project_config as PC  # noqa: E402
from pipeline import init_project as IP  # noqa: E402

SAMPLE_TILT_COM = (
    "$tilt\n"
    "# a revised reconstruction command file\n"
    "THICKNESS 1200\n"
    "XAXISTILT 1.82\n"
    "OFFSET -11.5\n"
    "SHIFT 0.0 -8.1\n"
    "RADIAL 0.35 0.05\n"
)


def _load_module(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class GeometryRoundTripTests(unittest.TestCase):
    """The Geometry dataclass must carry the positioning table through from_dict/to_dict."""

    def _resolved_cfg(self, table):
        return {
            "project": {"basename": "TS1"},
            "paths": {"data_root": "/x", "output_dir": "/y"},
            "geometry": {
                "tilt_axis_angle_deg": 84.5,
                "raw_shape_xyz": [4096, 4096, 41], "raw_pixel_size_A": 2.0,
                "aligned_shape_xyz": [2048, 2048, 41], "aligned_pixel_size_A": 4.0,
                "target_volume_shape_xyz": [2048, 512, 2048], "target_pixel_size_A": 4.0,
                "imod_positioning": table,
            },
            "conversion": {"initial_conditions": ["raw_xf_affine_fixed"]},
        }

    def test_geometry_has_imod_positioning_field(self):
        self.assertIn("imod_positioning", PC.Geometry().__dict__)

    def test_from_dict_to_dict_preserves_positioning(self):
        table = parse_imod_positioning(
            None, unbinned_pixel_size_A=2.0,
            overrides={"tilt_angle_offset_deg": -11.5, "x_axis_tilt_deg": 1.82,
                       "shift_unbinned_px": [0.0, -8.1], "thickness_unbinned_px": 1200},
        ).to_toml_table()
        rc = PC.from_dict(self._resolved_cfg(table))
        self.assertEqual(rc.geometry.imod_positioning["tilt_angle_offset_deg"], -11.5)
        out = rc.to_dict()
        self.assertIn("imod_positioning", out["geometry"])
        self.assertEqual(out["geometry"]["imod_positioning"]["x_axis_tilt_deg"], 1.82)
        self.assertEqual(out["geometry"]["imod_positioning"]["shift_z_unbinned_px"], -8.1)

    def test_absent_positioning_is_dropped_not_faked(self):
        rc = PC.from_dict(self._resolved_cfg(None))
        self.assertIsNone(rc.geometry.imod_positioning)
        self.assertNotIn("imod_positioning", rc.to_dict()["geometry"])  # None -> filtered


class InitProjectResolutionTests(unittest.TestCase):
    """init_project resolves the table from tilt.com with config overrides + pixel scaling."""

    def _fake_inv(self, tilt_com: Path):
        inv = types.SimpleNamespace()
        inv.tilt_com = str(tilt_com)
        return inv

    def test_overrides_mapping_including_shift_pair(self):
        ov = IP._positioning_overrides({
            "tilt_angle_offset_deg": -3.0, "shift_x_unbinned_px": 1.5,
            "shift_z_unbinned_px": -2.5, "unbinned_pixel_size_A": 2.0})
        self.assertEqual(ov["tilt_angle_offset_deg"], -3.0)
        self.assertEqual(ov["shift_unbinned_px"], [1.5, -2.5])
        self.assertEqual(ov["unbinned_pixel_size_A"], 2.0)

    def test_table_parsed_from_tilt_com_with_measured_pixel(self):
        with self._tmp_tilt_com() as tc:
            table = IP._imod_positioning_table(
                self._fake_inv(tc), tc.parent, {}, {"raw_pixel_size_A": 2.0})
        self.assertEqual(table["tilt_angle_offset_deg"], -11.5)
        self.assertEqual(table["x_axis_tilt_deg"], 1.82)
        self.assertEqual(table["shift_z_unbinned_px"], -8.1)
        self.assertEqual(table["thickness_unbinned_px"], 1200)
        self.assertEqual(table["unbinned_pixel_size_A"], 2.0)      # measured raw pixel
        self.assertEqual(table["source_kind"], "tilt.com")
        self.assertIn("SHIFT", table["present_fields"])            # presence recorded

    def test_config_override_beats_tilt_com(self):
        with self._tmp_tilt_com() as tc:
            table = IP._imod_positioning_table(
                self._fake_inv(tc), tc.parent,
                {"imod_positioning": {"tilt_angle_offset_deg": -20.0}},
                {"raw_pixel_size_A": 2.0})
        self.assertEqual(table["tilt_angle_offset_deg"], -20.0)     # override wins
        self.assertEqual(table["x_axis_tilt_deg"], 1.82)           # tilt.com kept
        self.assertIn("OFFSET", table["overridden"])

    def test_nonzero_shift_without_pixel_fails_loudly(self):
        with self._tmp_tilt_com() as tc:
            with self.assertRaises(ValueError) as cm:
                IP._imod_positioning_table(self._fake_inv(tc), tc.parent, {}, {})  # no pixel
        self.assertIn("SHIFT", str(cm.exception))

    # -- helpers ----------------------------------------------------------
    def _tmp_tilt_com(self):
        import contextlib
        import tempfile

        @contextlib.contextmanager
        def _cm():
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "tilt.com"
                p.write_text(SAMPLE_TILT_COM)
                yield p
        return _cm()


class TomlSerializationTests(unittest.TestCase):
    """write_toml must emit [geometry.imod_positioning] that round-trips through a parser."""

    def test_write_toml_emits_positioning_table(self):
        import tempfile
        table = ImodPositioning(
            tilt_angle_offset_deg=-11.5, x_axis_tilt_deg=1.82,
            shift_x_unbinned_px=0.0, shift_z_unbinned_px=-8.1,
            unbinned_pixel_size_A=2.0, thickness_unbinned_px=1200,
            source_kind="tilt.com",
            present_fields=("THICKNESS", "OFFSET", "XAXISTILT", "SHIFT")).to_toml_table()
        resolved = {
            "project": {"basename": "TS1"},
            "geometry": {"tilt_axis_angle_deg": 84.5, "imod_positioning": table},
        }
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "project_settings.toml"
            IP.write_toml(out, resolved)
            parsed = tomllib.loads(out.read_text())
        pos = parsed["geometry"]["imod_positioning"]
        self.assertEqual(pos["contract_version"], 1)
        self.assertEqual(pos["tilt_angle_offset_deg"], -11.5)
        self.assertEqual(pos["shift_z_unbinned_px"], -8.1)
        self.assertEqual(pos["present_fields"], ["THICKNESS", "OFFSET", "XAXISTILT", "SHIFT"])


class StagingAndCacheContractTests(unittest.TestCase):
    """Staging manifest -> cluster rehydration, and positioning-aware marker staleness."""

    def _table(self, offset=-11.5, shift_z=-8.1):
        return ImodPositioning(
            tilt_angle_offset_deg=offset, x_axis_tilt_deg=1.82,
            shift_x_unbinned_px=0.0, shift_z_unbinned_px=shift_z,
            unbinned_pixel_size_A=2.0, thickness_unbinned_px=1200).to_toml_table()

    def test_manifest_table_rehydrates_to_equal_object(self):
        table = self._table()
        pos = from_toml_table(table)
        self.assertEqual(pos.tilt_angle_offset_deg, -11.5)
        self.assertEqual(pos.shift_z_unbinned_px, -8.1)
        self.assertAlmostEqual(pos.shift_z_A, -16.2, places=6)

    def test_positioning_hash_changes_with_any_value(self):
        base = from_toml_table(self._table()).positioning_hash()
        diff_offset = from_toml_table(self._table(offset=-10.0)).positioning_hash()
        diff_shift = from_toml_table(self._table(shift_z=-9.0)).positioning_hash()
        self.assertNotEqual(base, diff_offset)   # cache must invalidate
        self.assertNotEqual(base, diff_shift)

    def test_marker_current_predicate(self):
        rwc = _load_module("scripts/run_warp_conversion.py", "run_warp_conversion_prop_test")
        h = from_toml_table(self._table()).positioning_hash()
        # a fresh marker with the same hash is current
        self.assertTrue(rwc.positioning_marker_current({"positioning_hash": h}, h))
        # a changed positioning invalidates
        self.assertFalse(rwc.positioning_marker_current({"positioning_hash": h}, h + "x"))
        # a pre-contract marker (no hash) is stale once the manifest carries a real one
        self.assertFalse(rwc.positioning_marker_current({}, h))
        # both absent -> still current (backward compatible)
        self.assertTrue(rwc.positioning_marker_current({}, "none"))


class LegacyPathTests(unittest.TestCase):
    """02_convert_using_params must find the table and pass --imod-positioning-json."""

    def setUp(self):
        self.mod = _load_module("scripts/02_convert_using_params.py", "convert02_prop_test")

    def test_resolve_prefers_geometry_then_imod_parameters(self):
        table = {"contract_version": 1, "tilt_angle_offset_deg": -11.5}
        self.assertEqual(self.mod.resolve_positioning_table(
            {"geometry": {"imod_positioning": table}}), table)
        self.assertEqual(self.mod.resolve_positioning_table(
            {"imod_parameters": {"imod_positioning_table": table}}), table)
        self.assertIsNone(self.mod.resolve_positioning_table({"geometry": {}}))

    def test_converter_command_includes_positioning_flag(self):
        import tempfile
        cfg = {"volume_shape_xyz": [2048, 512, 2048], "alignment_mode": "full-affine",
               "axis_frame": "aligned"}
        with tempfile.TemporaryDirectory() as d:
            pj = self.mod.write_positioning_json(
                {"contract_version": 1, "tilt_angle_offset_deg": -11.5}, Path(d))
            self.assertTrue(pj.is_file())
            cmd = self.mod.converter_command(
                Path("conv.py"), Path("in"), Path("out"), 84.5, cfg, 4.0, (5, 5),
                positioning_json=pj)
        self.assertIn("--imod-positioning-json", cmd)
        self.assertIn(str(pj), cmd)

    def test_no_positioning_no_flag(self):
        self.assertIsNone(self.mod.write_positioning_json(None, Path(".")))
        cfg = {"volume_shape_xyz": [1, 1, 1]}
        cmd = self.mod.converter_command(
            Path("c.py"), Path("i"), Path("o"), 1.0, cfg, 1.0, (5, 5), positioning_json=None)
        self.assertNotIn("--imod-positioning-json", cmd)


if __name__ == "__main__":
    unittest.main()
