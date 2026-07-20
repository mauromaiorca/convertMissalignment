from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import setup_missalign_project as setup


def _resolved(extra_binning=1):
    geom = types.SimpleNamespace(
        raw_shape_xyz=[5760, 4092, 42],
        raw_pixel_size_A=1.363,
        aligned_shape_xyz=[2046, 2880, 42],
        aligned_pixel_size_A=1.363,
        tilt_axis_angle_deg=84.0,
        tilt_axis_source="test",
        target_volume_shape_xyz=[100, 100, 100],
        target_pixel_size_A=2.726,
        target_volume_physical_A=[272.6, 272.6, 272.6],
        target_volume_source="test",
    )
    return types.SimpleNamespace(
        basename="64x_Vero_02",
        conditions=["raw_xf_affine_fixed"],
        warp_alignment_modes={"raw_xf_affine_fixed": "full-affine"},
        refinement_mode="standard",
        result_backend="warp_xml",
        geometry=geom,
        extra_projection_binning=extra_binning,
    )


class SetupExtraBinningTests(unittest.TestCase):
    def test_cli_uses_safe_default_and_rejects_extra_binning(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "data"; data.mkdir()
            out = Path(td) / "out"
            calls = []

            def fake_init(**kwargs):
                calls.append(kwargs)
                out.mkdir(exist_ok=True)
                return out / "project_settings.toml", _resolved(kwargs["extra_binning"])

            with mock.patch.object(setup, "_maybe_reexec_under_env"), \
                 mock.patch.object(setup, "_canonical_init", side_effect=fake_init):
                rc = setup._initialise(["--data-dir", str(data), "--out-dir", str(out),
                                        "--basename", "64x_Vero_02", "--no-prepare"])
                self.assertEqual(rc, 0)
                self.assertEqual(calls[-1]["extra_binning"], 1)

                with self.assertRaises(SystemExit) as caught:
                    setup._initialise(["--data-dir", str(data), "--out-dir", str(out),
                                       "--basename", "64x_Vero_02", "--extra-binning", "2",
                                       "--no-prepare"])
                self.assertEqual(caught.exception.code, 2)
                self.assertEqual(len(calls), 1)

    def test_canonical_toml_records_selected_factor(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "data"; data.mkdir()
            out = Path(td) / "out"; out.mkdir()
            canonical = out / "project_settings.toml"

            def write_toml(path, config):
                lines = []
                for name, table in config.items():
                    lines.append(f"[{name}]")
                    for key, value in table.items():
                        if isinstance(value, str):
                            value = f'"{value}"'
                        elif isinstance(value, list):
                            value = "[" + ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in value) + "]"
                        lines.append(f"{key} = {value}")
                    lines.append("")
                Path(path).write_text("\n".join(lines))

            def fake_init_project(config, **kwargs):
                write_toml(canonical, {**config, "provenance": {"resolved": True}})
                man = out / "manifests"; man.mkdir(exist_ok=True)
                return {"resolved_toml": str(canonical), "manifests_dir": str(man),
                        "tilt_axis": [84.0, "test"], "warp_modes": {}}

            init_mod = types.SimpleNamespace(init_project=fake_init_project)
            pc_mod = types.SimpleNamespace(
                ConfigError=RuntimeError,
                load=lambda path: _resolved(extra_binning=2),
                validate=lambda *a, **k: [],
            )
            pipeline_mod = types.SimpleNamespace(init_project=init_mod, project_config=pc_mod)
            with mock.patch.dict(sys.modules, {
                "pipeline": pipeline_mod,
                "pipeline.init_project": init_mod,
                "pipeline.project_config": pc_mod,
            }):
                path, _ = setup._canonical_init(
                    data_dir=data, out_dir=out, basename="64x_Vero_02",
                    conditions=("raw_xf_affine_fixed",), missalign_env="/env",
                    target_shape=None, target_pixel_size=None, tilt_axis_angle=None,
                    reconstruction_stack=None, extra_binning=2,
                    cluster_profile="maxwell", cluster_profile_data={},
                    imod_module=None, imod_bin_dir=None, warp_module=None,
                    reconstruct_snapshots=("pre_missalign", "smoke", "full"),
                    disable_imod_reconstruction=False,
                    imod_cpu_partition="", imod_cpus=16,
                    imod_memory="64G", imod_time="08:00:00",
                    imod_newst_bin=0, imod_halfmaps=False,
                    imod_use_gpu=False, imod_gpu_id=0,
                )
            text = Path(path).read_text()
            self.assertIn("[multiresolution]", text)
            self.assertIn("extra_projection_binning = 2", text)

    def test_user_output_reports_factor2_working_geometry(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            setup._print_resolved(_resolved(extra_binning=2))
        out = buf.getvalue()
        self.assertIn("projection source    : 5760 × 4092 × 42 @ 1.363 Å/px", out)
        self.assertIn("extra binning        : 2", out)
        self.assertIn("Warp working stack   : 2880 × 2046 × 42 @ 2.726 Å/px", out)


try:
    import numpy as np
    from imod_affine import homogeneous_to_xf, xf_to_homogeneous
    from multiresolution import Grid2D, integer_binned_grid
    from multiresolution import transfer as T
    from pipeline import orchestrate as O
    HAVE_GEOM = True
except Exception:
    HAVE_GEOM = False


@unittest.skipUnless(HAVE_GEOM, "numpy geometry helpers unavailable")
class ProjectionBinningGeometryTests(unittest.TestCase):
    def test_prepare_binning_comes_from_toml_and_rejects_mismatch(self):
        args = types.SimpleNamespace(extra_binning=None)
        self.assertEqual(O._effective_binning({"multiresolution": {"extra_projection_binning": 2}}, args), 2)
        args.extra_binning = 4
        with self.assertRaises(ValueError):
            O._effective_binning({"multiresolution": {"extra_projection_binning": 2}}, args)

    def test_affine_grid_conversion_round_trip_and_point_round_trip(self):
        sr = Grid2D.axis_aligned("source_raw", (5760, 4092), 1.363)
        sa = Grid2D.axis_aligned("source_aligned", (2046, 2880), 1.363)
        wr = integer_binned_grid(sr, 2, out_shape_xy=(2880, 2046))
        wa = integer_binned_grid(sa, 2, out_shape_xy=(1023, 1440))
        G_r = wr.mapping_to(sr)
        G_a = wa.mapping_to(sa)
        A = np.array([[0.998, 0.012], [-0.004, 1.001]])
        d = np.array([12.5, -8.25])
        H = xf_to_homogeneous(A, d, sr.shape_xy, sa.shape_xy)
        Hw = T.h0_working(H, G_r, G_a)
        Hrt = T.h0_source_from_working(Hw, G_r, G_a)
        self.assertLess(float(np.max(np.abs(Hrt - H))), 1e-9)
        aw, dw = homogeneous_to_xf(Hw, wr.shape_xy, wa.shape_xy)
        H_again = xf_to_homogeneous(aw, dw, wr.shape_xy, wa.shape_xy)
        self.assertLess(float(np.max(np.abs(H_again - Hw))), 1e-9)

        p = np.array([123.25, 456.5, 1.0])
        p_rt = np.linalg.inv(G_r) @ (G_r @ p)
        self.assertLess(float(np.max(np.abs(p_rt - p))), 1e-9)

    def test_binned_header_validation_preserves_sections_and_pixel_size(self):
        source = types.SimpleNamespace(
            shape_xy=(5760, 4092), n_sections=42, pixel_size_xy_A=(1.363, 1.363)
        )
        measured = types.SimpleNamespace(
            shape_xy=(2880, 2046), n_sections=42, pixel_size_xy_A=(2.726, 2.726),
            mode=2, to_dict=lambda: {
                "shape_xy": [2880, 2046], "n_sections": 42,
                "pixel_size_xy_A": [2.726, 2.726], "mode": 2, "grid": {},
            },
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "bin2.mrc"; out.write_bytes(b"x")
            with mock.patch.object(O.G, "measure_mrc_grid", return_value=measured):
                info = O._validate_binned_stack(source_measure=source, output_path=out, factor=2)
        self.assertEqual(info["shape_xy"], [2880, 2046])
        self.assertEqual(info["n_sections"], 42)
        self.assertEqual(info["pixel_size_xy_A"][0], 2.726)


if __name__ == "__main__":
    unittest.main()
