"""Revised-IMOD export: canonical revision object, composition, representability,
writer layout, source protection, reports, manifest, reconstruct script, idempotency
and the optional Scipion audit.

Every test here runs WITHOUT IMOD, Scipion, WarpTools or warpylib: pure config / file /
command-construction / synthetic-geometry. `bash -n` is used only to syntax-check the
generated script and is skipped when bash is unavailable.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from imod_affine import forward_points_pixels, image_center_xy  # noqa: E402
from pipeline.imod_revision import (  # noqa: E402
    Affine2D, OriginalImodGeometry, RefinedWarpGeometry, RevisionError, RevisionPolicy,
    build_revision, compose_final_transform, converge_revision,
    sample_affine_correspondences,
)
from pipeline.imod_revision_writer import (  # noqa: E402
    ExportPaths, assert_not_under_imported, build_change_report, change_report_tsv,
    export_cache_key, render_reconstruct_script, update_com_field, write_revision_export,
)
from pipeline import scipion_compat as SC  # noqa: E402

RAW_XY = (4096, 4096)
ALI_XY = (2048, 2048)
RAW_PX, ALI_PX = 2.0, 4.0


def _original(n=5, scale=0.5, rot_deg=2.0):
    th = np.deg2rad(rot_deg)
    A = scale * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    transforms = [Affine2D(A, np.array([1.0 * i, -0.5 * i])) for i in range(n)]
    angles = [-40.0 + 20.0 * i for i in range(n)]
    return OriginalImodGeometry("TS1", RAW_XY, ALI_XY, RAW_PX, ALI_PX, transforms, angles)


def _refined_from_deltas(deltas, included=None):
    samp = [sample_affine_correspondences(d, ALI_XY)[0] for d in deltas]
    refd = [sample_affine_correspondences(d, ALI_XY)[1] for d in deltas]
    return RefinedWarpGeometry("constrained_json", samp, refd,
                               included or [True] * len(deltas))


POS = {"contract_version": 1, "tilt_angle_offset_deg": -11.5, "x_axis_tilt_deg": 1.82,
       "shift_x_unbinned_px": 0.0, "shift_z_unbinned_px": -8.1, "thickness_unbinned_px": 1200}


def _build(n=5, deltas=None, policy=None):
    og = _original(n)
    deltas = deltas or [Affine2D(np.eye(2), np.array([0.3 * i, -0.2 * i])) for i in range(n)]
    refined = _refined_from_deltas(deltas)
    return build_revision(og, refined, policy=policy or RevisionPolicy(),
                          original_positioning=POS,
                          provenance={"positioning_hash": "phash", "original_geometry_hash": "src",
                                      "refined_geometry_hash": "ref",
                                      "volume_frame_contract_version": 2})


class CompositionTests(unittest.TestCase):
    def test_identity_delta_gives_original(self):
        og = _original()
        final = compose_final_transform(og.original_transforms[0], Affine2D.identity(),
                                        raw_shape_xy=RAW_XY, aligned_shape_xy=ALI_XY)
        self.assertTrue(np.allclose(final.matrix, og.original_transforms[0].matrix, atol=1e-9))
        self.assertTrue(np.allclose(final.shift, og.original_transforms[0].shift, atol=1e-6))

    def test_H_final_equals_deltaH_at_matmul_original(self):
        # Independent check: H_final @ x == DeltaH_abs @ (H_original_abs @ x) at the centre.
        og = _original()
        delta = Affine2D(np.array([[1.001, 0.0], [0.0, 0.999]]), np.array([4.0, 2.0]))
        final = compose_final_transform(og.original_transforms[0], delta,
                                        raw_shape_xy=RAW_XY, aligned_shape_xy=ALI_XY)
        pts = np.array([[100.0, 200.0], [3000.0, 500.0]])
        via_final = forward_points_pixels(pts, final.matrix, final.shift, RAW_XY, ALI_XY, "imod")
        step1 = forward_points_pixels(pts, og.original_transforms[0].matrix,
                                      og.original_transforms[0].shift, RAW_XY, ALI_XY, "imod")
        via_two = forward_points_pixels(step1, delta.matrix, delta.shift, ALI_XY, ALI_XY, "imod")
        self.assertTrue(np.allclose(via_final, via_two, atol=1e-6))

    def test_imod_centre_convention_used(self):
        # centre is (n-1)/2, not n/2
        self.assertTrue(np.allclose(image_center_xy(ALI_XY, "imod"),
                                    [(ALI_XY[0] - 1) / 2, (ALI_XY[1] - 1) / 2]))

    def test_translation_delta_moves_centre_exactly(self):
        og = _original()
        delta = Affine2D(np.eye(2), np.array([5.0, -3.0]))
        final = compose_final_transform(og.original_transforms[2], delta,
                                        raw_shape_xy=RAW_XY, aligned_shape_xy=ALI_XY)
        c = image_center_xy(RAW_XY, "imod")[None, :]
        o = forward_points_pixels(c, og.original_transforms[2].matrix,
                                  og.original_transforms[2].shift, RAW_XY, ALI_XY, "imod")[0]
        f = forward_points_pixels(c, final.matrix, final.shift, RAW_XY, ALI_XY, "imod")[0]
        self.assertTrue(np.allclose(f - o, [5.0, -3.0], atol=1e-6))


class RepresentabilityTests(unittest.TestCase):
    def test_exact_affine_classified(self):
        rev = _build()
        self.assertTrue(all(c == "exact_affine" for c in rev.representability.tilt_class))

    def test_non_affine_fail_policy_raises(self):
        og = _original(n=3)
        grid = sample_affine_correspondences(Affine2D.identity(), ALI_XY)[0]
        refined_pts = []
        for i in range(3):
            r = grid.copy().astype(float)
            if i == 1:  # add a non-affine quadratic bump to tilt 1
                r[:, 0] += 0.02 * ALI_XY[0] * ((grid[:, 0] / ALI_XY[0] - 0.5) ** 2)
            refined_pts.append(r)
        refined = RefinedWarpGeometry("warp_xml", [grid] * 3, refined_pts, [True] * 3)
        with self.assertRaises(RevisionError):
            build_revision(og, refined, policy=RevisionPolicy(non_affine_policy="fail"))

    def test_affine_within_tolerance_class(self):
        # residual just under tolerance -> affine_within_tolerance via converge path
        og = _original(n=2)
        deltas = [Affine2D.identity(), Affine2D.identity()]
        stats = [{"rms_residual_px": 0.05, "max_residual_px": 0.2}] * 2
        rev = converge_revision(og, deltas, policy=RevisionPolicy(), backend="warp_xml",
                                representability_stats=stats)
        self.assertEqual(rev.representability.worst_class, "affine_within_tolerance")

    def test_tolerance_boundary(self):
        og = _original(n=1)
        # rms within, max over -> non_affine
        stats = [{"rms_residual_px": 0.05, "max_residual_px": 0.30}]
        with self.assertRaises(RevisionError):
            converge_revision(og, [Affine2D.identity()], policy=RevisionPolicy(non_affine_policy="fail"),
                              backend="warp_xml", representability_stats=stats)


class WriterLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        self.imported = self.tmp / "imported_data" / "imod"
        (self.imported / "data").mkdir(parents=True)
        self.raw = self.imported / "data" / "TS1.mrc"
        self.raw.write_bytes(b"MRCFAKE" * 100)
        self.phys = self.tmp / "exported_data" / "imod" / "17.6Apx"
        self.link = self.tmp / "missalignment" / "runs" / "17.6Apx" / "export" / "imod"
        self.link.parent.mkdir(parents=True)

    def _write(self, rev=None, policy=None):
        rev = rev or _build()
        paths = ExportPaths.resolve(self.phys, self.link)
        return paths, write_revision_export(
            rev, paths, policy=policy or RevisionPolicy(), imported_imod_dir=self.imported,
            raw_stack_source=self.raw,
            original_tilt_com="$tilt\nTHICKNESS 1200\nOFFSET -11.5\nRADIAL 0.35 0.05\nXAXISTILT 1.82\n",
            original_newst_com="$newstack\nBinByFactor 2\nSizeToOutputInXandY 2048,2048\n",
            source_hashes={"raw_stack": {"sha256": "deadbeefcafe", "path": str(self.raw)}},
            measured_pixel_size_A=17.596357, condition_id="17.6Apx")

    def test_one_physical_dir_and_one_symlink(self):
        self._write()
        self.assertTrue(self.phys.is_dir() and not self.phys.is_symlink())
        self.assertTrue(self.link.is_symlink())
        self.assertEqual(os.path.realpath(self.link), os.path.realpath(self.phys))

    def test_exact_visible_tree_no_extra_dirs(self):
        self._write()
        visible = sorted(p.name for p in self.phys.iterdir())
        self.assertEqual(set(visible), {
            "configuration", "data", "reconstruct_with_imod.sh", "manifest.json",
            "alignment_change_report.json", "alignment_change_report.tsv",
            "alignment_change_summary.txt", "scipion_compatibility.json"})
        for forbidden in ("original", "residual", "validation", "scripts", "reports", "provenance"):
            self.assertNotIn(forbidden, visible)

    def test_no_duplicate_export_tree(self):
        self._write()
        # the compat link must not be a second physical copy
        self.assertTrue(self.link.is_symlink())
        # configuration exists only once physically
        configs = [p for p in self.phys.rglob("configuration") if p.is_dir() and not p.is_symlink()]
        self.assertEqual(len(configs), 1)

    def test_relative_raw_stack_symlink(self):
        paths, _ = self._write()
        raw_link = paths.data_dir / "TS1.mrc"
        self.assertTrue(raw_link.is_symlink())
        self.assertFalse(os.path.isabs(os.readlink(raw_link)))
        self.assertEqual(os.path.realpath(raw_link), os.path.realpath(self.raw))

    def test_imported_data_not_mutated(self):
        before = self.raw.read_bytes()
        self._write()
        self.assertEqual(self.raw.read_bytes(), before)
        # nothing new written under imported_data/imod
        self.assertEqual(sorted(p.name for p in (self.imported / "data").iterdir()), ["TS1.mrc"])

    def test_final_vs_residual_xf_distinct(self):
        self._write()
        final = (self.phys / "configuration" / "TS1.xf").read_text().splitlines()
        residual = (self.phys / "configuration" / "TS1.residual.xf").read_text().splitlines()
        self.assertEqual(len(final), 5)
        self.assertEqual(len(residual), 5)
        self.assertNotEqual(final[2], residual[2])

    def test_manifest_distinguishes_final_and_residual_semantics(self):
        _, man = self._write()
        self.assertIn("complete revised raw->aligned", man["final_xf_semantics"])
        self.assertIn("NOT a complete", man["residual_xf_semantics"])
        self.assertIsNotNone(man["final_xf"]["sha256"])
        self.assertIsNotNone(man["residual_xf"]["sha256"])
        self.assertNotEqual(man["final_xf"]["sha256"], man["residual_xf"]["sha256"])

    def test_unchanged_tlt_reported_explicitly(self):
        _, man = self._write()
        self.assertTrue(man["tlt"]["unchanged"])
        self.assertTrue((self.phys / "configuration" / "TS1.tlt").is_file())

    def test_idempotent_no_new_dirs(self):
        self._write()
        first = sorted(p.name for p in self.phys.iterdir())
        run_dirs_before = list((self.tmp / "exported_data" / "imod").iterdir())
        self._write()  # re-run identical
        self.assertEqual(sorted(p.name for p in self.phys.iterdir()), first)
        run_dirs_after = list((self.tmp / "exported_data" / "imod").iterdir())
        self.assertEqual(len(run_dirs_before), len(run_dirs_after))  # no numbered/timestamped dirs
        self.assertEqual([p.name for p in run_dirs_after], ["17.6Apx"])

    def test_source_protection_refuses_under_imported(self):
        with self.assertRaises(RevisionError):
            assert_not_under_imported(self.imported / "data" / "x.xf", self.imported)
        # end-to-end refuse
        paths = ExportPaths.resolve(self.imported / "sneaky", self.link)
        with self.assertRaises(RevisionError):
            write_revision_export(_build(), paths, policy=RevisionPolicy(),
                                  imported_imod_dir=self.imported, raw_stack_source=self.raw)


class CommandFileTests(unittest.TestCase):
    def test_update_com_field_preserves_unrelated_lines(self):
        text = "$tilt\n# comment\nTHICKNESS 1200\nRADIAL 0.35 0.05\nOFFSET -11.5\n"
        out = update_com_field(text, "OFFSET", "-11.5")
        self.assertIn("RADIAL 0.35 0.05", out)
        self.assertIn("# comment", out)
        self.assertIn("$tilt", out)
        self.assertEqual(out.count("OFFSET"), 1)

    def test_update_com_field_appends_when_absent(self):
        out = update_com_field("$tilt\nTHICKNESS 1200\n", "OFFSET", "-3.0")
        self.assertIn("OFFSET", out)
        self.assertIn("THICKNESS 1200", out)

    def test_offset_applied_once_not_baked_into_tlt(self):
        # revised .tlt holds raw angles; OFFSET stays a separate tilt.com field
        tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        imported = tmp / "imported_data" / "imod"; (imported / "data").mkdir(parents=True)
        raw = imported / "data" / "TS1.mrc"; raw.write_bytes(b"x" * 10)
        phys = tmp / "exported_data" / "imod" / "5Apx"
        rev = _build()
        paths = ExportPaths.resolve(phys, tmp / "runs" / "export" / "imod")
        write_revision_export(rev, paths, policy=RevisionPolicy(), imported_imod_dir=imported,
                              raw_stack_source=raw, original_tilt_com="$tilt\nOFFSET -11.5\n",
                              condition_id="5Apx")
        tlt_vals = [float(x) for x in (phys / "configuration" / "TS1.tlt").read_text().split()]
        self.assertTrue(np.allclose(tlt_vals, rev.original_tilt_angles_deg, atol=1e-6))  # raw, not +OFFSET
        tc = (phys / "configuration" / "tilt.com").read_text()
        self.assertEqual(tc.count("OFFSET"), 1)  # OFFSET present exactly once
        self.assertIn("-11.5", tc)

    def test_positioning_preserved_offset_xaxis_shift_thickness(self):
        tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        imported = tmp / "imported_data" / "imod"; (imported / "data").mkdir(parents=True)
        raw = imported / "data" / "TS1.mrc"; raw.write_bytes(b"x" * 10)
        phys = tmp / "exported_data" / "imod" / "5Apx"
        rev = _build()  # revised_positioning empty -> preserve original
        paths = ExportPaths.resolve(phys, tmp / "runs" / "export" / "imod")
        _, man = None, write_revision_export(
            rev, paths, policy=RevisionPolicy(), imported_imod_dir=imported, raw_stack_source=raw,
            original_tilt_com="$tilt\nTHICKNESS 1200\nOFFSET -11.5\nXAXISTILT 1.82\nSHIFT 0.0 -8.1\n",
            condition_id="5Apx")
        tc = (phys / "configuration" / "tilt.com").read_text()
        self.assertIn("THICKNESS 1200", tc)     # THICKNESS preserved
        self.assertIn("XAXISTILT 1.82", tc)     # XAXISTILT unchanged (sign not cluster-validated)
        self.assertIn("SHIFT 0 -8.1", tc.replace("0.0", "0"))  # SHIFT preserved
        rep = json.loads((phys / "alignment_change_report.json").read_text())
        for field in ("tilt_angle_offset_deg", "x_axis_tilt_deg", "shift_z_unbinned_px",
                      "thickness_unbinned_px"):
            self.assertTrue(rep["project"]["positioning_changes"][field]["unchanged"])


class ReconstructScriptTests(unittest.TestCase):
    def _script(self):
        return render_reconstruct_script(series="TS1", condition_id="17.6Apx",
                                         imported_imod_dir=Path("/proj/imported_data/imod"),
                                         default_out="/proj/exported_data/imod/17.6Apx/reconstruction")

    def test_has_strict_bash_and_guards(self):
        s = self._script()
        self.assertIn("set -euo pipefail", s)
        self.assertIn("BASH_SOURCE", s)                       # self-locating
        self.assertIn("refusing to write under imported_data/imod", s)
        self.assertIn("--output-dir", s)
        self.assertIn("raw-stack link does not resolve", s)
        self.assertIn("!= stack sections", s)                 # row-count checks
        self.assertIn("!= recorded", s)                       # source-hash check

    def test_newstack_and_tilt_command_construction(self):
        s = self._script()
        self.assertIn("newstack -InputFile", s)
        self.assertIn("-TransformFile", s)
        self.assertIn("$SERIES.missalign_ali.mrc", s)   # SERIES is a shell var
        self.assertIn("$SERIES.missalign_rec.mrc", s)
        self.assertIn("submfg tilt.com", s)                   # prefer command file
        self.assertIn("tilt -InputProjections", s)            # explicit fallback

    def test_bash_syntax_valid(self):
        if shutil.which("bash") is None:
            self.skipTest("bash not available")
        tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        script = tmp / "reconstruct_with_imod.sh"
        script.write_text(self._script())
        r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


class ReportAndCacheTests(unittest.TestCase):
    def test_report_uses_detector_grid_displacement(self):
        rev = _build()
        report = build_change_report(rev, aligned_pixel_size_A=ALI_PX, raw_pixel_size_A=RAW_PX)
        t = report["per_tilt"][3]
        self.assertIn("rms_displacement_px", t)
        self.assertIn("corner_displacement_px", t)
        self.assertIn("centre_displacement_A", t)
        self.assertEqual(len(t["corner_displacement_px"]), 4)
        # tilt 0 delta is zero -> ~no displacement; tilt 4 delta larger
        self.assertLess(report["per_tilt"][0]["rms_displacement_px"], 1e-6)
        self.assertGreater(report["per_tilt"][4]["rms_displacement_px"], 0.0)

    def test_report_project_level_fields(self):
        rev = _build()
        report = build_change_report(rev, aligned_pixel_size_A=ALI_PX, raw_pixel_size_A=RAW_PX)
        p = report["project"]
        for key in ("n_included", "n_excluded", "n_modified", "n_unchanged",
                    "global_rms_displacement_px", "tilt_with_max_displacement",
                    "original_effective_angle_range_deg", "positioning_changes"):
            self.assertIn(key, p)

    def test_tsv_has_header_and_rows(self):
        rev = _build()
        tsv = change_report_tsv(build_change_report(rev, aligned_pixel_size_A=ALI_PX,
                                                    raw_pixel_size_A=RAW_PX))
        lines = tsv.strip().splitlines()
        self.assertEqual(len(lines), 1 + rev.n_tilts)
        self.assertIn("representability_class", lines[0])

    def test_cache_key_invalidation(self):
        base = dict(source_geometry_hash="s", refined_geometry_hash="r",
                    positioning_hash="p", volume_frame_contract_version=2,
                    policy=RevisionPolicy(), imod_version="5.1.11")
        k0 = export_cache_key(**base)
        self.assertEqual(k0, export_cache_key(**base))  # deterministic
        self.assertNotEqual(k0, export_cache_key(**{**base, "positioning_hash": "p2"}))
        self.assertNotEqual(k0, export_cache_key(**{**base, "refined_geometry_hash": "r2"}))
        self.assertNotEqual(k0, export_cache_key(**{**base, "imod_version": "5.1.9"}))
        self.assertNotEqual(k0, export_cache_key(
            **{**base, "policy": RevisionPolicy(affine_fit_rms_tolerance_px=0.2)}))


class ScipionOptionalTests(unittest.TestCase):
    def test_absent_provider_not_required_is_not_covered(self):
        rev = _build()
        r = SC.audit_revision(rev, required=False)
        self.assertEqual(r["status"], "NOT_COVERED_BY_SCIPION")

    def test_absent_provider_required_is_unresolved(self):
        rev = _build()
        r = SC.audit_revision(rev, required=True)
        self.assertEqual(r["status"], "UNRESOLVED")

    def test_matching_provider_is_compatible(self):
        rev = _build()
        og = rev.original_geometry

        def provider(i, pts):
            f = rev.final_transforms[i]
            return forward_points_pixels(pts, f.matrix, f.shift, og.raw_shape_xy,
                                         og.aligned_shape_xy, "imod")
        r = SC.audit_revision(rev, scipion_mapping_provider=provider)
        self.assertEqual(r["status"], "COMPATIBLE")

    def test_disagreeing_provider_is_incompatible_with_diagnostic(self):
        rev = _build()

        def provider(i, pts):
            return np.asarray(pts, dtype=float) + 50.0  # gross disagreement
        r = SC.audit_revision(rev, scipion_mapping_provider=provider)
        self.assertEqual(r["status"], "INCOMPATIBLE")
        bad = [c for c in r["comparisons"] if c["status"] == "INCOMPATIBLE"][0]
        self.assertIn("diagnostic", bad)
        self.assertIn("suspected_convention_difference", bad["diagnostic"])


class OrchestrationCliTests(unittest.TestCase):
    """Drive `export revise` main() end-to-end from synthetic finalize outputs (no warpylib)."""

    def _project(self, tmp, backend="constrained_json", residual_shift=lambda i: (0.3 * i, -0.2 * i)):
        from imod_affine import write_xf
        from pipeline.init_project import write_toml
        out = tmp / "proj"
        (out / "imported_data" / "imod" / "data").mkdir(parents=True)
        (out / "imported_data" / "imod" / "configuration").mkdir(parents=True)
        src = tmp / "etomo"; src.mkdir()
        raw = src / "TS1.mrc"; raw.write_bytes(b"MRC" * 50)
        n = 5
        write_xf(src / "TS1.xf", np.stack([0.5 * np.eye(2)] * n),
                 np.stack([np.array([1.0 * i, 0.0]) for i in range(n)]))
        (src / "TS1.tlt").write_text("".join(f"{-40 + 20 * i:.2f}\n" for i in range(n)))
        (src / "tilt.com").write_text("$tilt\nTHICKNESS 1200\nOFFSET -11.5\nXAXISTILT 1.82\nRADIAL 0.35 0.05\n")
        (src / "newst.com").write_text("$newstack\nBinByFactor 2\n")
        tdir = out / "missalignment" / "runs" / "5Apx" / "results" / "transforms"
        tdir.mkdir(parents=True)
        write_xf(tdir / "source_residual.xf", np.stack([np.eye(2)] * n),
                 np.stack([np.array(residual_shift(i)) for i in range(n)]))
        config = {
            "project": {"basename": "TS1"}, "paths": {"output_dir": str(out)},
            "input": {"raw_stack": str(raw), "final_xf_file": str(src / "TS1.xf"),
                      "final_tilt_file": str(src / "TS1.tlt"), "tilt_com": str(src / "tilt.com"),
                      "newst_com": str(src / "newst.com")},
            "geometry": {"raw_shape_xyz": [4096, 4096, n], "raw_pixel_size_A": 2.0,
                         "aligned_shape_xyz": [2048, 2048, n], "aligned_pixel_size_A": 4.0,
                         "target_pixel_size_A": 5.0, "tilt_axis_angle_deg": 84.5,
                         "imod_positioning": {"contract_version": 1, "tilt_angle_offset_deg": -11.5,
                                              "x_axis_tilt_deg": 1.82, "shift_x_unbinned_px": 0.0,
                                              "shift_z_unbinned_px": -8.1, "thickness_unbinned_px": 1200,
                                              "positioning_hash": "HH"}},
            "datasets": {"native_id": "5Apx", "selected_id": "5Apx"},
            "missalignment": {"result_backend": backend},
            "export": {"imod_revision": {"enabled": True, "non_affine_policy": "fail"}},
            "provenance": {"resolved": True},
        }
        settings = out / "project_settings.toml"
        write_toml(settings, config)
        return out, settings, raw

    def test_main_publishes_single_export(self):
        from pipeline.imod_revision_export import main
        tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        out, settings, raw = self._project(tmp)
        self.assertEqual(main([str(settings)]), 0)
        phys = out / "exported_data" / "imod" / "5Apx"
        link = out / "missalignment" / "runs" / "5Apx" / "export" / "imod"
        self.assertTrue(phys.is_dir() and link.is_symlink())
        man = json.loads((phys / "manifest.json").read_text())
        self.assertEqual(man["condition_id"], "5Apx")            # canonical id, not measured
        self.assertEqual(man["reconstruction_angpix_A"], 5.0)
        self.assertEqual(raw.read_bytes(), b"MRC" * 50)          # source untouched

    def test_main_missing_finalize_transforms_errors(self):
        from pipeline.imod_revision_export import main
        tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        out, settings, _ = self._project(tmp)
        (out / "missalignment" / "runs" / "5Apx" / "results" / "transforms" / "source_residual.xf").unlink()
        self.assertEqual(main([str(settings)]), 3)               # ExportInputsMissing -> rc 3


if __name__ == "__main__":
    unittest.main()
