"""Focused tests for TOML-driven Phase 3 IMOD reconstruction planning."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline import imod_reconstruction as IR


def _write_settings(tmp: Path) -> Path:
    data = tmp / "data"
    data.mkdir()
    for name in ("raw.mrc", "base.xf", "angles.tlt", "newst.com", "tilt.com"):
        (data / name).write_text("x\n")
    (data / "base.xf").write_text("1 0 0 1 0 0\n")
    (data / "angles.tlt").write_text("0\n")
    (data / "newst.com").write_text(
        "$setenv IMOD_OUTPUT_FORMAT MRC\n$newstack -StandardInput\n"
        "InputFile old.mrc\nTransformFile old.xf\nOutputFile old_ali.mrc\n"
        "$if (-e ./savework) ./savework\n"
    )
    (data / "tilt.com").write_text(
        "$tilt -StandardInput\nInputProjections old_ali.mrc\nOutputFile old.rec\nTiltFile old.tlt\n"
    )
    settings = tmp / "project_settings.toml"
    settings.write_text(f"""
[project]
basename = "series"
schema_version = 2
[paths]
data_root = "{data}"
output_dir = "{tmp}/runs"
[input]
raw_stack = "{data / 'raw.mrc'}"
final_tilt_file = "{data / 'angles.tlt'}"
newst_com = "{data / 'newst.com'}"
tilt_com = "{data / 'tilt.com'}"
[geometry]
raw_shape_xyz = [4, 4, 1]
raw_pixel_size_A = 1.0
target_volume_shape_xyz = [4, 4, 4]
target_pixel_size_A = 1.0
[conversion]
initial_conditions = ["raw_xf_affine_fixed"]
[conversion.condition_modes]
raw_xf_affine_fixed = "full-affine"
[multiresolution]
extra_projection_binning = 1
[ctf]
mode = "off"
[missalignment]
refinement_mode = "standard"
result_backend = "warp_xml"
[cluster]
environment = "/env"
[reconstruction]
enabled = true
backend = "imod"
snapshots = ["pre_missalign", "smoke", "full"]
canonical_snapshot = "full"
diagnostic_snapshots = ["pre_missalign", "smoke"]
[reconstruction.imod]
newst_template = "{data / 'newst.com'}"
tilt_template = "{data / 'tilt.com'}"
newstack_executable = "newstack"
tilt_executable = "tilt"
submfg_executable = "submfg"
execution_mode = "submfg_command_file"
newst_bin = 0
use_gpu = false
gpu_id = 0
[reconstruction.outputs]
canonical_root = "final/reconstruction"
diagnostic_root = "diagnostics/reconstruction_validation"
[reconstruction.validation]
require_stack_section_match = true
[provenance]
resolved = true
""")
    return settings


class Phase3ReconstructionIntegrationTests(unittest.TestCase):
    def test_full_snapshot_requires_result_manifest_final_xml(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            settings = _write_settings(tmp)
            run_dir = tmp / "runs" / "series_raw_xf_affine_fixed_standard"
            (run_dir / "manifests").mkdir(parents=True)
            (run_dir / "manifests" / "result_manifest.json").write_text(json.dumps({"final_xml": None}))
            with self.assertRaisesRegex(IR.ReconstructionError, "final_xml"):
                IR.build_plan(settings, "full")

    def test_pre_missalign_xml_is_resolved_from_snapshot_directory(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            settings = _write_settings(tmp)
            run_dir = tmp / "runs" / "series_raw_xf_affine_fixed_standard"
            warp = run_dir / "warp" / "pre_missalign"
            warp.mkdir(parents=True)
            xml = warp / "series.xml"
            xml.write_text("<xml/>\n")
            (run_dir / "manifests").mkdir(parents=True)
            (run_dir / "manifests" / "result_manifest.json").write_text(json.dumps({
                "pre_missalign_directory": str(warp)
            }))
            plan = IR.build_plan(settings, "pre_missalign")
            self.assertEqual(plan.xml, xml.resolve())
            self.assertIn("diagnostics/reconstruction_validation/pre_missalign", str(plan.work_dir))

    def test_materialize_command_files_uses_controlled_updates(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            settings = _write_settings(tmp)
            run_dir = tmp / "runs" / "series_raw_xf_affine_fixed_standard"
            warp = run_dir / "warp" / "pre_missalign"
            warp.mkdir(parents=True)
            (warp / "series.xml").write_text("<xml/>\n")
            (run_dir / "manifests").mkdir(parents=True)
            (run_dir / "manifests" / "result_manifest.json").write_text(json.dumps({
                "pre_missalign_directory": str(warp)
            }))
            plan = IR.build_plan(settings, "pre_missalign")
            plan.work_dir.mkdir(parents=True)
            plan.exported_xf.parent.mkdir(parents=True)
            plan.exported_xf.write_text("1 0 0 1 0 0\n")
            report = IR.materialize_command_files(plan)
            self.assertIn("InputFile", plan.generated_newst.read_text())
            self.assertIn(str(plan.raw_stack), plan.generated_newst.read_text())
            self.assertIn(plan.aligned_stack.name, plan.generated_tilt.read_text())
            self.assertEqual(report["newst"]["updates"][0]["key"], "InputFile")
            newst = IR.validate_command_file(plan.generated_newst, expected="newstack", forbidden="tilt")
            tilt = IR.validate_command_file(plan.generated_tilt, expected="tilt", forbidden="newstack")
            self.assertEqual([c["program"] for c in newst["commands"] if c["program"] == "newstack"], ["newstack"])
            self.assertEqual([c["program"] for c in tilt["commands"] if c["program"] == "tilt"], ["tilt"])

    def test_rejects_wrong_or_ambiguous_command_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "newst.com"
            path.write_text("$newstack -StandardInput\n$tilt -StandardInput\n")
            with self.assertRaisesRegex(IR.ReconstructionError, "invalid IMOD command file"):
                IR.validate_command_file(path, expected="newstack", forbidden="tilt")

    def test_run_imod_command_file_invokes_submfg_not_program_directly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            calls = root / "calls.txt"
            fake = root / "submfg"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                f"pathlib.Path({str(calls)!r}).write_text(' '.join(sys.argv[1:]))\n"
                "pathlib.Path(pathlib.Path(sys.argv[1]).stem + '.log').write_text('native log\\n')\n"
            )
            fake.chmod(0o755)
            com = root / "newst.com"
            com.write_text("$setenv IMOD_OUTPUT_FORMAT MRC\n$newstack -StandardInput\nInputFile raw.mrc\nOutputFile ali.mrc\nTransformFile series.xf\n")
            result = IR.run_imod_command_file(
                submfg=str(fake), command_file=com, cwd=root,
                consolidated_log=root / "newstack.log")
            self.assertEqual(calls.read_text(), "newst.com")
            self.assertIn("native log", (root / "newstack.log").read_text())
            self.assertEqual(Path(result["native_log"]).name, "newst.log")

    def test_runtime_hash_mismatch_blocks_execution(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            settings = _write_settings(tmp)
            run_dir = tmp / "runs" / "series_raw_xf_affine_fixed_standard"
            warp = run_dir / "warp" / "pre_missalign"
            warp.mkdir(parents=True)
            (warp / "series.xml").write_text("<xml/>\n")
            (run_dir / "manifests").mkdir(parents=True)
            (run_dir / "manifests" / "result_manifest.json").write_text(json.dumps({
                "pre_missalign_directory": str(warp)
            }))
            plan = IR.build_plan(settings, "pre_missalign")
            with self.assertRaisesRegex(IR.ReconstructionError, "executor changed"):
                IR.verify_runtime_hashes(
                    plan, expected_executor_sha="0" * 64,
                    expected_settings_sha=IR.sha256_file(settings),
                    allow_mismatch=False)


if __name__ == "__main__":
    unittest.main()
