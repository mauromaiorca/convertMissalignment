from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v6.config import CapabilitySet, ClusterConfig, ProjectConfig, SoftwareConfig, TiltSeriesConfig
from v6.jobs import generate_stage_jobs
from v6.stage_result import write_result
from v6.stages import StagePlanningError, plan_stages
from v6.warp_project import SnapshotManager, V6Layout


def fake_tools(bin_dir: Path) -> None:
    bin_dir.mkdir()
    from tests.test_v6_config_sources import write_fake_warptools
    write_fake_warptools(bin_dir / "WarpTools")
    (bin_dir / "miss-alignment").write_text("#!/usr/bin/env bash\necho 'miss-alignment fake 1.0'\n")
    (bin_dir / "miss-alignment").chmod(0o755)


def write_mrc(path: Path, *, nx=4, ny=4, nz=3, pixel=1.5) -> None:
    header = bytearray(1024)
    struct.pack_into("<4i", header, 0, nx, ny, nz, 2)
    struct.pack_into("<3i", header, 28, nx, ny, nz)
    struct.pack_into("<3f", header, 40, nx * pixel, ny * pixel, nz * pixel)
    header[208:212] = b"MAP "
    data = struct.pack("<" + "f" * (nx * ny * nz), *[float(i) for i in range(nx * ny * nz)])
    path.write_bytes(header + data)


def write_stack_fixture(data: Path, *, basename="TS1", nz=3) -> None:
    write_mrc(data / f"{basename}.st", nz=nz)
    write_mrc(data / f"{basename}.ali", nz=nz)
    write_mrc(data / f"{basename}_rec.mrc", nx=6, ny=6, nz=5)
    (data / f"{basename}.tlt").write_text("\n".join(str(x) for x in range(nz)) + "\n")
    (data / f"{basename}.xf").write_text("".join("1 0 0 1 0 0\n" for _ in range(nz)))


class V6SnapshotAndPlanTests(unittest.TestCase):
    def _cfg(self, source_mode="tilt_stack", motion=False):
        ts = TiltSeriesConfig(id="TS1", basename="TS1")
        ts.source.mode = source_mode
        ts.capabilities = CapabilitySet(
            movies_available=source_mode == "movies",
            acquisition_order_known=False,
            tilt_ctf_available=True,
            imod_alignment_available=True,
            motion_refinement_in_m_available=source_mode == "movies",
        )
        cfg = ProjectConfig(
            schema_version=6,
            project={"name": "TS1", "output_dir": "/tmp/out", "software_version": "6.0.0"},
            cluster=ClusterConfig(gres=""),
            software=SoftwareConfig(),
            tilt_series=[ts],
        )
        cfg.m.motion_refinement = motion
        return cfg

    def test_stack_only_rejects_m_motion_refinement_and_movie_ingest(self):
        with self.assertRaises(StagePlanningError):
            plan_stages(self._cfg("tilt_stack", motion=True))
        with self.assertRaises(StagePlanningError):
            plan_stages(self._cfg("movies"))

    def test_snapshot_ids_are_unique_and_smoke_full_do_not_share_xml(self):
        with tempfile.TemporaryDirectory() as td:
            layout = V6Layout(Path(td)).create()
            mgr = SnapshotManager(layout, "a" * 64)
            pre_xml = Path(td) / "pre.xml"; pre_xml.write_text("<xml />\n")
            smoke_xml = Path(td) / "smoke.xml"; smoke_xml.write_text("<xml smoke='1' />\n")
            full_xml = Path(td) / "full.xml"; full_xml.write_text("<xml full='1' />\n")
            pre = mgr.create_snapshot("pre_missalign", parent_snapshot_id="base", copy_files=[pre_xml])
            smoke = mgr.create_snapshot("missalign_smoke", parent_snapshot_id=pre.snapshot_id, copy_files=[smoke_xml])
            full = mgr.create_snapshot("missalign_full", parent_snapshot_id=pre.snapshot_id, copy_files=[full_xml])
            self.assertEqual(smoke.parent_snapshot_id, pre.snapshot_id)
            self.assertEqual(full.parent_snapshot_id, pre.snapshot_id)
            self.assertNotEqual(smoke.snapshot_id, full.snapshot_id)
            Path(smoke.copied_files[0]).write_text("changed\n")
            self.assertNotEqual(Path(smoke.copied_files[0]).read_text(), Path(full.copied_files[0]).read_text())
            self.assertNotEqual(Path(smoke.copied_files[0]).read_text(), Path(pre.copied_files[0]).read_text())

    def test_generated_jobs_parse_and_failure_manifest_is_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            settings = root / "project_settings.toml"; settings.write_text("schema_version = 6\n")
            cfg = self._cfg()
            stages = plan_stages(cfg)
            written = generate_stage_jobs(
                jobs_dir=root / "jobs", run_dir=root, settings_path=settings,
                toml_hash="deadbeef", cluster=cfg.cluster, stages=stages)
            self.assertEqual(set(written), {"10_warp_ingest", "20_initial_alignment_and_qc"})
            for path in written.values():
                cp = subprocess.run(["bash", "-n", path], text=True, capture_output=True, check=False)
                self.assertEqual(cp.returncode, 0, cp.stderr)
                self.assertNotIn("sbatch ", Path(path).read_text())
            write_result(
                run_dir=root,
                stage_id="10_warp_ingest",
                status="failed",
                exit_code=7,
                failed_command='python - <<EOF\nprint("quoted")\nEOF',
            )
            data = json.loads((root / "manifests" / "10_warp_ingest_stage_result.json").read_text())
            self.assertIn('print("quoted")', data["failed_command"])

    def test_generated_10_and_20_execute_with_fake_tools_and_unlock_30(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"; data.mkdir()
            write_stack_fixture(data)
            fake = root / "bin"
            fake_tools(fake)
            out = root / "out"
            env = os.environ.copy()
            env["PATH"] = f"{fake}:{env.get('PATH', '')}"
            setup = subprocess.run([
                sys.executable, str(ROOT / "setup_warp_project.py"),
                "--data-dir", str(data),
                "--basename", "TS1",
                "--out-dir", str(out),
                "--source-mode", "tilt_stack",
            ], text=True, capture_output=True, check=False, env=env)
            self.assertEqual(setup.returncode, 0, setup.stderr + setup.stdout)
            self.assertFalse((out / "jobs" / "20_initial_alignment_and_qc.sbatch").exists())
            cp = subprocess.run(["bash", "-n", str(out / "jobs" / "10_warp_ingest.sbatch")],
                                text=True, capture_output=True, check=False)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            run = subprocess.run(["bash", str(out / "jobs" / "10_warp_ingest.sbatch")],
                                 text=True, capture_output=True, check=False, env=env)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            self.assertTrue((out / "warp" / "base" / "snapshot_manifest.json").is_file())
            self.assertTrue((out / "warp" / "base" / "tilt_series.settings").is_file())
            self.assertFalse((out / "warp" / "base" / "tilt_series.settings").read_text().lstrip().startswith("{"))
            self.assertFalse((out / "jobs" / "30_missalignment.sbatch").exists())
            status = subprocess.run([sys.executable, str(ROOT / "warp_project.py"), "status", str(out / "project_settings.toml")],
                                    text=True, capture_output=True, check=False, env=env)
            self.assertEqual(status.returncode, 0, status.stderr + status.stdout)
            self.assertIn("20_initial_alignment_and_qc is blocked", status.stdout)

    def test_completed_without_validated_does_not_unlock_dependency(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            settings = root / "project_settings.toml"
            cfg = self._cfg()
            cfg.project["output_dir"] = str(root)
            from v6.config import write_toml, to_plain
            write_toml(settings, to_plain(cfg))
            V6Layout(root).create()
            write_result(run_dir=root, stage_id="10_warp_ingest", status="completed")
            status = subprocess.run([sys.executable, str(ROOT / "warp_project.py"), "status", str(settings)],
                                    text=True, capture_output=True, check=False)
            self.assertEqual(status.returncode, 0, status.stderr + status.stdout)
            self.assertIn("completed but is not validated", status.stdout)


if __name__ == "__main__":
    unittest.main()
