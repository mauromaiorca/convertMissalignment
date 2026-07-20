from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v6.config import V6ConfigError, load, load_v5_compatibility_config
from v6.sources import SourceDiscoveryError, resolve_source


def write_fake_warptools(path: Path) -> None:
    path.write_text("""#!/usr/bin/env python3
import sys
from pathlib import Path

args = sys.argv[1:]
if args == ["--version"]:
    print("WarpTools fake contract 1.0")
    raise SystemExit(0)
if args == ["--help"]:
    print("commands: stack-ingest validate-project")
    raise SystemExit(0)
if args == ["stack-ingest", "--help"]:
    print("stack-ingest --input-stack --tilt-file --output-dir --series-id --pixel-size")
    raise SystemExit(0)
if args == ["validate-project", "--help"]:
    print("validate-project --project-dir")
    raise SystemExit(0)
if args and args[0] == "stack-ingest":
    opts = dict(zip(args[1::2], args[2::2]))
    out = Path(opts["--output-dir"])
    sid = opts["--series-id"]
    (out / "tomostar").mkdir(parents=True, exist_ok=True)
    (out / "processing").mkdir(parents=True, exist_ok=True)
    (out / "frame_series.settings").write_text("WarpFrameSeriesSettings\\n")
    (out / "tilt_series.settings").write_text("WarpTiltSeriesSettings\\n")
    (out / "tomostar" / f"{sid}.tomostar").write_text("data_\\n")
    (out / f"{sid}.xml").write_text(f"<TiltSeries Id='{sid}' PixelSize='{opts['--pixel-size']}' />\\n")
    (out / "processing" / "metadata.txt").write_text("accepted\\n")
    print("stack ingest complete")
    raise SystemExit(0)
if args and args[0] == "validate-project":
    opts = dict(zip(args[1::2], args[2::2]))
    root = Path(opts["--project-dir"])
    required = [root / "frame_series.settings", root / "tilt_series.settings", root / "processing"]
    if not all(p.exists() for p in required):
        print("missing outputs", file=sys.stderr)
        raise SystemExit(2)
    print("project accepted")
    raise SystemExit(0)
print("unsupported fake WarpTools command", args, file=sys.stderr)
raise SystemExit(2)
""")
    path.chmod(0o755)


def write_mrc(path: Path, *, nx=4, ny=4, nz=3, pixel=1.5) -> None:
    header = bytearray(1024)
    struct.pack_into("<4i", header, 0, nx, ny, nz, 2)
    struct.pack_into("<3i", header, 28, nx, ny, nz)
    struct.pack_into("<3f", header, 40, nx * pixel, ny * pixel, nz * pixel)
    header[208:212] = b"MAP "
    data = struct.pack("<" + "f" * (nx * ny * nz), *[float(i) for i in range(nx * ny * nz)])
    path.write_bytes(header + data)


def write_stack_fixture(data: Path, *, basename="TS1", nz=3, xf=True, aligned=True, rec=True) -> None:
    write_mrc(data / f"{basename}.st", nz=nz)
    if aligned:
        write_mrc(data / f"{basename}.ali", nz=nz)
    if rec:
        write_mrc(data / f"{basename}_rec.mrc", nx=6, ny=6, nz=5)
    (data / f"{basename}.tlt").write_text("\n".join(str(x) for x in range(nz)) + "\n")
    if xf:
        (data / f"{basename}.xf").write_text("".join("1 0 0 1 0 0\n" for _ in range(nz)))


class V6ConfigAndSourceTests(unittest.TestCase):
    def test_stack_only_setup_resolves_schema_and_jobs_with_fake_warptools(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            data = base / "data"; data.mkdir()
            write_stack_fixture(data)
            fake = base / "bin"; fake.mkdir()
            write_fake_warptools(fake / "WarpTools")
            out = base / "out"
            env = os.environ.copy()
            env["PATH"] = f"{fake}:{env.get('PATH', '')}"
            cp = subprocess.run([
                sys.executable, str(ROOT / "setup_warp_project.py"),
                "--data-dir", str(data),
                "--basename", "TS1",
                "--out-dir", str(out),
                "--source-mode", "auto",
                "--condition", "raw_xf_affine_fixed",
                "--alignment-backend", "legacy_affine",
            ], text=True, capture_output=True, check=False, env=env)
            self.assertEqual(cp.returncode, 0, cp.stderr + cp.stdout)
            cfg = load(out / "project_settings.toml")
            ts = cfg.tilt_series[0]
            self.assertEqual(cfg.schema_version, 6)
            self.assertEqual(ts.source.mode, "tilt_stack")
            self.assertEqual(ts.binning.extra_projection_binning, 1)
            self.assertEqual(ts.imod.raw_dimensions_xyz, [4, 4, 3])
            self.assertAlmostEqual(ts.imod.raw_pixel_size_A, 1.5)
            self.assertEqual(ts.imod.tilt_count, 3)
            self.assertTrue(ts.imod.xf.endswith("TS1.xf"))
            self.assertTrue(ts.imod.aligned_stack.endswith("TS1.ali"))
            self.assertTrue(ts.imod.source_reconstruction.endswith("TS1_rec.mrc"))
            self.assertTrue((out / "jobs" / "10_warp_ingest.sbatch").is_file())
            self.assertFalse((out / "jobs" / "20_initial_alignment_and_qc.sbatch").exists())
            self.assertFalse((out / "jobs" / "30_missalignment.sbatch").exists())
            self.assertIn("[next] submit Warp ingest:", cp.stdout)

    def test_invalid_mrc_and_missing_xf_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            (data / "TS1.st").write_bytes(b"fake")
            (data / "TS1.tlt").write_text("0\n")
            with self.assertRaises(SourceDiscoveryError):
                resolve_source(data, "TS1", "tilt_stack", condition="raw_xf_affine_fixed")
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            write_stack_fixture(data, xf=False)
            with self.assertRaises(SourceDiscoveryError):
                resolve_source(data, "TS1", "tilt_stack", condition="raw_xf_affine_fixed")

    def test_ambiguous_xf_rejected_and_tlt_does_not_imply_acquisition_order(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            write_mrc(data / "sample.st")
            (data / "sample.tlt").write_text("0\n1\n2\n")
            (data / "a.xf").write_text("1 0 0 1 0 0\n" * 3)
            (data / "b.xf").write_text("1 0 0 1 0 0\n" * 3)
            with self.assertRaises(SourceDiscoveryError):
                resolve_source(data, "sample", "tilt_stack", condition="raw_xf_affine_fixed")
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            write_stack_fixture(data)
            result = resolve_source(data, "TS1", "tilt_stack", condition="raw_xf_affine_fixed")
            self.assertTrue(result.capabilities.imod_alignment_available)
            self.assertFalse(result.capabilities.acquisition_order_known)
            self.assertFalse(result.capabilities.motion_refinement_in_m_available)

    def test_v5_config_requires_explicit_compatibility_loader(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "v5.toml"
            path.write_text(
                "[project]\nbasename = \"TS1\"\n"
                "[paths]\ndata_root = \"/data\"\noutput_dir = \"/out\"\n"
                "[input]\nraw_stack = \"/data/TS1.st\"\nfinal_tilt_file = \"/data/TS1.tlt\"\n"
                "[multiresolution]\nextra_projection_binning = 2\n"
            )
            with self.assertRaises(V6ConfigError):
                load(path)
            cfg, inferred = load_v5_compatibility_config(path)
            self.assertEqual(cfg.schema_version, 6)
            self.assertIn("source.mode=tilt_stack", inferred)


if __name__ == "__main__":
    unittest.main()
