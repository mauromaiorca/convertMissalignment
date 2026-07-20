from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import mrcfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREPROCESS = ROOT / "warp_preprocess.py"
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline.runlayout import RunLayout


def _settings(root: Path) -> Path:
    path = root / "project_settings.toml"
    path.write_text(f'''[project]
basename="x"
layout_version=8
[paths]
data_root="{root}"
output_dir="{root}"
[geometry]
raw_shape_xyz=[120,90,5]
raw_pixel_size_A=2.0
aligned_shape_xyz=[120,90,5]
aligned_pixel_size_A=2.0
target_volume_shape_xyz=[120,20,90]
target_pixel_size_A=2.0
tilt_axis_angle_deg=84.5
[conversion]
initial_conditions=["raw_xf_affine_fixed"]
[conversion.condition_modes]
raw_xf_affine_fixed="quarter-turn-affine"
[datasets]
native_id="2Apx"
selected_id="2Apx"
[missalignment]
refinement_mode="standard"
result_backend="warp_xml"
executable="miss-alignment"
[ctf]
mode="off"
[multiresolution]
extra_projection_binning=1
[cluster]
profile="maxwell"
partition="vds"
cpus=4
time="01:00:00"
generate_slurm=true
[reconstruction]
enabled=false
[provenance]
resolved=true
''')
    return path


def _source_dataset(root: Path) -> RunLayout:
    layout = RunLayout.from_settings(
        out_dir=root, basename="x", condition="raw_xf_affine_fixed",
        refinement_mode="standard", dataset_id="2Apx",
    ).create()
    project = layout.training_dir.resolve()
    stack = project / "tiltstack" / "TS_x" / "TS_x.st"
    stack.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(stack, overwrite=True) as handle:
        handle.set_data(np.arange(5 * 90 * 120, dtype=np.float32).reshape(5, 90, 120))
        handle.voxel_size = 2.0
    (project / "TS_x.xml").write_text('<TiltSeries VolumeDimensionsAngstrom="240,180,40" ImageDimensionsAngstrom="240,180"/>\n')
    (project / "_converted.marker").write_text("ok\n")
    (project / "conversion_validation.json").write_text(json.dumps({"volume_frame_contract_version": 2, "warp_volume_shape_xyz": [120,90,20]}))
    layout.dataset_manifest.write_text(json.dumps({
        "artifact_id": "warp-native", "dataset_id": "2Apx", "pixel_size_A": 2.0,
        "stack_shape_zyx": [5, 90, 120], "status": "complete",
    }))
    return layout


def _fake_newstack(directory: Path) -> Path:
    script = directory / "newstack"
    script.write_text(f'''#!{sys.executable}
import sys
from pathlib import Path
import mrcfile
args=sys.argv[1:]
def val(flag): return args[args.index(flag)+1]
src=Path(val("-input")); dst=Path(val("-output")); factor=float(val("-shrink"))
with mrcfile.open(src, permissive=True) as h:
    data=h.data.copy(); px=float(h.voxel_size.x)
f=int(round(factor))
dst.parent.mkdir(parents=True, exist_ok=True)
with mrcfile.new(dst, overwrite=True) as h:
    h.set_data(data[:, ::f, ::f].astype("float32")); h.voxel_size=px*factor
''')
    script.chmod(0o755)
    return script


def test_bin3_plans_and_materialises_complete_dataset():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "project"
        root.mkdir()
        settings = _settings(root)
        _source_dataset(root)
        plan = subprocess.run(
            [sys.executable, str(PREPROCESS), "--project", str(root), "--bin", "3"],
            text=True, capture_output=True,
        )
        assert plan.returncode == 0, plan.stdout + plan.stderr
        target = root / "warp_data" / "6Apx"
        assert (target / "manifest.json").is_file()
        assert (root / "batches" / "warp_data" / "6Apx" / "preprocess.sbatch").is_file()
        bindir = Path(td) / "bin"; bindir.mkdir(); _fake_newstack(bindir)
        env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ.get("PATH", ""))
        run = subprocess.run(
            [sys.executable, str(PREPROCESS), "run", "--project", str(settings),
             "--source", "2Apx", "--target", "6Apx", "--factor", "3"],
            text=True, capture_output=True, env=env,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        manifest = json.loads((target / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert manifest["pixel_size_A"] == 6.0
        assert manifest["stack_shape_zyx"] == [5, 30, 40]
        assert manifest["source_artifact_id"] == "warp-native"
        assert manifest["preprocessing"]["factor"] == 3.0
        registry = json.loads((root / "provenance" / "artifact_registry.json").read_text())
        assert isinstance(registry["artifacts"], dict)
        assert registry["artifacts"]["6Apx"]["status"] == "complete"
        project_status = json.loads((root / "project_status.json").read_text())
        assert project_status["datasets"]["6Apx"]["status"] == "complete"
        assert project_status["selected_dataset_id"] == "6Apx"
