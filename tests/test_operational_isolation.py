from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_clone_warp_projects_separates_writable_xml(tmp_path: Path):
    source = tmp_path / "base"
    (source / "tiltstack" / "TS").mkdir(parents=True)
    (source / "TS.xml").write_text("<TiltSeries />\n")
    (source / "tiltstack" / "TS" / "TS.st").write_bytes(b"stack")
    manifest = tmp_path / "manifest.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "clone_warp_projects.py"
    subprocess.run([
        sys.executable, str(script),
        "--source", str(source),
        "--pre", str(tmp_path / "pre"),
        "--smoke", str(tmp_path / "smoke"),
        "--full", str(tmp_path / "full"),
        "--manifest", str(manifest),
    ], check=True, capture_output=True, text=True)
    pre_xml = tmp_path / "pre" / "TS.xml"
    smoke_xml = tmp_path / "smoke" / "TS.xml"
    full_xml = tmp_path / "full" / "TS.xml"
    assert not pre_xml.is_symlink()
    assert not smoke_xml.is_symlink()
    assert not full_xml.is_symlink()
    smoke_xml.write_text("<TiltSeries smoke='1' />\n")
    assert pre_xml.read_text() == "<TiltSeries />\n"
    assert full_xml.read_text() == "<TiltSeries />\n"
    data = json.loads(manifest.read_text())
    assert data["policy"]["smoke_and_full_share_writable_metadata"] is False


def test_phase2_job_targets_isolated_snapshots(tmp_path: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from pipeline.jobs import generate_jobs
    from pipeline.project_config import ClusterConfig
    from pipeline.runlayout import RunLayout

    layout = RunLayout.from_settings(
        out_dir=tmp_path,
        basename="TS",
        condition="raw_xf_affine_fixed",
        refinement_mode="standard",
    ).create()
    written = generate_jobs(
        layout,
        profile="maxwell",
        ma_command="miss-alignment --config-file full.yaml",
        smoke_command="miss-alignment --config-file smoke.yaml",
        run_script="unused-run_missalignment.sh",
        settings_path="settings.toml",
        cluster=ClusterConfig(profile="maxwell", environment="/env", partition="vds", constraint="V100", gres=""),
        warp_staging_manifest=str(layout.manifest("warp_staging_manifest.json")),
    )
    text = Path(written["phase2.sbatch"]).read_text()
    assert "warp/pre_missalign" in text
    assert "warp/missalign_smoke" in text
    assert "warp/missalign_full" in text
    assert 'export MISSALIGN_FINAL_XML="${FINAL_XMLS[0]}"' in text
    subprocess.run(["bash", "-n", written["phase2.sbatch"]], check=True)
