from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.project_workflow import _synchronous_warp_import
from pipeline.runlayout import RunLayout

REAL_SUBPROCESS_RUN = subprocess.run


def _layout(root: Path) -> RunLayout:
    return RunLayout.from_settings(
        out_dir=root,
        basename="TS1",
        condition="raw_xf_affine_fixed",
        refinement_mode="standard",
        dataset_id="5.45Apx",
    ).create()


def _publish_fake_import(layout: RunLayout) -> None:
    project = layout.training_dir.resolve()
    project.mkdir(parents=True, exist_ok=True)
    (project / "_converted.marker").write_text("ok\n")
    (project / "conversion_validation.json").write_text("{}\n")
    (project / "TS_TS1_raw_xf_affine_fixed.xml").write_text("<TiltSeries />\n")
    layout.dataset_manifest.write_text(json.dumps({
        "dataset_id": layout.dataset_id,
        "status": "complete",
    }) + "\n")


def test_sync_import_executes_conversion_and_keeps_recovery_batch_path():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "project"
        layout = _layout(root)
        staging = layout.manifest("warp_staging_manifest.json")
        staging.write_text("{}\n")
        cfg = {
            "cluster": {
                "warp_module": "warp/2.0.39",
                "module_init_script": "/usr/share/Modules/init/bash",
                "environment": str(Path(sys.executable).resolve().parent.parent),
            }
        }

        def fake_run(command, **kwargs):
            assert command[:2] == ["bash", "-lc"]
            shell = command[2]
            syntax = REAL_SUBPROCESS_RUN(
                ["bash", "-n", "-c", shell], capture_output=True, text=True
            )
            assert syntax.returncode == 0, syntax.stderr
            assert "module load warp/2.0.39" in shell
            assert "run_warp_conversion.py" in shell
            assert f"--training-dir {layout.training_dir}" in shell
            _publish_fake_import(layout)
            return subprocess.CompletedProcess(command, 0)

        with patch("pipeline.project_workflow.subprocess.run", side_effect=fake_run) as mocked:
            result = _synchronous_warp_import(layout, cfg, staging)

        assert mocked.call_count == 1
        assert result["execution"] == "synchronous"
        assert result["status"] == "complete"
        assert Path(result["xml"]).is_file()


def test_sync_import_reuses_valid_dataset_without_subprocess():
    with tempfile.TemporaryDirectory() as td:
        layout = _layout(Path(td) / "project")
        staging = layout.manifest("warp_staging_manifest.json")
        staging.write_text("{}\n")
        _publish_fake_import(layout)

        with patch("pipeline.project_workflow.subprocess.run") as mocked:
            result = _synchronous_warp_import(layout, {"cluster": {}}, staging)

        mocked.assert_not_called()
        assert result["execution"] == "reused"


def test_force_sync_import_passes_force_to_converter():
    with tempfile.TemporaryDirectory() as td:
        layout = _layout(Path(td) / "project")
        staging = layout.manifest("warp_staging_manifest.json")
        staging.write_text("{}\n")
        _publish_fake_import(layout)

        def fake_run(command, **kwargs):
            assert command[0:2] == ["bash", "-lc"]
            assert "--force" in command[2]
            _publish_fake_import(layout)
            return subprocess.CompletedProcess(command, 0)

        with patch("pipeline.project_workflow.subprocess.run", side_effect=fake_run) as mocked:
            result = _synchronous_warp_import(
                layout,
                {"cluster": {"environment": str(Path(sys.executable).resolve().parent.parent)}},
                staging,
                force=True,
            )

        assert mocked.call_count == 1
        assert result["execution"] == "synchronous"
