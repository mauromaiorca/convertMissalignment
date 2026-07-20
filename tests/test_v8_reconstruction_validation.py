from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pipeline.reconstruction_validation import record_reconstruction_validation
from pipeline.runlayout import RunLayout


def _layout(tmp_path: Path) -> RunLayout:
    layout = RunLayout.from_settings(
        out_dir=tmp_path / "project",
        basename="TS1",
        condition="raw_xf_affine_fixed",
        refinement_mode="standard",
        dataset_id="5.452Apx",
    ).create()
    layout.dataset_manifest.write_text(json.dumps({
        "dataset_id": layout.dataset_id,
        "status": "complete",
        "pixel_size_A": 5.452,
    }) + "\n")
    (layout.run_dir / "project_status.json").write_text(json.dumps({
        "schema_version": 1,
        "layout_version": 8,
        "datasets": {layout.dataset_id: {"status": "complete"}},
    }) + "\n")
    attempt = (layout.attempts_dir / "reconstruction" / layout.dataset_id /
               "warp_dataset" / "attempt_1")
    attempt.mkdir(parents=True)
    volume = attempt / "rec.mrc"
    volume.write_bytes(b"mrc")
    result = attempt / "result_manifest.json"
    result.write_text(json.dumps({
        "status": "completed",
        "reconstruction": str(volume),
    }) + "\n")
    (attempt.parent / "latest_success").symlink_to(attempt.name, target_is_directory=True)
    return layout


def test_technical_validation_is_automatic_and_does_not_claim_visual_review(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    record = record_reconstruction_validation(
        layout,
        level="technical",
        note="automatic test",
    )
    assert record["status"] == "validated"
    assert record["validation_level"] == "technical"
    assert record["visual_inspection"] is False
    assert json.loads(layout.dataset_manifest.read_text())["status"] == "validated"


def test_visual_review_upgrades_but_preserves_technical_record(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    technical = record_reconstruction_validation(
        layout,
        level="technical",
        note="automatic test",
    )
    visual = record_reconstruction_validation(
        layout,
        level="visual",
        note="map inspected",
        actor="scientist",
    )
    assert visual["status"] == "accepted"
    assert visual["validation_level"] == "visual"
    assert visual["visual_inspection"] is True
    assert visual["previous_validation"]["validation_level"] == technical["validation_level"]
