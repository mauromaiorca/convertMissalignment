#!/usr/bin/env python3
"""Record technical or visual validation of a completed Warp reconstruction.

Version 8 distinguishes two validation levels:

* ``technical``: written automatically after WarpTools exits successfully and the
  reconstruction and manifests pass filesystem checks. This is sufficient to
  prepare MissAlignment input.
* ``visual``: an optional human review that upgrades the same record without
  changing the dataset or reconstruction.

The distinction is retained in provenance so automatic workflow progression does
not claim that a scientist inspected the map.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset_selection import record_selected_dataset
from .runlayout import RunLayout
from .warptools_reconstruction import atomic_json


class ReconstructionValidationError(RuntimeError):
    """Raised when a reconstruction cannot be validated safely."""


def _json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReconstructionValidationError(f"expected a JSON object in {path}")
    return data


def latest_completed_reconstruction(layout: RunLayout) -> tuple[Path, Path, dict[str, Any]]:
    """Return ``(attempt, result_manifest, result)`` for the latest successful run."""
    root = layout.attempts_dir / "reconstruction" / layout.dataset_id / "warp_dataset"
    latest = root / "latest_success"
    if not latest.exists():
        raise ReconstructionValidationError(
            f"no successful Warp reconstruction for dataset {layout.dataset_id}"
        )
    attempt = latest.resolve()
    result_path = attempt / "result_manifest.json"
    if not result_path.is_file() or result_path.stat().st_size <= 0:
        raise ReconstructionValidationError(f"result manifest missing: {result_path}")
    result = _json(result_path)
    if result.get("status") != "completed":
        raise ReconstructionValidationError(
            f"Warp reconstruction is not completed: status={result.get('status')!r}"
        )
    reconstruction_value = result.get("reconstruction")
    if not reconstruction_value:
        raise ReconstructionValidationError("completed result has no reconstruction path")
    reconstruction = Path(str(reconstruction_value))
    if not reconstruction.is_file() or reconstruction.stat().st_size <= 0:
        raise ReconstructionValidationError(
            f"completed reconstruction is missing or empty: {reconstruction}"
        )
    return attempt, result_path, result


def record_reconstruction_validation(
    layout: RunLayout,
    *,
    level: str,
    note: str,
    actor: str | None = None,
) -> dict[str, Any]:
    """Write validation provenance and update public dataset state.

    ``level`` must be ``technical`` or ``visual``. Technical validation is
    automatic and records no claim of human inspection. Visual validation is an
    explicit review and supersedes, but preserves, any earlier technical record.
    """
    if level not in {"technical", "visual"}:
        raise ValueError(f"unsupported validation level: {level}")

    attempt, result_path, result = latest_completed_reconstruction(layout)
    previous: dict[str, Any] | None = None
    if layout.acceptance_path.is_file() and layout.acceptance_path.stat().st_size > 0:
        try:
            previous = _json(layout.acceptance_path)
        except ReconstructionValidationError:
            previous = None

    now = datetime.now(timezone.utc).isoformat()
    if level == "technical":
        validation = {
            "schema_version": 2,
            "status": "validated",
            "validation_level": "technical",
            "validation_mode": "automatic_after_successful_warp_reconstruction",
            "validated_at": now,
            "validated_by": "pipeline",
            "visual_inspection": False,
            "note": note,
            "dataset_id": layout.dataset_id,
            "attempt": str(attempt),
            "reconstruction": result["reconstruction"],
            "result_manifest": str(result_path),
            "checks": {
                "warptools_completed": True,
                "result_manifest_completed": True,
                "reconstruction_exists": True,
                "reconstruction_nonempty": True,
            },
        }
        # Never downgrade a manual visual review when a reconstruction is
        # re-indexed or a legacy project is revisited.
        if previous and previous.get("validation_level") == "visual":
            return previous
    else:
        validation = {
            "schema_version": 2,
            "status": "accepted",
            "validation_level": "visual",
            "validation_mode": "manual_visual_review",
            "validated_at": now,
            "validated_by": actor
            or os.environ.get("USER")
            or os.environ.get("LOGNAME")
            or "unknown",
            "visual_inspection": True,
            "note": note,
            "dataset_id": layout.dataset_id,
            "attempt": str(attempt),
            "reconstruction": result["reconstruction"],
            "result_manifest": str(result_path),
        }
        if previous:
            validation["previous_validation"] = previous

    atomic_json(layout.acceptance_path, validation)

    result["acceptance_state"] = (
        "visually_accepted" if level == "visual" else "technically_validated"
    )
    result["acceptance"] = validation
    atomic_json(result_path, result)

    public_manifest_value = result.get("public_manifest")
    if public_manifest_value:
        public_manifest_path = Path(str(public_manifest_value))
        if public_manifest_path.is_file():
            public_manifest = _json(public_manifest_path)
            public_manifest["acceptance_state"] = result["acceptance_state"]
            public_manifest["acceptance"] = validation
            atomic_json(public_manifest_path, public_manifest)

    if layout.dataset_manifest.is_file():
        dataset_manifest = _json(layout.dataset_manifest)
        dataset_manifest["status"] = "validated"
        dataset_manifest["validation"] = {
            "record": str(layout.acceptance_path),
            "level": level,
            "visual_inspection": level == "visual",
            "validated_reconstruction": result["reconstruction"],
        }
        atomic_json(layout.dataset_manifest, dataset_manifest)

    status_path = layout.run_dir / "project_status.json"
    if status_path.is_file() and status_path.stat().st_size > 0:
        try:
            project_status = _json(status_path)
        except ReconstructionValidationError:
            project_status = {"schema_version": 1, "layout_version": 8}
    else:
        project_status = {"schema_version": 1, "layout_version": 8}
    datasets = dict(project_status.get("datasets") or {})
    current = dict(datasets.get(layout.dataset_id) or {})
    current.update(
        {
            "status": "validated",
            "manifest": str(layout.dataset_manifest),
            "reconstruction_validation": {
                "level": level,
                "record": str(layout.acceptance_path),
            },
        }
    )
    datasets[layout.dataset_id] = current
    project_status["datasets"] = datasets
    atomic_json(status_path, project_status)
    record_selected_dataset(layout.run_dir, layout.dataset_id)
    return validation
