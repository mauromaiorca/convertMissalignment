#!/usr/bin/env python3
"""Prepare isolated Warp snapshots for MissAlignment without Slurm or CUDA."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from .dataset_selection import (
    discover_datasets,
    record_selected_dataset,
    resolve_project_settings,
    select_dataset,
)
from .runlayout import RunLayout, dataset_id_from_config


def load_settings(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def layout_for(cfg: dict[str, Any], dataset_id: str | None = None) -> RunLayout:
    project = cfg.get("project", {}) or {}
    conversion = cfg.get("conversion", {}) or {}
    missalignment = cfg.get("missalignment", {}) or {}
    paths = cfg.get("paths", {}) or {}
    conditions = conversion.get("initial_conditions", ["ali_identity"])
    condition = conditions[0] if isinstance(conditions, list) else conditions
    return RunLayout.from_settings(
        out_dir=Path(paths.get("output_dir") or "."),
        basename=str(project.get("basename") or project.get("name") or "series"),
        condition=str(condition),
        refinement_mode=str(missalignment.get("refinement_mode") or "standard"),
        dataset_id=dataset_id or dataset_id_from_config(cfg),
    )


def _validate_source(layout: RunLayout) -> dict[str, Any]:
    source = layout.training_dir.resolve()
    if not source.is_dir():
        raise RuntimeError(f"imported Warp project missing: {source}")
    marker = source / "_converted.marker"
    validation = source / "conversion_validation.json"
    xmls = sorted(path for path in source.glob("*.xml") if path.is_file() and path.stat().st_size > 0)
    stacks = sorted((source / "tiltstack").glob("*/*.st")) if (source / "tiltstack").is_dir() else []
    if not marker.is_file():
        raise RuntimeError(f"missing conversion marker: {marker}")
    if not validation.is_file() or validation.stat().st_size <= 0:
        raise RuntimeError(f"missing conversion validation: {validation}")
    if len(xmls) != 1:
        raise RuntimeError(f"expected exactly one root Warp XML in {source}; found {len(xmls)}")
    if not stacks:
        raise RuntimeError(f"no tiltstack/*/*.st found in {source}")
    return {"source": str(source), "xml": str(xmls[0]), "tiltstacks": [str(p) for p in stacks]}


def prepare(
    settings_path: Path,
    *,
    dataset_id: str | None = None,
    force: bool = False,
    allow_without_acceptance: bool = False,
) -> dict[str, Any]:
    resolved_settings, cfg, project_root = resolve_project_settings(Path(settings_path))
    records, recorded_selected = discover_datasets(resolved_settings, cfg, project_root)
    selected = select_dataset(
        records,
        project_root=project_root,
        requested=dataset_id,
        recorded_selected=recorded_selected,
    )
    layout = layout_for(cfg, selected.dataset_id).create()
    source_info = _validate_source(layout)

    if layout.condition == "raw_xf_affine_fixed" and not allow_without_acceptance:
        if not layout.acceptance_path.is_file() or layout.acceptance_path.stat().st_size <= 0:
            raise RuntimeError(
                "the affine Warp dataset has no successful reconstruction validation: "
                f"{layout.acceptance_path}"
            )
        try:
            validation = json.loads(layout.acceptance_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid reconstruction validation record: {layout.acceptance_path}"
            ) from exc
        if validation.get("status") not in {"accepted", "validated"}:
            raise RuntimeError(
                "the affine Warp reconstruction validation is not usable: "
                f"status={validation.get('status')!r}"
            )

    from clone_warp_projects import prepare_snapshots

    result = prepare_snapshots(
        Path(source_info["source"]),
        layout.pre_missalign_dir,
        layout.smoke_warp_dir,
        layout.full_warp_dir,
        layout.manifest("warp_snapshot_manifest.json"),
        force=force,
    )

    status_path = layout.run_dir / "project_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {"schema_version": 1}
    status["missalignment_input"] = {
        "status": "prepared",
        "dataset_id": layout.dataset_id,
        "manifest": str(layout.manifest("warp_snapshot_manifest.json")),
        "execution": "synchronous_local",
    }
    tmp = status_path.with_suffix(status_path.suffix + ".tmp")
    tmp.write_text(json.dumps(status, indent=2) + "\n")
    tmp.replace(status_path)
    record_selected_dataset(project_root, layout.dataset_id)
    return {"dataset_id": layout.dataset_id, "layout": layout.to_dict(), **result}
