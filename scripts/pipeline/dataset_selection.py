#!/usr/bin/env python3
"""User-facing discovery and selection of version 8 Warp datasets.

The project TOML remains authoritative for project identity and geometry. Dataset
availability and workflow state are read from the public ``warp_data`` tree,
its manifests, and ``project_status.json``. No scientific parameters are inferred
from directory modification times.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runlayout import RunLayout, dataset_id_from_config, parse_angpix_id


class DatasetSelectionError(RuntimeError):
    """Raised when a user-facing dataset request cannot be resolved safely."""


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    directory: Path
    manifest_path: Path
    status: str
    pixel_size_A: float | None
    origin: str
    source_dataset_id: str | None
    source_valid: bool
    source_problem: str | None
    reconstruction_complete: bool
    accepted: bool
    validation_level: str | None
    acceptance_path: Path
    selected: bool
    native: bool

    @property
    def ready_for_missalignment(self) -> bool:
        return self.source_valid and self.accepted and self.status in {"complete", "validated"}


def resolve_project_settings(value: Path) -> tuple[Path, dict[str, Any], Path]:
    """Resolve a project directory (preferred) or a TOML file (compatibility)."""
    supplied = Path(value).expanduser()
    if supplied.is_dir():
        settings = supplied / "project_settings.toml"
    else:
        settings = supplied
    settings = settings.resolve()
    if not settings.is_file():
        expected = supplied / "project_settings.toml" if supplied.is_dir() else supplied
        raise FileNotFoundError(
            f"project_settings.toml not found: {expected}. "
            "Pass the project directory with --directory."
        )
    with settings.open("rb") as handle:
        cfg = tomllib.load(handle)
    paths = cfg.get("paths", {}) or {}
    project_root = Path(paths.get("output_dir") or settings.parent).expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"project directory recorded in settings does not exist: {project_root}")
    return settings, cfg, project_root


def _json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _identity(cfg: dict[str, Any]) -> tuple[str, str, str]:
    project = cfg.get("project", {}) or {}
    conversion = cfg.get("conversion", {}) or {}
    missalignment = cfg.get("missalignment", {}) or {}
    conditions = conversion.get("initial_conditions") or ["ali_identity"]
    if not isinstance(conditions, list):
        conditions = [conditions]
    if len(conditions) != 1:
        raise DatasetSelectionError("version 8 requires exactly one conversion condition per project")
    return (
        str(project.get("basename") or project.get("name") or "series"),
        str(conditions[0]),
        str(missalignment.get("refinement_mode") or "standard"),
    )


def _source_contract(layout: RunLayout) -> tuple[bool, str | None]:
    source = layout.training_dir.resolve()
    if not source.is_dir():
        return False, f"Warp project missing: {source}"
    marker = source / "_converted.marker"
    validation = source / "conversion_validation.json"
    xmls = [path for path in source.glob("*.xml") if path.is_file() and path.stat().st_size > 0]
    stacks = list((source / "tiltstack").glob("*/*.st")) if (source / "tiltstack").is_dir() else []
    if not marker.is_file():
        return False, f"conversion marker missing: {marker}"
    if not validation.is_file() or validation.stat().st_size <= 0:
        return False, f"conversion validation missing: {validation}"
    if len(xmls) != 1:
        return False, f"expected one root Warp XML, found {len(xmls)}"
    if not stacks:
        return False, "no tiltstack/*/*.st found"
    return True, None


def _reconstruction_complete(layout: RunLayout) -> bool:
    latest = layout.attempts_dir / "reconstruction" / layout.dataset_id / "warp_dataset" / "latest_success"
    if not latest.exists():
        return False
    result = latest.resolve() / "result_manifest.json"
    if not result.is_file():
        return False
    data = _json(result)
    return data.get("status") == "completed" and bool(data.get("reconstruction"))


def discover_datasets(settings: Path, cfg: dict[str, Any], project_root: Path) -> tuple[list[DatasetRecord], str | None]:
    """Return all public Warp datasets and the recorded default dataset ID."""
    basename, condition, mode = _identity(cfg)
    project_status = _json(project_root / "project_status.json")
    status_records = project_status.get("datasets") or {}
    if not isinstance(status_records, dict):
        status_records = {}
    native_id = str(project_status.get("native_dataset_id") or dataset_id_from_config(cfg))
    # The resolved TOML records the native dataset used during setup. The user's
    # current working choice is mutable workflow state and therefore belongs only
    # in project_status.json. This lets a completed preprocessing dataset become
    # the default without rewriting the authoritative TOML.
    recorded_selected = project_status.get("selected_dataset_id")
    recorded_selected = str(recorded_selected) if recorded_selected else None

    ids: set[str] = set(str(key) for key in status_records)
    warp_root = project_root / "warp_data"
    if warp_root.is_dir():
        ids.update(path.name for path in warp_root.iterdir() if path.is_dir())

    records: list[DatasetRecord] = []
    for dataset_id in sorted(ids, key=_dataset_sort_key):
        directory = warp_root / dataset_id
        manifest_path = directory / "manifest.json"
        manifest = _json(manifest_path)
        status_info = status_records.get(dataset_id) or {}
        if not isinstance(status_info, dict):
            status_info = {}
        status = str(manifest.get("status") or status_info.get("status") or "unknown")
        pixel = manifest.get("pixel_size_A") or status_info.get("pixel_size_A")
        try:
            pixel_value = float(pixel) if pixel not in (None, "") else parse_angpix_id(dataset_id)
        except (TypeError, ValueError):
            pixel_value = None
        preprocessing = manifest.get("preprocessing")
        source_id = None
        if isinstance(preprocessing, dict):
            source_id = preprocessing.get("source_dataset_id")
        source_id = source_id or manifest.get("source_dataset_id")
        origin = "preprocessed" if preprocessing or manifest.get("source_artifact_id") else "imported"
        layout = RunLayout.from_settings(
            out_dir=project_root,
            basename=basename,
            condition=condition,
            refinement_mode=mode,
            dataset_id=dataset_id,
        )
        source_valid, source_problem = _source_contract(layout)
        acceptance = _json(layout.acceptance_path)
        validation_level = acceptance.get("validation_level")
        if not validation_level and acceptance.get("status") == "accepted":
            validation_level = "visual"
        accepted = acceptance.get("status") in {"accepted", "validated"}
        records.append(DatasetRecord(
            dataset_id=dataset_id,
            directory=directory,
            manifest_path=manifest_path,
            status=status,
            pixel_size_A=pixel_value,
            origin=origin,
            source_dataset_id=str(source_id) if source_id else None,
            source_valid=source_valid,
            source_problem=source_problem,
            reconstruction_complete=_reconstruction_complete(layout),
            accepted=accepted,
            validation_level=str(validation_level) if validation_level else None,
            acceptance_path=layout.acceptance_path,
            selected=dataset_id == recorded_selected,
            native=dataset_id == native_id,
        ))
    return records, recorded_selected


def _dataset_sort_key(value: str) -> tuple[int, float, str]:
    try:
        return (0, parse_angpix_id(value), value)
    except ValueError:
        return (1, float("inf"), value)


def _record_by_id(records: list[DatasetRecord], dataset_id: str) -> DatasetRecord | None:
    return next((record for record in records if record.dataset_id == dataset_id), None)


def dataset_id_from_user_value(value: str, project_root: Path) -> str:
    """Accept a dataset ID, dataset directory, manifest path, or Warp-project path."""
    text = str(value).strip()
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        direct = (Path.cwd() / candidate).resolve()
        project_relative = (project_root / candidate).resolve()
        candidate = direct if direct.exists() else project_relative
    else:
        candidate = candidate.resolve()

    if candidate.exists():
        if candidate.is_file():
            if candidate.name == "manifest.json":
                data = _json(candidate)
                dataset_id = data.get("dataset_id")
                if dataset_id:
                    return str(dataset_id)
            candidate = candidate.parent
        current = candidate
        while current != current.parent:
            manifest = current / "manifest.json"
            if manifest.is_file():
                dataset_id = _json(manifest).get("dataset_id")
                if dataset_id:
                    return str(dataset_id)
            if current.parent.name == "warp_data":
                return current.name
            current = current.parent
    # Plain ID path: validate syntax only after path resolution has been attempted.
    parse_angpix_id(text)
    return text


def select_dataset(
    records: list[DatasetRecord],
    *,
    project_root: Path,
    requested: str | None = None,
    recorded_selected: str | None = None,
) -> DatasetRecord:
    """Select a dataset without unsafe guesses when several derived datasets exist."""
    if not records:
        raise DatasetSelectionError(
            f"no Warp datasets were found under {project_root / 'warp_data'}"
        )
    if requested:
        try:
            dataset_id = dataset_id_from_user_value(requested, project_root)
        except ValueError as exc:
            raise DatasetSelectionError(str(exc)) from exc
        record = _record_by_id(records, dataset_id)
        if record is None:
            raise DatasetSelectionError(f"dataset {dataset_id!r} is not registered in this project")
        return record

    if recorded_selected:
        record = _record_by_id(records, recorded_selected)
        if record is not None and record.status in {"complete", "validated"}:
            return record

    complete = [record for record in records if record.status in {"complete", "validated"}]
    derived = [record for record in complete if record.origin == "preprocessed"]
    if len(derived) == 1:
        return derived[0]
    if len(derived) > 1:
        ids = ", ".join(record.dataset_id for record in derived)
        raise DatasetSelectionError(
            "more than one processed dataset is available and no default is recorded: "
            f"{ids}. Select one with --dataset."
        )
    native = [record for record in complete if record.native]
    if len(native) == 1:
        return native[0]
    if len(complete) == 1:
        return complete[0]
    if not complete:
        raise DatasetSelectionError("no complete Warp dataset is available")
    ids = ", ".join(record.dataset_id for record in complete)
    raise DatasetSelectionError(
        f"more than one complete dataset is available: {ids}. Select one with --dataset."
    )


def format_dataset_table(records: list[DatasetRecord], default_id: str | None = None) -> str:
    if not records:
        return "Available datasets: none"
    lines = [
        "Available datasets:",
        "  default  dataset       origin        status       reconstruction  validation  source",
    ]
    for record in records:
        marker = "*" if record.dataset_id == default_id or (default_id is None and record.selected) else ""
        reconstruction = "yes" if record.reconstruction_complete else "no"
        validation = record.validation_level or ("legacy" if record.accepted else "no")
        source = record.source_dataset_id or "-"
        lines.append(
            f"  {marker:<7}  {record.dataset_id:<12}  {record.origin:<12}  "
            f"{record.status:<11}  {reconstruction:<14}  {validation:<10}  {source}"
        )
        if not record.source_valid:
            lines.append(f"           problem: {record.source_problem}")
    return "\n".join(lines)


def record_selected_dataset(project_root: Path, dataset_id: str) -> None:
    """Record an explicit/default choice without changing scientific parameters."""
    status_path = project_root / "project_status.json"
    status = _json(status_path)
    status.setdefault("schema_version", 1)
    status.setdefault("layout_version", 8)
    status["selected_dataset_id"] = dataset_id
    temporary = status_path.with_suffix(status_path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(status_path)
