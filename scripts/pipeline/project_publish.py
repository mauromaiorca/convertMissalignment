#!/usr/bin/env python3
"""Publish internal tool workspaces into the concise v8 project tree."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .runlayout import RunLayout, format_angpix


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def hash_file(path: Path, *, full_limit: int = 64 << 20) -> tuple[str, str]:
    size = path.stat().st_size
    h = hashlib.sha256()
    if size <= full_limit:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest(), "full_sha256"
    with path.open("rb") as handle:
        h.update(handle.read(8 << 20))
        handle.seek(-(8 << 20), 2)
        h.update(handle.read(8 << 20))
    h.update(str(size).encode())
    return h.hexdigest(), "partial_sha256_head8M_tail8M_size"


def _replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    relative = os.path.relpath(target, start=link.parent)
    link.symlink_to(relative, target_is_directory=target.is_dir())


def _copy_small_or_link(source: Path, destination: Path, *, copy_limit: int = 8 << 20) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    if source.stat().st_size <= copy_limit:
        shutil.copy2(source, destination)
        return "copy"
    _replace_symlink(destination, source)
    return "symlink"


def publish_imod_import(layout: RunLayout, inventory: Any) -> Path:
    """Create imported_data/imod without duplicating large image stacks."""
    layout.create()
    data_fields = (
        "raw_stack",
        "aligned_stack",
    )
    configuration_fields = (
        "final_xf",
        "tilt_file",
        "raw_tilt_file",
        "xtilt_file",
        "tltxf_file",
        "defocus_file",
        "mdoc_file",
        "newst_com",
        "tilt_com",
        "ctf_com",
    )
    records: dict[str, Any] = {}
    for category, fields in (("data", data_fields), ("configuration", configuration_fields)):
        for field in fields:
            value = getattr(inventory, field, None)
            if not value:
                continue
            source = Path(value).resolve()
            if not source.is_file():
                continue
            destination = layout.imported_imod_dir / category / source.name
            policy = _copy_small_or_link(source, destination)
            digest, digest_mode = hash_file(source)
            records[field] = {
                "source": str(source),
                "published": str(destination),
                "policy": policy,
                "size": source.stat().st_size,
                "sha256": digest,
                "hash_mode": digest_mode,
            }
    source_reconstruction = getattr(inventory, "source_reconstruction", None)
    if source_reconstruction and Path(source_reconstruction).is_file():
        source = Path(source_reconstruction).resolve()
        destination = layout.imported_imod_dir / "reconstructions" / "native" / source.name
        policy = _copy_small_or_link(source, destination)
        digest, digest_mode = hash_file(source)
        records["source_reconstruction"] = {
            "source": str(source),
            "published": str(destination),
            "policy": policy,
            "size": source.stat().st_size,
            "sha256": digest,
            "hash_mode": digest_mode,
        }

    manifest = {
        "schema_version": 1,
        "artifact_type": "imported_imod_project",
        "artifact_id": f"imod-import-{hashlib.sha256(str(layout.run_dir).encode()).hexdigest()[:12]}",
        "basename": layout.basename,
        "condition": layout.condition,
        "immutable": True,
        "files": records,
        "allowed_uses": ["provenance", "reconstruction", "Warp import"],
    }
    target = layout.imported_imod_dir / "manifest.json"
    atomic_json(target, manifest)
    return target


def _mrc_stack_info(stack: Path) -> dict[str, Any]:
    try:
        import mrcfile
    except ImportError:
        return {}
    with mrcfile.mmap(stack, mode="r", permissive=True) as handle:
        if handle.data.ndim != 3:
            return {}
        ntilts, ny, nx = map(int, handle.data.shape)
        pixel = float(handle.voxel_size.x)
    return {
        "shape_zyx": [ntilts, ny, nx],
        "pixel_size_A": pixel,
    }




def _update_project_dataset_records(layout: RunLayout, manifest: dict[str, Any]) -> None:
    """Publish dataset state at project level without making it a second config source."""
    registry_path = layout.provenance_dir / "artifact_registry.json"
    registry: dict[str, Any] = {}
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text())
    raw_artifacts = registry.get("artifacts") or {}
    if isinstance(raw_artifacts, list):
        artifacts = {
            str(item.get("dataset_id") or item.get("artifact_id") or f"artifact_{index}"): item
            for index, item in enumerate(raw_artifacts)
            if isinstance(item, dict)
        }
    else:
        artifacts = dict(raw_artifacts)
    artifacts[layout.dataset_id] = {
        "artifact_id": manifest.get("artifact_id"),
        "artifact_type": manifest.get("artifact_type"),
        "manifest": str(layout.dataset_manifest),
        "pixel_size_A": manifest.get("pixel_size_A"),
        "source_artifact_id": manifest.get("source_artifact_id"),
        "status": manifest.get("status", "complete"),
    }
    registry.update({"schema_version": 1, "artifacts": artifacts})
    atomic_json(registry_path, registry)

    status_path = layout.run_dir / "project_status.json"
    status: dict[str, Any] = {}
    if status_path.is_file():
        status = json.loads(status_path.read_text())
    datasets = dict(status.get("datasets") or {})
    datasets[layout.dataset_id] = {
        "status": manifest.get("status", "complete"),
        "manifest": str(layout.dataset_manifest),
        "pixel_size_A": manifest.get("pixel_size_A"),
        "artifact_id": manifest.get("artifact_id"),
    }
    status.update({
        "schema_version": 1,
        "layout_version": 8,
        "native_dataset_id": status.get("native_dataset_id") or layout.dataset_id,
        "selected_dataset_id": status.get("selected_dataset_id") or layout.dataset_id,
        "datasets": datasets,
    })
    atomic_json(status_path, status)

def publish_warp_dataset(
    layout: RunLayout,
    *,
    source_artifact_id: str | None = None,
    preprocessing: dict[str, Any] | None = None,
) -> Path:
    """Expose one complete Warp dataset while keeping its mutable project hidden."""
    layout.create()
    project = layout.training_dir.resolve()
    if not project.is_dir():
        raise FileNotFoundError(f"Warp project is missing: {project}")

    tiltstack = project / "tiltstack"
    if tiltstack.is_dir():
        _replace_symlink(layout.warp_data_dir / "tiltstack", tiltstack)
    for candidate in ("average", "averages", "tilt_images", "raw_data"):
        path = project / candidate
        if path.exists():
            _replace_symlink(layout.warp_data_dir / candidate, path)

    metadata_records: list[dict[str, Any]] = []
    metadata_suffixes = {".xml", ".tomostar", ".settings", ".xf", ".json", ".rawtlt", ".tlt"}
    for source in sorted(project.iterdir()):
        if not source.is_file() or source.name.startswith("_"):
            continue
        if source.suffix.lower() not in metadata_suffixes and source.name != "conversion_validation.json":
            continue
        destination = layout.warp_metadata_dir / source.name
        _replace_symlink(destination, source)
        metadata_records.append({"source": str(source), "published": str(destination)})

    stacks = sorted(tiltstack.glob("*/*.st")) if tiltstack.is_dir() else []
    if len(stacks) != 1:
        raise RuntimeError(f"expected one Warp tilt stack in {tiltstack}, found {len(stacks)}")
    stack_info = _mrc_stack_info(stacks[0])
    pixel = float(stack_info.get("pixel_size_A") or 0.0)
    if layout.dataset_id != "native" and pixel > 0:
        expected = format_angpix(pixel)
        # Dataset names are human-readable and may round to fewer decimals, so only
        # reject material mismatches rather than string differences.
        from .runlayout import parse_angpix_id
        nominal = parse_angpix_id(layout.dataset_id)
        if abs(nominal - pixel) > max(1e-3, pixel * 5e-3):
            raise RuntimeError(
                f"dataset directory {layout.dataset_id} does not match stack pixel size {pixel:g} A/px"
            )

    stack_digest, stack_hash_mode = hash_file(stacks[0])
    conversion_manifests = sorted(project.glob("*.conversion.json"))
    conversion = None
    if len(conversion_manifests) == 1:
        conversion = json.loads(conversion_manifests[0].read_text())

    manifest = {
        "schema_version": 1,
        "artifact_type": "warp_tilt_series_dataset",
        "artifact_id": f"warp-{layout.dataset_id}-{stack_digest[:12]}",
        "source_artifact_id": source_artifact_id,
        "dataset_id": layout.dataset_id,
        "basename": layout.basename,
        "condition": layout.condition,
        "pixel_size_A": pixel,
        "stack": str(stacks[0]),
        "stack_sha256": stack_digest,
        "stack_hash_mode": stack_hash_mode,
        "stack_shape_zyx": stack_info.get("shape_zyx"),
        "warp_project": str(layout.training_dir),
        "data_directory": str(layout.warp_data_dir),
        "metadata_directory": str(layout.warp_metadata_dir),
        "reconstructions_directory": str(layout.warp_reconstructions_dir),
        "metadata": metadata_records,
        "conversion": conversion,
        "preprocessing": preprocessing,
        "coordinate_policy": "physical coordinates preserved; detector resampling is recorded in preprocessing",
        "allowed_uses": ["visualization", "MissAlignment input", "Warp reconstruction"],
        "forbidden_uses": ["silent replacement of quantitative observations"],
        "status": "complete",
    }
    atomic_json(layout.dataset_manifest, manifest)
    _update_project_dataset_records(layout, manifest)
    layout.dataset_config.write_text(
        "[dataset]\n"
        f'id = "{layout.dataset_id}"\n'
        f'pixel_size_A = {pixel!r}\n'
        f'condition = "{layout.condition}"\n'
        f'warp_project = ".warp_project"\n'
        f'status = "complete"\n'
    )
    return layout.dataset_manifest
