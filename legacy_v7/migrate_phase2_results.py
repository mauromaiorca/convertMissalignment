#!/usr/bin/env python3
"""Safely import completed Phase-2 MissAlignment results into a fresh Phase-1 project.

This is a reconstruction-focused migration.  It imports only the immutable Warp
geometry snapshots and the Phase-2 result contracts needed by Phase 3.  It does
NOT import old jobs, code provenance, checkpoints, TensorBoard logs, or Phase-3
outputs.

Expected inputs are the source and destination *run directories*, e.g.:

  .../testABCDE/64x_Vero_02_raw_xf_affine_fixed_standard

Project roots containing exactly one run directory are also accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import time
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_VERSION = "1.1.0"
SNAPSHOTS = ("pre_missalign", "missalign_smoke", "missalign_full")
COPY_SUFFIXES = {".xml", ".json"}
COPY_NAMES = {"_converted.marker"}
LARGE_FINGERPRINT_CHUNK = 4 * 1024 * 1024


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunRef:
    run_dir: Path
    project_root: Path
    settings: Path


@dataclass(frozen=True)
class CopyRecord:
    source: str
    destination: str
    source_sha256: str | None
    destination_sha256: str | None
    kind: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path: Path, *, full: bool) -> dict[str, Any]:
    path = path.resolve()
    st = path.stat()
    if full or st.st_size <= 2 * LARGE_FINGERPRINT_CHUNK:
        return {
            "path": str(path),
            "size": st.st_size,
            "method": "sha256-full",
            "sha256": sha256_file(path),
        }
    h = hashlib.sha256()
    with path.open("rb") as handle:
        first = handle.read(LARGE_FINGERPRINT_CHUNK)
        handle.seek(max(0, st.st_size - LARGE_FINGERPRINT_CHUNK))
        last = handle.read(LARGE_FINGERPRINT_CHUNK)
    h.update(str(st.st_size).encode("ascii"))
    h.update(b"\0")
    h.update(first)
    h.update(b"\0")
    h.update(last)
    return {
        "path": str(path),
        "size": st.st_size,
        "method": "sha256-size-first4MiB-last4MiB",
        "sha256": h.hexdigest(),
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise MigrationError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise MigrationError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MigrationError(f"{label} must contain a JSON object: {path}")
    return value


def load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise MigrationError(f"project_settings.toml is missing: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise MigrationError(f"invalid project_settings.toml: {path}: {exc}") from exc
    return value


def _project_root_to_run(path: Path) -> RunRef | None:
    settings = path / "project_settings.toml"
    if not settings.is_file():
        return None
    candidates = sorted(
        child for child in path.iterdir()
        if child.is_dir() and (child / "warp").is_dir() and (child / "manifests").is_dir()
    )
    if len(candidates) != 1:
        raise MigrationError(
            f"project root must contain exactly one run directory; found {len(candidates)} in {path}: "
            + ", ".join(str(p.name) for p in candidates)
        )
    return RunRef(candidates[0], path, settings)


def resolve_run(value: str) -> RunRef:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise MigrationError(f"directory does not exist: {path}")

    # Direct run directory.
    if (path / "warp").is_dir() and (path / "manifests").is_dir():
        project_root = path.parent
        settings = project_root / "project_settings.toml"
        if not settings.is_file() and (path / "project_settings.toml").is_file():
            project_root = path
            settings = path / "project_settings.toml"
        return RunRef(path, project_root, settings)

    # Direct project root.
    direct = _project_root_to_run(path)
    if direct is not None:
        return direct

    # Repository root containing exactly one prepared project.  Search only
    # immediate children so unrelated nested examples are not selected.
    projects: list[RunRef] = []
    for child in sorted(path.iterdir()):
        if not child.is_dir():
            continue
        candidate = _project_root_to_run(child)
        if candidate is not None:
            projects.append(candidate)
    if len(projects) == 1:
        return projects[0]
    if len(projects) > 1:
        raise MigrationError(
            f"repository root contains multiple prepared projects; pass one project or run directory explicitly: "
            + ", ".join(str(item.project_root) for item in projects)
        )

    raise MigrationError(
        f"cannot identify a run directory, project root, or repository root at {path}; expected "
        "warp/ + manifests/, project_settings.toml, or exactly one child project"
    )


def first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, "", []):
            return mapping[key]
    return None


def canonical_config(cfg: dict[str, Any]) -> dict[str, Any]:
    project = cfg.get("project", {}) or {}
    input_cfg = cfg.get("input", {}) or {}
    conversion = cfg.get("conversion", {}) or {}
    missalignment = cfg.get("missalignment", {}) or {}
    multires = cfg.get("multiresolution", {}) or {}
    geometry = cfg.get("geometry", {}) or {}
    tilt_series = cfg.get("tilt_series", []) or []
    ts0 = tilt_series[0] if tilt_series and isinstance(tilt_series[0], dict) else {}
    ts_imod = ts0.get("imod", {}) or {}
    ts_bin = ts0.get("binning", {}) or {}

    conditions = first_present(input_cfg, ("conditions",))
    if conditions is None:
        conditions = first_present(conversion, ("initial_conditions",))
    if conditions is None and project.get("condition"):
        conditions = [project.get("condition")]
    if isinstance(conditions, str):
        conditions = [conditions]

    basename = first_present(project, ("basename", "name")) or ts0.get("basename") or ts0.get("id")
    raw_stack = first_present(input_cfg, ("raw_stack",)) or ts_imod.get("raw_stack")
    final_tilt = first_present(input_cfg, ("final_tilt_file", "raw_tilt_file")) or ts_imod.get("tlt")
    final_xf = first_present(input_cfg, ("final_xf_file",)) or ts_imod.get("xf")
    extra_bin = first_present(multires, ("extra_projection_binning",))
    if extra_bin is None:
        extra_bin = ts_bin.get("extra_projection_binning")

    geometry_keys = (
        "tilt_axis_angle_deg",
        "raw_shape_xyz",
        "raw_pixel_size_A",
        "aligned_shape_xyz",
        "aligned_pixel_size_A",
        "target_volume_shape_xyz",
        "target_pixel_size_A",
        "target_volume_physical_A",
    )
    geometry_core = {key: geometry.get(key) for key in geometry_keys if geometry.get(key) is not None}
    if not geometry_core and ts_imod:
        geometry_core = {
            "raw_shape_xyz": ts_imod.get("raw_dimensions_xyz"),
            "raw_pixel_size_A": ts_imod.get("raw_pixel_size_A"),
            "aligned_shape_xyz": ts_imod.get("aligned_dimensions_xyz"),
            "aligned_pixel_size_A": ts_imod.get("aligned_pixel_size_A"),
            "target_volume_shape_xyz": ts_imod.get("target_volume_dimensions_xyz"),
            "target_pixel_size_A": ts_imod.get("target_voxel_size_A"),
            "tilt_count": ts_imod.get("tilt_count"),
        }

    return {
        "basename": basename,
        "conditions": conditions,
        "refinement_mode": missalignment.get("refinement_mode") or project.get("refinement_mode") or "standard",
        "result_backend": missalignment.get("result_backend"),
        "extra_projection_binning": extra_bin,
        "raw_stack": str(Path(raw_stack).expanduser()) if raw_stack else None,
        "final_tilt_file": str(Path(final_tilt).expanduser()) if final_tilt else None,
        "final_xf_file": str(Path(final_xf).expanduser()) if final_xf else None,
        "geometry": geometry_core,
    }


def compare_configs(src: dict[str, Any], dst: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for key in (
        "basename",
        "conditions",
        "refinement_mode",
        "result_backend",
        "extra_projection_binning",
        "raw_stack",
        "final_tilt_file",
        "final_xf_file",
        "geometry",
    ):
        a = src.get(key)
        b = dst.get(key)
        # Missing legacy values do not establish incompatibility; conflicting known values do.
        if a in (None, "", [], {}) or b in (None, "", [], {}):
            continue
        if a != b:
            mismatches.append(f"{key}: source={a!r}, destination={b!r}")
    return mismatches


def exactly_one_root_xml(directory: Path, label: str) -> Path:
    xmls = sorted(p for p in directory.glob("*.xml") if p.is_file() and p.stat().st_size > 0)
    if len(xmls) != 1:
        raise MigrationError(f"{label}: expected exactly one non-empty root XML in {directory}, found {len(xmls)}")
    try:
        ET.parse(xmls[0])
    except ET.ParseError as exc:
        raise MigrationError(f"{label}: XML is malformed: {xmls[0]}: {exc}") from exc
    return xmls[0]


def resolve_snapshot_xml_reference(
    raw_value: Any,
    *,
    source: RunRef,
    snapshot_dir: Path,
    snapshot_name: str,
    label: str,
    fallback: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a possibly stale manifest XML path inside a source snapshot.

    Older projects often store absolute paths.  Moving or copying the project
    makes those paths stale, even though the XML exists in the source snapshot.
    The snapshot directory is authoritative; the embedded absolute prefix is not.
    """
    raw = str(raw_value or "").strip()
    candidates: list[tuple[str, Path]] = []

    if raw:
        original = Path(raw).expanduser()
        if original.is_absolute():
            candidates.append(("manifest-absolute", original))
        else:
            candidates.append(("run-relative", source.run_dir / original))
            candidates.append(("snapshot-relative", snapshot_dir / original))

        parts = original.parts
        for index in range(len(parts) - 1):
            if parts[index] == "warp" and parts[index + 1] == snapshot_name:
                suffix = Path(*parts[index + 2:])
                candidates.append(("remapped-stale-prefix", snapshot_dir / suffix))
                break

        if original.name:
            candidates.append(("snapshot-basename", snapshot_dir / original.name))
            for match in sorted(snapshot_dir.rglob(original.name)):
                candidates.append(("snapshot-recursive-basename", match))

    if fallback is not None:
        candidates.append(("validated-root-fallback", fallback))

    seen: set[Path] = set()
    valid: list[tuple[str, Path]] = []
    for method, candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, RuntimeError):
            continue
        if resolved in seen or not resolved.is_file() or resolved.stat().st_size <= 0:
            continue
        try:
            resolved.relative_to(snapshot_dir.resolve())
        except ValueError:
            continue
        if resolved.suffix.lower() != ".xml":
            continue
        try:
            ET.parse(resolved)
        except ET.ParseError:
            continue
        seen.add(resolved)
        valid.append((method, resolved))

    if not valid:
        raise MigrationError(
            f"{label} cannot be resolved inside {snapshot_dir}; embedded value={raw!r}. "
            "The manifest may be stale and no matching XML was found in the source snapshot."
        )

    # Prefer the most specific remapped path over a generic basename/fallback.
    priority = {
        "manifest-absolute": 0,
        "run-relative": 1,
        "snapshot-relative": 2,
        "remapped-stale-prefix": 3,
        "snapshot-basename": 4,
        "snapshot-recursive-basename": 5,
        "validated-root-fallback": 6,
    }
    valid.sort(key=lambda item: priority[item[0]])
    method, resolved = valid[0]
    return resolved, {
        "embedded_value": raw or None,
        "resolved_path": str(resolved),
        "resolution_method": method,
        "embedded_path_was_stale": bool(raw) and str(resolved) != str(Path(raw).expanduser()),
    }


def find_tilt_stack(base_warp_dir: Path) -> Path:
    stacks = sorted(p for p in (base_warp_dir / "tiltstack").glob("*/*.st") if p.exists() or p.is_symlink())
    if len(stacks) != 1:
        raise MigrationError(
            f"expected exactly one tilt stack below {base_warp_dir / 'tiltstack'}, found {len(stacks)}"
        )
    resolved = stacks[0].resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise MigrationError(f"tilt stack is missing or empty: {stacks[0]} -> {resolved}")
    return stacks[0]


def identify_base_warp(run: RunRef, condition: str | None) -> Path:
    preferred = run.run_dir / "warp" / f"warp_{condition}" if condition else None
    if preferred and preferred.is_dir():
        return preferred
    candidates = sorted(
        p for p in (run.run_dir / "warp").glob("warp_*")
        if p.is_dir() and p.name not in SNAPSHOTS
    )
    if len(candidates) != 1:
        raise MigrationError(
            f"cannot identify exactly one Phase-1 base Warp directory in {run.run_dir / 'warp'}; "
            f"found {[p.name for p in candidates]}"
        )
    return candidates[0]


def validate_phase1_compatibility(
    source: RunRef,
    destination: RunRef,
    src_cfg: dict[str, Any],
    dst_cfg: dict[str, Any],
    *,
    full_large_hash: bool,
) -> dict[str, Any]:
    src_core = canonical_config(src_cfg)
    dst_core = canonical_config(dst_cfg)
    mismatches = compare_configs(src_core, dst_core)
    if mismatches:
        raise MigrationError("Phase-1 configuration mismatch:\n  - " + "\n  - ".join(mismatches))

    condition = None
    if src_core.get("conditions"):
        condition = src_core["conditions"][0]
    src_base = identify_base_warp(source, condition)
    dst_base = identify_base_warp(destination, condition)
    src_xml = exactly_one_root_xml(src_base, "source base Warp")
    dst_xml = exactly_one_root_xml(dst_base, "destination base Warp")
    src_xml_hash = sha256_file(src_xml)
    dst_xml_hash = sha256_file(dst_xml)
    if src_xml_hash != dst_xml_hash:
        raise MigrationError(
            "Phase-1 base Warp XML differs between source and destination. "
            f"source={src_xml_hash}, destination={dst_xml_hash}"
        )

    src_stack_link = find_tilt_stack(src_base)
    dst_stack_link = find_tilt_stack(dst_base)
    src_stack = src_stack_link.resolve(strict=True)
    dst_stack = dst_stack_link.resolve(strict=True)
    src_fp = file_fingerprint(src_stack, full=full_large_hash)
    dst_fp = file_fingerprint(dst_stack, full=full_large_hash)
    if (src_fp["size"], src_fp["sha256"]) != (dst_fp["size"], dst_fp["sha256"]):
        raise MigrationError(
            "Phase-1 tilt-stack content differs between source and destination. "
            f"source={src_fp}, destination={dst_fp}"
        )

    return {
        "source_canonical_config": src_core,
        "destination_canonical_config": dst_core,
        "source_base_warp": str(src_base),
        "destination_base_warp": str(dst_base),
        "base_xml": {
            "source": str(src_xml),
            "destination": str(dst_xml),
            "sha256": src_xml_hash,
        },
        "tilt_stack": {
            "source_link": str(src_stack_link),
            "destination_link": str(dst_stack_link),
            "source_fingerprint": src_fp,
            "destination_fingerprint": dst_fp,
        },
    }


def validate_source_phase2(source: RunRef) -> dict[str, Any]:
    snapshots = {name: source.run_dir / "warp" / name for name in SNAPSHOTS}
    for name, directory in snapshots.items():
        if not directory.is_dir():
            raise MigrationError(f"source Phase-2 snapshot is missing: {directory}")

    root_xmls = {name: exactly_one_root_xml(directory, f"source {name}") for name, directory in snapshots.items()}

    smoke_path = source.run_dir / "missalignment" / "results" / "smoke_verdict.json"
    smoke = load_json(smoke_path, "smoke verdict")
    if smoke.get("smoke") != "ok":
        raise MigrationError(f"source smoke verdict is not ok: {smoke_path}: {smoke.get('smoke')!r}")
    smoke_xml, smoke_xml_resolution = resolve_snapshot_xml_reference(
        smoke.get("xml"),
        source=source,
        snapshot_dir=snapshots["missalign_smoke"],
        snapshot_name="missalign_smoke",
        label="source smoke verdict XML",
        fallback=root_xmls["missalign_smoke"],
    )

    phase2_path = source.run_dir / "manifests" / "phase2_manifest.json"
    phase2 = load_json(phase2_path, "Phase-2 manifest")
    if phase2.get("status") != "completed":
        raise MigrationError(f"source Phase-2 status is not completed: {phase2.get('status')!r}")

    result_path = source.run_dir / "manifests" / "result_manifest.json"
    result = load_json(result_path, "result manifest")
    final_xml, final_xml_resolution = resolve_snapshot_xml_reference(
        result.get("final_xml"),
        source=source,
        snapshot_dir=snapshots["missalign_full"],
        snapshot_name="missalign_full",
        label="source result_manifest.final_xml",
        fallback=root_xmls["missalign_full"],
    )
    final_rel = final_xml.relative_to(snapshots["missalign_full"].resolve())

    return {
        "snapshots": {name: str(path) for name, path in snapshots.items()},
        "root_xmls": {name: str(path) for name, path in root_xmls.items()},
        "smoke_verdict_path": str(smoke_path),
        "smoke_verdict": smoke,
        "smoke_xml": str(smoke_xml),
        "smoke_xml_resolution": smoke_xml_resolution,
        "phase2_manifest_path": str(phase2_path),
        "phase2_manifest": phase2,
        "result_manifest_path": str(result_path),
        "result_manifest": result,
        "final_xml": str(final_xml),
        "final_xml_relative": str(final_rel),
        "final_xml_resolution": final_xml_resolution,
    }


def selected_entries(snapshot_root: Path) -> list[Path]:
    selected: list[Path] = []
    for path in snapshot_root.rglob("*"):
        rel = path.relative_to(snapshot_root)
        if rel.parts and rel.parts[0] == "tiltstack":
            selected.append(path)
            continue
        if path.is_file() or path.is_symlink():
            if path.name in COPY_NAMES or path.suffix.lower() in COPY_SUFFIXES:
                selected.append(path)
    return sorted(selected, key=lambda p: (len(p.relative_to(snapshot_root).parts), str(p)))


def rewrite_path_string(value: str, source: RunRef, destination: RunRef) -> tuple[str, bool]:
    mappings = (
        (str(source.run_dir), str(destination.run_dir)),
        (str(source.project_root), str(destination.project_root)),
    )
    for old, new in mappings:
        if value == old:
            return new, True
        if value.startswith(old + os.sep):
            return new + value[len(old):], True
    return value, False


def rewrite_json_paths(value: Any, source: RunRef, destination: RunRef) -> tuple[Any, int]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        count = 0
        for key, item in value.items():
            new_item, changed = rewrite_json_paths(item, source, destination)
            out[key] = new_item
            count += changed
        return out, count
    if isinstance(value, list):
        out_list = []
        count = 0
        for item in value:
            new_item, changed = rewrite_json_paths(item, source, destination)
            out_list.append(new_item)
            count += changed
        return out_list, count
    if isinstance(value, str):
        rewritten, changed = rewrite_path_string(value, source, destination)
        return rewritten, int(changed)
    return value, 0


def copy_selected_snapshot(
    source_root: Path,
    stage_root: Path,
    final_root: Path,
    source: RunRef,
    destination: RunRef,
) -> tuple[list[CopyRecord], list[dict[str, str]], int]:
    records: list[CopyRecord] = []
    link_rewrites: list[dict[str, str]] = []
    json_rewrites = 0
    stage_root.mkdir(parents=True, exist_ok=True)

    for src in selected_entries(source_root):
        rel = src.relative_to(source_root)
        dst = stage_root / rel
        if src.is_dir() and not src.is_symlink():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_symlink():
            raw_target = os.readlink(src)
            absolute_target = (src.parent / raw_target).resolve() if not os.path.isabs(raw_target) else Path(raw_target).resolve()
            mapped_target = absolute_target
            rewritten = False
            try:
                suffix = absolute_target.relative_to(source.run_dir)
                mapped_target = destination.run_dir / suffix
                rewritten = True
            except ValueError:
                try:
                    suffix = absolute_target.relative_to(source.project_root)
                    mapped_target = destination.project_root / suffix
                    rewritten = True
                except ValueError:
                    pass
            if rewritten:
                if not mapped_target.exists():
                    raise MigrationError(
                        f"cannot rewire snapshot symlink because destination target is absent: "
                        f"{src} -> {absolute_target}; expected {mapped_target}"
                    )
                final_link = final_root / rel
                target_text = os.path.relpath(mapped_target, final_link.parent)
                link_rewrites.append({
                    "source_link": str(src),
                    "source_target": str(absolute_target),
                    "destination_link": str(final_link),
                    "destination_target": str(mapped_target),
                })
            else:
                if not absolute_target.exists():
                    raise MigrationError(f"external source symlink is broken: {src} -> {absolute_target}")
                target_text = raw_target
            dst.symlink_to(target_text)
            records.append(CopyRecord(str(src), str(dst), None, None, "symlink"))
            continue

        if src.suffix.lower() == ".json":
            try:
                obj = json.loads(src.read_text())
            except json.JSONDecodeError:
                shutil.copy2(src, dst)
            else:
                obj, changed = rewrite_json_paths(obj, source, destination)
                json_rewrites += changed
                atomic_json(dst, obj)
        else:
            shutil.copy2(src, dst)
        records.append(CopyRecord(
            str(src), str(dst), sha256_file(src), sha256_file(dst), "file"
        ))

    return records, link_rewrites, json_rewrites


def map_source_path(path_value: Any, source: RunRef, destination: RunRef) -> str | None:
    if path_value in (None, ""):
        return None
    original = str(path_value)
    mapped, changed = rewrite_path_string(original, source, destination)
    if changed:
        return mapped
    path = Path(original)
    if not path.is_absolute():
        return str(destination.run_dir / path)
    return original


def destination_has_phase2(destination: RunRef) -> list[Path]:
    found: list[Path] = []
    for name in SNAPSHOTS:
        path = destination.run_dir / "warp" / name
        if path.exists() and any(path.iterdir()):
            found.append(path)
    for path in (
        destination.run_dir / "manifests" / "phase2_manifest.json",
        destination.run_dir / "missalignment" / "results" / "smoke_verdict.json",
    ):
        if path.exists():
            found.append(path)
    result_path = destination.run_dir / "manifests" / "result_manifest.json"
    if result_path.is_file():
        result = load_json(result_path, "destination result manifest")
        if result.get("final_xml"):
            found.append(result_path)
    return found


def backup_existing(
    paths: Iterable[Path],
    backup_root: Path,
    *,
    anchor: Path,
) -> list[dict[str, str]]:
    backed_up: list[dict[str, str]] = []
    for path in paths:
        if not path.exists() and not path.is_symlink():
            continue
        try:
            rel = path.relative_to(anchor)
        except ValueError:
            rel = Path(path.name)
        target = backup_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        backed_up.append({"original": str(path), "backup": str(target)})
    return backed_up


def build_updated_contracts(
    source: RunRef,
    destination: RunRef,
    source_info: dict[str, Any],
    import_manifest_path: Path,
) -> dict[Path, dict[str, Any]]:
    result_path = destination.run_dir / "manifests" / "result_manifest.json"
    destination_result = load_json(result_path, "destination result manifest")
    source_result = source_info["result_manifest"]

    destination_result.update({
        "training_directory": str(destination.run_dir / "warp" / "missalign_full"),
        "pre_missalign_directory": str(destination.run_dir / "warp" / "pre_missalign"),
        "smoke_directory": str(destination.run_dir / "warp" / "missalign_smoke"),
        "initial_xml": map_source_path(source_result.get("initial_xml"), source, destination),
        "final_xml": str(destination.run_dir / "warp" / "missalign_full" / source_info["final_xml_relative"]),
        "final_iteration": source_result.get("final_iteration"),
        "phase2_completed_at": source_result.get("phase2_completed_at"),
        "phase2_slurm_job_id": source_result.get("phase2_slurm_job_id"),
        "phase2_hostname": source_result.get("phase2_hostname"),
        "phase2_imported": True,
        "phase2_import_manifest": str(import_manifest_path),
        "phase2_import_source_run": str(source.run_dir),
    })

    smoke, _ = rewrite_json_paths(source_info["smoke_verdict"], source, destination)
    source_checkpoints = list(smoke.get("checkpoints") or [])
    smoke["training_directory"] = str(destination.run_dir / "warp" / "missalign_smoke")
    smoke["source_checkpoints"] = source_checkpoints
    smoke["checkpoints"] = []
    smoke["checkpoints_imported"] = False
    smoke["reconstruction_only_import"] = True
    smoke["phase2_imported"] = True
    smoke["phase2_import_manifest"] = str(import_manifest_path)

    phase2, _ = rewrite_json_paths(source_info["phase2_manifest"], source, destination)
    phase2.update({
        "status": "completed",
        "imported": True,
        "imported_at": utc_now(),
        "import_source_run": str(source.run_dir),
        "import_destination_run": str(destination.run_dir),
        "import_manifest": str(import_manifest_path),
        "result_manifest": str(result_path),
        "smoke_verdict": str(destination.run_dir / "missalignment" / "results" / "smoke_verdict.json"),
        "final_xml": destination_result["final_xml"],
    })

    contracts: dict[Path, dict[str, Any]] = {
        result_path: destination_result,
        destination.run_dir / "missalignment" / "results" / "smoke_verdict.json": smoke,
        destination.run_dir / "manifests" / "phase2_manifest.json": phase2,
    }

    source_snapshot_manifest = source.run_dir / "manifests" / "warp_snapshot_manifest.json"
    if source_snapshot_manifest.is_file():
        snapshot = load_json(source_snapshot_manifest, "source Warp snapshot manifest")
        snapshot, _ = rewrite_json_paths(snapshot, source, destination)
        snapshot["imported"] = True
        snapshot["import_manifest"] = str(import_manifest_path)
        contracts[destination.run_dir / "manifests" / "warp_snapshot_manifest.json"] = snapshot

    return contracts


def validate_staged_snapshots(
    stage_warp: Path,
    final_warp: Path,
    source_info: dict[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in SNAPSHOTS:
        directory = stage_warp / name
        root_xml = exactly_one_root_xml(directory, f"staged {name}")
        tilt_stacks = sorted(p for p in (directory / "tiltstack").glob("*/*.st") if p.exists() or p.is_symlink())
        if len(tilt_stacks) != 1:
            raise MigrationError(f"staged {name}: expected one tilt-stack link, found {len(tilt_stacks)}")
        staged_link = tilt_stacks[0]
        if staged_link.is_symlink():
            raw_target = os.readlink(staged_link)
            final_link = final_warp / name / staged_link.relative_to(directory)
            candidate = Path(os.path.normpath(os.path.join(str(final_link.parent), raw_target)))
            target_from_final_location = candidate.resolve(strict=True)
        else:
            target_from_final_location = staged_link.resolve(strict=True)
        report[name] = {
            "root_xml": str(root_xml),
            "root_xml_sha256": sha256_file(root_xml),
            "tilt_stack_link": str(staged_link),
            "tilt_stack_target_after_install": str(target_from_final_location),
        }
    final_staged = stage_warp / "missalign_full" / source_info["final_xml_relative"]
    if not final_staged.is_file():
        raise MigrationError(f"staged final XML was not copied: {final_staged}")
    report["final_xml"] = str(final_staged)
    report["final_xml_sha256"] = sha256_file(final_staged)
    return report


def script_sha256() -> str | None:
    try:
        return sha256_file(Path(__file__).resolve())
    except OSError:
        return None


def perform(args: argparse.Namespace) -> int:
    source = resolve_run(args.source)
    destination = resolve_run(args.destination)
    if source.run_dir == destination.run_dir:
        raise MigrationError("source and destination resolve to the same run directory")

    src_cfg = load_toml(source.settings)
    dst_cfg = load_toml(destination.settings)
    compatibility = validate_phase1_compatibility(
        source, destination, src_cfg, dst_cfg, full_large_hash=args.full_large_file_hash
    )
    source_info = validate_source_phase2(source)

    existing = destination_has_phase2(destination)
    if existing and not args.replace_existing:
        raise MigrationError(
            "destination already contains Phase-2 results; refusing to overwrite:\n  - "
            + "\n  - ".join(str(p) for p in existing)
            + "\nUse --replace-existing to back them up and replace them."
        )

    print("Phase-1 compatibility: OK")
    print(f"Source run:      {source.run_dir}")
    print(f"Destination run: {destination.run_dir}")
    print("Import set:")
    for name in SNAPSHOTS:
        print(f"  - warp/{name}: XML/JSON metadata and tiltstack links only")
    print("  - manifests/phase2_manifest.json")
    print("  - manifests/result_manifest.json (destination contract updated, not blindly replaced)")
    print("  - manifests/warp_snapshot_manifest.json, when present")
    print("  - missalignment/results/smoke_verdict.json")
    print("Excluded: checkpoints, models, TensorBoard data, old jobs, logs, code provenance, Phase-3 outputs")

    if not args.apply:
        print("DRY RUN ONLY: no files were changed. Re-run with --apply to perform the import.")
        return 0

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    stage_root = destination.run_dir / f".phase2_import_staging_{os.getpid()}"
    stage_warp = stage_root / "warp"
    backup_root = destination.run_dir / "phase2_import_backups" / stamp
    import_manifest_path = destination.run_dir / "manifests" / "phase2_import_manifest.json"

    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_warp.mkdir(parents=True)

    copy_records: list[CopyRecord] = []
    link_rewrites: list[dict[str, str]] = []
    json_rewrites = 0
    try:
        for name in SNAPSHOTS:
            records, rewrites, changed = copy_selected_snapshot(
                source.run_dir / "warp" / name,
                stage_warp / name,
                destination.run_dir / "warp" / name,
                source,
                destination,
            )
            copy_records.extend(records)
            link_rewrites.extend(rewrites)
            json_rewrites += changed

        staged_validation = validate_staged_snapshots(
            stage_warp, destination.run_dir / "warp", source_info
        )

        contract_payloads = build_updated_contracts(
            source, destination, source_info, import_manifest_path
        )

        to_backup = [destination.run_dir / "warp" / name for name in SNAPSHOTS]
        to_backup.extend(contract_payloads.keys())
        to_backup.append(import_manifest_path)
        # Always preserve destination contracts before updating them. Existing
        # Phase-2 artifacts have already been refused unless --replace-existing
        # was supplied.
        backed_up = backup_existing(
            to_backup, backup_root, anchor=destination.run_dir
        )

        installed_paths: list[Path] = []
        try:
            for name in SNAPSHOTS:
                final_path = destination.run_dir / "warp" / name
                final_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(stage_warp / name), str(final_path))
                installed_paths.append(final_path)

            for path, payload in contract_payloads.items():
                atomic_json(path, payload)
                installed_paths.append(path)

            final_validation: dict[str, Any] = {}
            for name in SNAPSHOTS:
                final_dir = destination.run_dir / "warp" / name
                root_xml = exactly_one_root_xml(final_dir, f"installed {name}")
                stack = find_tilt_stack(final_dir)
                final_validation[name] = {
                    "root_xml": str(root_xml),
                    "root_xml_sha256": sha256_file(root_xml),
                    "tilt_stack_link": str(stack),
                    "tilt_stack_target": str(stack.resolve(strict=True)),
                }

            final_result = load_json(
                destination.run_dir / "manifests" / "result_manifest.json",
                "installed destination result manifest",
            )
            final_xml = Path(final_result["final_xml"])
            if not final_xml.is_file():
                raise MigrationError(f"installed result_manifest.final_xml does not exist: {final_xml}")
            final_validation["result_manifest_final_xml"] = str(final_xml)
            final_validation["result_manifest_final_xml_sha256"] = sha256_file(final_xml)

            manifest = {
                "schema_version": 1,
                "artifact_type": "phase2_result_import",
                "script_version": SCRIPT_VERSION,
                "script_path": str(Path(__file__).resolve()),
                "script_sha256": script_sha256(),
                "created_at": utc_now(),
                "host": socket.getfqdn() or socket.gethostname(),
                "user": os.environ.get("USER"),
                "source_project_root": str(source.project_root),
                "source_run_dir": str(source.run_dir),
                "source_settings": str(source.settings),
                "source_settings_sha256": sha256_file(source.settings),
                "destination_project_root": str(destination.project_root),
                "destination_run_dir": str(destination.run_dir),
                "destination_settings": str(destination.settings),
                "destination_settings_sha256": sha256_file(destination.settings),
                "mode": "reconstruction_only",
                "compatibility": compatibility,
                "source_phase2": {
                    "phase2_status": source_info["phase2_manifest"].get("status"),
                    "phase2_slurm_job_id": source_info["phase2_manifest"].get("slurm_job_id"),
                    "final_xml": source_info["final_xml"],
                    "final_xml_sha256": sha256_file(Path(source_info["final_xml"])),
                    "smoke_status": source_info["smoke_verdict"].get("smoke"),
                    "smoke_xml_resolution": source_info["smoke_xml_resolution"],
                    "final_xml_resolution": source_info["final_xml_resolution"],
                },
                "copied_entries": [record.__dict__ for record in copy_records],
                "symlink_rewrites": link_rewrites,
                "json_path_rewrites": json_rewrites,
                "staged_validation": staged_validation,
                "installed_validation": final_validation,
                "backups": backed_up,
                "excluded": [
                    "model checkpoints",
                    "models/ and TensorBoard logs",
                    "old Slurm jobs",
                    "old code_provenance.json",
                    "Phase-2 runtime logs",
                    "Phase-3 outputs",
                ],
                "allowed_use": [
                    "Phase-3 IMOD reconstruction of pre_missalign, smoke, and full snapshots",
                    "diagnostic comparison of imported geometries",
                ],
                "forbidden_use": [
                    "claiming the imported run was generated by destination code",
                    "retraining or resuming MissAlignment from omitted checkpoints",
                    "silently replacing destination Phase-1 quantitative inputs",
                ],
            }
            atomic_json(import_manifest_path, manifest)
        except Exception:
            # Best-effort rollback of newly installed paths, then restore backups.
            for path in reversed(installed_paths):
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
            for item in reversed(backed_up):
                original = Path(item["original"])
                backup = Path(item["backup"])
                if backup.exists() or backup.is_symlink():
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(backup), str(original))
            raise
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root, ignore_errors=True)

    print("Phase-2 import completed successfully.")
    print(f"Import manifest: {import_manifest_path}")
    print("Next checks:")
    print(f"  python -m json.tool {import_manifest_path}")
    print(f"  python -m json.tool {destination.run_dir / 'manifests' / 'result_manifest.json'}")
    print(f"  python -m json.tool {destination.run_dir / 'missalignment' / 'results' / 'smoke_verdict.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import completed Phase-2 MissAlignment geometry snapshots into a fresh, compatible "
            "Phase-1 project. The default is a validation-only dry run."
        )
    )
    parser.add_argument("source", help="completed source project root or source run directory")
    parser.add_argument("destination", help="fresh destination project root or destination run directory")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the import; without this option the command only validates and reports",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="back up and replace existing destination Phase-2 artifacts",
    )
    parser.add_argument(
        "--full-large-file-hash",
        action="store_true",
        help=(
            "fully SHA-256 hash the Phase-1 tilt stack in source and destination; by default a "
            "size + first/last 4 MiB fingerprint is used for large stacks"
        ),
    )
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return perform(args)
    except MigrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
