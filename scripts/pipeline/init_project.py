#!/usr/bin/env python3
"""Discover and measure once, then write one canonical resolved TOML.

The output is ``OUT_DIR/project_settings.toml`` with provenance under
``OUT_DIR/provenance``. Later workflow steps consume that TOML and its explicit paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from . import discovery as DISC
from . import project_config as PC
from .runlayout import format_angpix


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def write_toml(path: Path, data: dict) -> None:
    """Minimal deterministic TOML writer for nested tables of scalars/lists."""
    lines = []

    def emit_table(name, table):
        lines.append(f"[{name}]")
        nested = []
        for k, v in table.items():
            if isinstance(v, dict):
                nested.append((k, v))
            else:
                lines.append(f"{k} = {_toml_value(v)}")
        lines.append("")
        for k, v in nested:
            emit_table(f"{name}.{k}", v)

    for name, table in data.items():
        emit_table(name, table)
    tmp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)


def _resolve_executable(name: str) -> tuple[str, str]:
    if "/" in name:
        return str(Path(name)), "configured"
    found = shutil.which(name)
    if found:
        return str(Path(found).resolve()), "PATH"
    return name, "unresolved-path"


def _reconstruction_contract(input_cfg: dict, inv: DISC.SourceInventory,
                             geom: dict, output_dir: Path) -> dict:
    rec = dict(input_cfg.get("reconstruction", {}) or {})
    imod = dict(rec.get("imod", {}) or {})
    volume = dict(imod.get("volume", {}) or {})
    filt = dict(imod.get("filter", {}) or {})
    outputs = dict(rec.get("outputs", {}) or {})
    validation = dict(rec.get("validation", {}) or {})
    warptools = dict(rec.get("warptools", {}) or {})

    imod_bin = str(imod.get("imod_bin_dir") or "")
    def _from_bin(program: str) -> str:
        return str(Path(imod_bin) / program) if imod_bin else program
    newstack_exe, newstack_src = _resolve_executable(str(imod.get("newstack_executable") or _from_bin("newstack")))
    tilt_exe, tilt_src = _resolve_executable(str(imod.get("tilt_executable") or _from_bin("tilt")))
    submfg_exe, submfg_src = _resolve_executable(str(imod.get("submfg_executable") or _from_bin("submfg")))
    ctfphaseflip_exe, ctfphaseflip_src = _resolve_executable(str(imod.get("ctfphaseflip_executable") or _from_bin("ctfphaseflip")))
    target_shape = geom.get("target_volume_shape_xyz") or []
    target_pixel = geom.get("target_pixel_size_A")
    volume_contract = {
        "size_source": volume.get("size_source", "target_geometry"),
        "voxel_size_source": volume.get("voxel_size_source", "target_geometry"),
        "thickness_source": volume.get("thickness_source", "target_geometry"),
        "shape_xyz": target_shape,
        "shape_frame": geom.get(
            "target_volume_frame",
            "imod_reconstruction_mrc_xyz__y_is_thickness",
        ),
    }
    if target_pixel is not None:
        volume_contract["voxel_size_A"] = target_pixel
    if len(target_shape) >= 2:
        volume_contract["thickness_px"] = int(target_shape[1])

    return {
        "enabled": bool(rec.get("enabled", True)),
        "backend": rec.get("backend", "imod"),
        "snapshots": rec.get("snapshots", ["pre_missalign", "smoke", "full"]),
        "canonical_snapshot": rec.get("canonical_snapshot", "full"),
        "diagnostic_snapshots": rec.get("diagnostic_snapshots", ["pre_missalign", "smoke"]),
        "warptools": {
            "enabled": bool(warptools.get("enabled", True)),
            "executable": str(warptools.get("executable") or "WarpTools"),
            "output_angpix_A": float(warptools.get("output_angpix_A", 0.0)),
            "device_list": str(warptools.get("device_list", "0")),
            "perdevice": int(warptools.get("perdevice", 1)),
            "dose_policy": str(
                warptools.get(
                    "dose_policy",
                    "preserve_if_valid_else_synthetic_monotonic_epsilon",
                )
            ),
            "allowed_uses": ["visualization", "diagnostic geometry comparison"],
            "forbidden_uses": ["FSC resolution estimation", "quantitative dose validation"],
        },
        "imod": {
            "imod_module": str(imod.get("imod_module") or ""),
            "imod_bin_dir": imod_bin,
            "newstack_executable": newstack_exe,
            "tilt_executable": tilt_exe,
            "submfg_executable": submfg_exe,
            "ctfphaseflip_executable": ctfphaseflip_exe,
            "execution_mode": str(imod.get("execution_mode") or "submfg_command_file"),
            "newst_template": str(inv.newst_com or imod.get("newst_template") or ""),
            "tilt_template": str(inv.tilt_com or imod.get("tilt_template") or ""),
            "newst_bin": int(imod.get("newst_bin", 0)),
            "use_gpu": bool(imod.get("use_gpu", False)),
            "gpu_id": int(imod.get("gpu_id", 0)),
            "halfmaps": bool(imod.get("halfmaps", False)),
            "half_split_mode": str(imod.get("half_split_mode", "angle")),
            "remove_temporary_stacks": bool(imod.get("remove_temporary_stacks", True)),
            "overwrite": bool(imod.get("overwrite", False)),
            "volume": volume_contract,
            "filter": {
                "policy": filt.get("policy", "inherit_from_tilt_com"),
            },
            "resolution": {
                "newstack_executable_source": newstack_src,
                "tilt_executable_source": tilt_src,
                "submfg_executable_source": submfg_src,
                "ctfphaseflip_executable_source": ctfphaseflip_src,
                "newst_template_source": "discovered" if inv.newst_com else "configured",
                "tilt_template_source": "discovered" if inv.tilt_com else "configured",
                "volume_geometry_source": "target_reconstruction",
            },
        },
        "outputs": {
            "canonical_root": outputs.get("canonical_root", "missalignment/runs/<dataset_id>/results/reconstructions/final"),
            "diagnostic_root": outputs.get("diagnostic_root", ".internal/attempts/reconstruction/<dataset_id>"),
        },
        "validation": {
            "require_stack_section_match": bool(validation.get("require_stack_section_match", True)),
            "require_transform_count_match": bool(validation.get("require_transform_count_match", True)),
            "require_tilt_count_match": bool(validation.get("require_tilt_count_match", True)),
            "require_finite_transforms": bool(validation.get("require_finite_transforms", True)),
            "require_nonsingular_transforms": bool(validation.get("require_nonsingular_transforms", True)),
            "require_round_trip_validation": bool(validation.get("require_round_trip_validation", True)),
            "max_round_trip_error_px": float(validation.get("max_round_trip_error_px", 0.05)),
            "verify_output_header": bool(validation.get("verify_output_header", True)),
            "verify_physical_volume": bool(validation.get("verify_physical_volume", True)),
        },
        "resolution": {
            "raw_stack_source": "resolved_input",
            "tilt_file_source": "resolved_input",
            "xtilt_file_source": "resolved_input" if inv.xtilt_file else "not_present",
            "target_geometry_source": geom.get("target_volume_source", ""),
            "output_root": str(output_dir),
        },
    }


def _measure_geometry(inv: DISC.SourceInventory):
    """Measure raw + aligned grids from real MRC headers (independent grids, 2.8)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline import geometry as G
    measured = G.measure_source_and_working(source_raw=inv.raw_stack,
                                            source_aligned=inv.aligned_stack)
    m = measured["measured"]
    out = {}
    if "source_raw" in m:
        r = m["source_raw"]
        out["raw_shape_xyz"] = [r.shape_xy[0], r.shape_xy[1], r.n_sections]
        out["raw_pixel_size_A"] = round(r.pixel_size_xy_A[0], 4)
    if "source_aligned" in m:
        a = m["source_aligned"]
        out["aligned_shape_xyz"] = [a.shape_xy[0], a.shape_xy[1], a.n_sections]
        out["aligned_pixel_size_A"] = round(a.pixel_size_xy_A[0], 4)
    return out, measured


def _tilt_axis(data_dir: Path, basename: str, inv: DISC.SourceInventory, cfg_geom: dict):
    """Source the tilt axis from align.com RotationAngle (2.11). Never silent 0.0."""
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "01_extract_etomo_params.py"
    # imod dir: the dir holding align.com (where newst.com/tilt.com were found), else data_dir
    imod_dir = Path(inv.tilt_com).parent if inv.tilt_com else Path(data_dir)
    try:
        import sys as _sys
        spec = importlib.util.spec_from_file_location("extract01_for_init", p)
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[spec.name] = mod          # dataclass decorator needs this registered
        spec.loader.exec_module(mod)
        value, source = mod.parse_tilt_axis_angle(imod_dir, Path(data_dir), basename,
                                                  Path(inv.mdoc_file) if inv.mdoc_file else None)
        if value is not None:
            return float(value), source
    except Exception as exc:  # extractor unavailable -> fall through to config
        source = f"extractor-error: {exc}"
    # explicit config override (recorded as the source), else hard fail
    cv = cfg_geom.get("tilt_axis_angle_deg")
    if cv not in (None, "", 0, 0.0):
        return float(cv), "config:[geometry].tilt_axis_angle_deg"
    raise PC.ConfigError(
        "tilt_axis_angle_deg could not be sourced from align.com (RotationAngle) and no explicit "
        "[geometry].tilt_axis_angle_deg was provided. Refusing to default to 0.0 (defect 2.11). "
        "Provide align.com or set the angle explicitly.")


def _hashes(inv: DISC.SourceInventory) -> dict:
    out = {}
    for fname in ("raw_stack", "aligned_stack", "final_xf", "tilt_file", "raw_tilt_file",
                  "xtilt_file", "tltxf_file", "defocus_file", "mdoc_file", "newst_com",
                  "tilt_com", "ctf_com", "source_reconstruction"):
        p = getattr(inv, fname, None)
        if not p or not Path(p).is_file():
            continue
        path = Path(p); size = path.stat().st_size
        h = hashlib.sha256()
        if size <= 64 * 1024 * 1024:
            with path.open("rb") as fh:
                for c in iter(lambda: fh.read(1 << 20), b""):
                    h.update(c)
            mode = "full_sha256"
        else:
            with path.open("rb") as fh:
                h.update(fh.read(8 << 20)); fh.seek(-(8 << 20), 2); h.update(fh.read(8 << 20))
            h.update(str(size).encode()); mode = "partial_sha256_head8M_tail8M_size"
        out[fname] = {"path": str(path), "size": size, "sha256": h.hexdigest()[:32], "mode": mode}
    return out


def init_project(input_cfg: dict, *, out_dir_override=None, data_dir_override=None,
                 basename_override=None) -> dict:
    """Run the single discover+measure step; return {resolved_toml, manifests, config}."""
    base = PC.from_dict(input_cfg)  # accepts legacy dialect too
    basename = basename_override or base.basename
    data_root = Path(data_dir_override or base.data_root)
    output_dir = Path(out_dir_override or base.output_dir)
    if not data_root.is_dir():
        raise PC.ConfigError(f"[paths].data_root is not a directory: {data_root}. "
                             "init must run where the (read-only) source project is readable.")

    resolved_dir = output_dir
    man_dir = output_dir / "provenance"
    man_dir.mkdir(parents=True, exist_ok=True)

    # 1. discover ONCE
    inp = input_cfg.get("input", {})
    overrides = {k: v for k, v in {
        "raw_stack": inp.get("raw_stack"), "aligned_stack": inp.get("aligned_stack"),
        "final_xf": inp.get("final_xf_file"), "tilt_file": inp.get("final_tilt_file"),
        "ctf_com": (input_cfg.get("ctf", {}) or {}).get("command_file"),
        "source_reconstruction": inp.get("source_reconstruction") or inp.get("reconstruction_stack"),
        "raw_tilt_file": inp.get("raw_tilt_file"),
        "xtilt_file": inp.get("xtilt_file"),
        "tltxf_file": inp.get("tltxf_file"),
        "mdoc_file": inp.get("mdoc_file"),
        "newst_com": inp.get("newst_com"),
        "tilt_com": inp.get("tilt_com"),
    }.items() if v}
    inv = DISC.discover_sources(data_root, basename, overrides=overrides)
    _atomic_json(man_dir / "source_inventory.json", inv.to_dict())
    _atomic_json(man_dir / "source_hashes.json", _hashes(inv))

    # 2. measure geometry (independent raw/aligned grids)
    geom_measured, full = _measure_geometry(inv)
    _atomic_json(man_dir / "geometry_manifest.json",
                 {"measured": geom_measured, "grids": full.get("Q", {}), "maps": full.get("maps", {})})

    # 3. tilt axis from align.com (never 0.0)
    axis, axis_src = _tilt_axis(data_root, basename, inv, input_cfg.get("geometry", {}))
    geom_measured["tilt_axis_angle_deg"] = round(axis, 4)
    geom_measured["tilt_axis_source"] = axis_src

    # 4. AUTHORITATIVE target reconstruction geometry (§4): never the raw/aligned
    # detector voxel count at the aligned/output pixel. Precedence: override ->
    # reconstruction header -> tilt.com THICKNESS -> fail.
    from . import imod_geometry as IG
    cfg_geom = input_cfg.get("geometry", {})
    imod_dir = str(Path(inv.tilt_com).parent) if inv.tilt_com else str(data_root)
    target = IG.resolve_target_geometry(
        reconstruction_path=inv.source_reconstruction,
        tilt_com_path=inv.tilt_com, newst_com_path=inv.newst_com, imod_dir=imod_dir,
        mdoc_path=inv.mdoc_file,
        aligned_shape_xyz=geom_measured.get("aligned_shape_xyz"),
        aligned_pixel_A=geom_measured.get("aligned_pixel_size_A"),
        raw_pixel_A=geom_measured.get("raw_pixel_size_A"),
        override_shape_xyz=cfg_geom.get("target_volume_shape_xyz"),
        override_pixel_A=cfg_geom.get("target_pixel_size_A"))
    geom_measured["target_volume_shape_xyz"] = target["shape_xyz"]
    geom_measured["target_volume_frame"] = (
        "imod_reconstruction_mrc_xyz__y_is_thickness"
    )
    geom_measured["target_pixel_size_A"] = target["pixel_size_A"]
    geom_measured["target_volume_physical_A"] = target["physical_size_A"]
    geom_measured["target_volume_source"] = target["source"]

    # 4. condition -> warp alignment mode (separate from refinement_mode)
    conds = base.conditions
    modes = {c: base.warp_mode(c) for c in conds}

    # 5. assemble the RESOLVED config
    sources = DISC.SourceInventory(basename=basename, data_dir=str(data_root))
    sources.__dict__.update({
        "raw_stack": inv.raw_stack, "aligned_stack": inv.aligned_stack,
        "final_xf": inv.final_xf, "tilt_file": inv.tilt_file, "raw_tilt_file": inv.raw_tilt_file,
        "xtilt_file": inv.xtilt_file, "tltxf_file": inv.tltxf_file, "defocus_file": inv.defocus_file,
        "mdoc_file": inv.mdoc_file, "newst_com": inv.newst_com, "tilt_com": inv.tilt_com,
        "ctf_com": inv.ctf_com, "source_reconstruction": inv.source_reconstruction})

    resolved = {
        "project": {"basename": basename, "schema_version": PC.SCHEMA_VERSION, "layout_version": 8},
        "paths": {"data_root": str(data_root), "output_dir": str(output_dir)},
        "input": {k: v for k, v in {
            "raw_stack": inv.raw_stack, "aligned_stack": inv.aligned_stack,
            "final_xf_file": inv.final_xf, "final_tilt_file": inv.tilt_file,
            "raw_tilt_file": inv.raw_tilt_file, "xtilt_file": inv.xtilt_file,
            "tltxf_file": inv.tltxf_file, "defocus_file": inv.defocus_file,
            "mdoc_file": inv.mdoc_file, "newst_com": inv.newst_com, "tilt_com": inv.tilt_com,
            "ctf_com": inv.ctf_com, "source_reconstruction": inv.source_reconstruction,
        }.items() if v},
        "geometry": {k: v for k, v in geom_measured.items() if v is not None},
        "conversion": {"initial_conditions": list(conds), "condition_modes": modes},
        "datasets": {
            "native_id": format_angpix(float(geom_measured["target_pixel_size_A"])),
            "native_pixel_size_A": float(geom_measured["target_pixel_size_A"]),
            "selected_id": format_angpix(float(geom_measured["target_pixel_size_A"])),
        },
        "multiresolution": {"extra_projection_binning": base.extra_projection_binning},
        "ctf": {"mode": base.ctf_mode},
        "missalignment": {"refinement_mode": base.refinement_mode,
                          "result_backend": base.result_backend},
        "cluster": {k: v for k, v in base.to_dict()["cluster"].items() if v is not None},
        "reconstruction": _reconstruction_contract(input_cfg, inv, geom_measured, output_dir),
        "provenance": {"resolved": True, "discovery": "provenance/source_inventory.json",
                       "geometry": "provenance/geometry_manifest.json",
                       "hashes": "provenance/source_hashes.json"},
    }
    reconstruction_cluster = ((input_cfg.get("cluster", {}) or {}).get("reconstruction_cluster") or {})
    if reconstruction_cluster:
        resolved["cluster"]["reconstruction_cluster"] = reconstruction_cluster
    warptools_reconstruction_cluster = ((input_cfg.get("cluster", {}) or {}).get("warptools_reconstruction_cluster") or {})
    if warptools_reconstruction_cluster:
        resolved["cluster"]["warptools_reconstruction_cluster"] = warptools_reconstruction_cluster

    # 6. validate the resolved config (geometry must be real; modes must be sane)
    rc = PC.from_dict(resolved)
    problems = PC.validate(rc, require_geometry=True, require_resolved=True)
    if problems:
        raise PC.ConfigError("resolved config failed validation:\n  - " + "\n  - ".join(problems))

    resolved_toml = output_dir / "project_settings.toml"
    write_toml(resolved_toml, resolved)
    return {"resolved_toml": str(resolved_toml), "resolved_dir": str(resolved_dir),
            "manifests_dir": str(man_dir), "config": rc, "warp_modes": modes,
            "tilt_axis": [axis, axis_src]}


def _atomic_json(path: Path, obj) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)
