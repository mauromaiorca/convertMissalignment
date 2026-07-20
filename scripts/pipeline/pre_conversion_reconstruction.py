#!/usr/bin/env python3
"""Reconstruct the converted PRE geometry before running MissAlignment.

This is the version 8 imported-Warp geometry gate.  It consumes only the canonical converted Warp
project in ``RunLayout.training_dir`` and creates one diagnostic WarpTools
reconstruction.  It does not require, start, or inspect MissAlignment.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .runlayout import format_angpix
    from .reconstruction_validation import record_reconstruction_validation
    from .warptools_reconstruction import (
        WarpToolsReconstructionError,
        _as_array,
        _patch_xml,
        _run,
        _vector_attribute,
        _xml_root,
        atomic_json,
        choose_dose_values,
        exactly_one_root_xml,
        find_reconstruction,
        layout_for,
        load_settings,
        load_conversion_volume_contract,
        sha256_file,
    )
except ImportError:  # direct cluster execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.runlayout import format_angpix
    from pipeline.reconstruction_validation import record_reconstruction_validation
    from pipeline.warptools_reconstruction import (
        WarpToolsReconstructionError,
        _as_array,
        _patch_xml,
        _run,
        _vector_attribute,
        _xml_root,
        atomic_json,
        choose_dose_values,
        exactly_one_root_xml,
        find_reconstruction,
        layout_for,
        load_settings,
        load_conversion_volume_contract,
        sha256_file,
    )


@dataclass(frozen=True)
class PreConversionPlan:
    settings_path: Path
    run_dir: Path
    source_dir: Path
    source_xml: Path
    quantitative_stack: Path
    attempt_dir: Path
    work_dir: Path
    raw_data_dir: Path
    input_processing: Path
    output_processing: Path
    tomostar: Path
    warptools_settings: Path
    preparation_manifest: Path
    result_manifest: Path
    executable: str
    output_angpix: float | None
    device_list: str
    perdevice: int
    dataset_id: str
    public_reconstruction_dir: Path


def _next_attempt(layout) -> Path:
    root = layout.attempts_dir / "reconstruction" / layout.dataset_id / "warp_dataset"
    root.mkdir(parents=True, exist_ok=True)
    slurm_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if slurm_id:
        candidate = root / f"attempt_{slurm_id}"
        if candidate.exists():
            raise WarpToolsReconstructionError(
                f"attempt directory already exists: {candidate}"
            )
        return candidate
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for index in range(1, 1000):
        candidate = root / f"attempt_{stamp}_{index:03d}"
        if not candidate.exists():
            return candidate
    raise WarpToolsReconstructionError("could not allocate pre-conversion attempt")


def build_plan(
    settings_path: Path,
    *,
    output_angpix: float | None = None,
    device_list: str = "0",
    perdevice: int = 1,
    dataset_id: str | None = None,
) -> PreConversionPlan:
    cfg = load_settings(settings_path)
    layout = layout_for(cfg, dataset_id)
    source = layout.training_dir.resolve()
    if not source.is_dir():
        raise WarpToolsReconstructionError(
            f"converted Warp project missing: {source}; run the conversion first"
        )
    source_xml = exactly_one_root_xml(source, "pre-conversion source")
    series = source_xml.stem
    stack = source / "tiltstack" / series / f"{series}.st"
    if not stack.is_file() or stack.stat().st_size <= 0:
        raise WarpToolsReconstructionError(f"converted tilt stack missing: {stack}")

    attempt = _next_attempt(layout)
    work = attempt / "work"
    rec = cfg.get("reconstruction", {}) or {}
    wt = rec.get("warptools", {}) or {}
    cluster = cfg.get("cluster", {}) or {}
    executable = str(
        wt.get("executable")
        or cluster.get("warp_tools_executable")
        or "WarpTools"
    )
    configured_angpix = wt.get("output_angpix_A")
    selected = output_angpix
    if selected in (None, 0, 0.0) and configured_angpix not in (None, 0, 0.0):
        selected = float(configured_angpix)

    return PreConversionPlan(
        settings_path=Path(settings_path).resolve(),
        run_dir=layout.run_dir,
        source_dir=source,
        source_xml=source_xml.resolve(),
        quantitative_stack=stack.resolve(),
        attempt_dir=attempt,
        work_dir=work,
        raw_data_dir=work / "raw_data",
        input_processing=work / "input_pre_conversion",
        output_processing=attempt / "output_pre_conversion",
        tomostar=work / "raw_data" / f"{series}.tomostar",
        warptools_settings=work / "warp_tiltseries.settings",
        preparation_manifest=attempt / "preparation_manifest.json",
        result_manifest=attempt / "result_manifest.json",
        executable=executable,
        output_angpix=float(selected) if selected not in (None, 0, 0.0) else None,
        device_list=str(device_list),
        perdevice=int(perdevice),
        dataset_id=layout.dataset_id,
        public_reconstruction_dir=layout.warp_reconstructions_dir,
    )


def prepare_workspace(plan: PreConversionPlan) -> dict[str, Any]:
    try:
        import mrcfile
        import numpy as np
        from warpylib import TiltSeries
    except ImportError as exc:
        raise WarpToolsReconstructionError(
            "pre-conversion reconstruction requires mrcfile, numpy and warpylib"
        ) from exc

    for directory in (
        plan.raw_data_dir,
        plan.work_dir / "default_processing",
        plan.input_processing,
        plan.output_processing,
    ):
        directory.mkdir(parents=True, exist_ok=False)
    average_dir = plan.raw_data_dir / "average"
    average_dir.mkdir()

    ts = TiltSeries(str(plan.source_xml))
    angles = _as_array(ts.angles)
    axes = _as_array(ts.tilt_axis_angles)
    offsets_x = _as_array(ts.tilt_axis_offset_x)
    offsets_y = _as_array(ts.tilt_axis_offset_y)
    source_dose = _as_array(getattr(ts, "dose", None))
    root = _xml_root(plan.source_xml)
    volume = _vector_attribute(root, "VolumeDimensionsAngstrom")

    with mrcfile.mmap(plan.quantitative_stack, mode="r", permissive=True) as source:
        if source.data.ndim != 3:
            raise WarpToolsReconstructionError(
                f"expected 3-D tilt stack, got {source.data.shape}"
            )
        n_tilts, ny, nx = map(int, source.data.shape)
        pixel_x = float(source.voxel_size.x)
        pixel_y = float(source.voxel_size.y)
        if pixel_x <= 0 or pixel_y <= 0:
            raise WarpToolsReconstructionError("invalid stack pixel size")
        if abs(pixel_x - pixel_y) > max(1e-4, pixel_x * 1e-5):
            raise WarpToolsReconstructionError(
                f"anisotropic projection pixels unsupported: {pixel_x}, {pixel_y}"
            )
        for label, values in (
            ("tilt angles", angles),
            ("axis angles", axes),
            ("offset X", offsets_x),
            ("offset Y", offsets_y),
        ):
            if len(values) != n_tilts:
                raise WarpToolsReconstructionError(
                    f"{label}: {len(values)} values for {n_tilts} sections"
                )

        dose, dose_policy, source_dose_ok, _ = choose_dose_values(
            source_dose, source_dose, n_tilts
        )
        series = plan.source_xml.stem
        movie_names: list[str] = []
        for index in range(n_tilts):
            name = f"{series}_tilt_{index:04d}.mrc"
            output = plan.raw_data_dir / name
            temporary = plan.raw_data_dir / f".{name}.tmp"
            with mrcfile.new(temporary, overwrite=True) as destination:
                destination.set_data(
                    np.asarray(source.data[index], dtype=np.float32)
                )
                destination.voxel_size = (pixel_x, pixel_y, pixel_x)
                destination.update_header_stats()
            os.replace(temporary, output)
            (average_dir / name).symlink_to(Path("..") / name)
            movie_names.append(name)

    (plan.raw_data_dir / "averages").symlink_to("average", target_is_directory=True)

    conversion_contract = load_conversion_volume_contract(
        plan.source_dir,
        plan.source_xml.stem,
        xml_volume_dimensions_A=volume,
    )
    volume_shape_xyz = tuple(conversion_contract["shape_warp_xyz"])
    dimensions = "x".join(str(value) for value in volume_shape_xyz)

    prepared_xml = plan.input_processing / plan.source_xml.name
    _patch_xml(
        plan.source_xml,
        prepared_xml,
        raw_data_dir=plan.raw_data_dir,
        movie_names=movie_names,
        dose_values=dose,
    )

    with plan.tomostar.open("w", encoding="utf-8") as handle:
        handle.write("data_\n\nloop_\n")
        handle.write("_wrpMovieName #1\n")
        handle.write("_wrpAngleTilt #2\n")
        handle.write("_wrpAxisAngle #3\n")
        handle.write("_wrpAxisOffsetX #4\n")
        handle.write("_wrpAxisOffsetY #5\n")
        handle.write("_wrpDose #6\n")
        for index, name in enumerate(movie_names):
            handle.write(
                f"{name} {angles[index]:.8f} {axes[index]:.8f} "
                f"{offsets_x[index]:.8f} {offsets_y[index]:.8f} "
                f"{dose[index]:.12g}\n"
            )

    manifest = {
        "schema_version": 3,
        "purpose": "Warp dataset geometry validation before MissAlignment",
        "source_warp_project": str(plan.source_dir),
        "source_xml": str(plan.source_xml),
        "source_xml_sha256": sha256_file(plan.source_xml),
        "quantitative_stack": str(plan.quantitative_stack),
        "quantitative_stack_sha256": sha256_file(plan.quantitative_stack),
        "prepared_xml": str(prepared_xml),
        "n_tilts": n_tilts,
        "stack_shape_zyx": [n_tilts, ny, nx],
        "input_pixel_size_A": pixel_x,
        "projection_dimensions_xy_A": [nx * pixel_x, ny * pixel_y],
        "volume_dimensions_A_xyz": [float(value) for value in volume],
        "conversion_volume_contract": conversion_contract,
        "tomo_dimensions_xyz": list(volume_shape_xyz),
        "tomo_dimensions_argument": dimensions,
        "tomo_dimensions_source": (
            "explicit current Warp XYZ shape from the conversion manifest; "
            "legacy axis inference is forbidden"
        ),
        "dose_policy": dose_policy,
        "source_dose_usable": source_dose_ok,
        "allowed_uses": ["visual geometry validation", "conversion acceptance"],
        "forbidden_uses": ["FSC", "resolution estimation", "final refinement"],
    }
    atomic_json(plan.preparation_manifest, manifest)
    return manifest


def run_reconstruction(plan: PreConversionPlan) -> dict[str, Any]:
    preparation = prepare_workspace(plan)
    input_angpix = float(preparation["input_pixel_size_A"])
    output_angpix = float(plan.output_angpix or input_angpix)
    if not math.isfinite(output_angpix) or output_angpix <= 0:
        raise WarpToolsReconstructionError(f"invalid output pixel size {output_angpix}")
    if output_angpix < input_angpix - 1e-6:
        raise WarpToolsReconstructionError("pre-conversion reconstruction would upsample")

    executable = shutil.which(plan.executable) or plan.executable
    if not Path(executable).is_file() and shutil.which(executable) is None:
        raise WarpToolsReconstructionError(f"WarpTools not found: {plan.executable}")

    _run(
        [
            executable,
            "create_settings",
            "--output",
            plan.warptools_settings.name,
            "--folder_processing",
            "default_processing",
            "--folder_data",
            "raw_data",
            "--extension",
            "*.tomostar",
            "--angpix",
            f"{input_angpix:.12g}",
            "--tomo_dimensions",
            str(preparation["tomo_dimensions_argument"]),
        ],
        cwd=plan.work_dir,
        log_path=plan.attempt_dir / "create_settings.log",
    )

    _run(
        [
            executable,
            "ts_reconstruct",
            "--settings",
            str(plan.warptools_settings),
            "--input_data",
            str(plan.tomostar),
            "--input_processing",
            str(plan.input_processing),
            "--output_processing",
            str(plan.output_processing),
            "--angpix",
            f"{output_angpix:.12g}",
            "--device_list",
            plan.device_list,
            "--perdevice",
            str(plan.perdevice),
            "--dont_invert",
            "--dont_normalize",
            "--dont_mask",
        ],
        cwd=plan.work_dir,
        log_path=plan.attempt_dir / "pre_conversion_reconstruct.log",
    )
    volume = find_reconstruction(plan.output_processing)
    result = {
        "schema_version": 1,
        "status": "completed",
        "purpose": "Warp dataset geometry validation before MissAlignment",
        "reconstruction": str(volume),
        "reconstruction_size": volume.stat().st_size,
        "output_pixel_size_A": output_angpix,
        "preparation_manifest": str(plan.preparation_manifest),
        "acceptance_state": "technical_validation_pending",
        "quantitative_warning": (
            "Diagnostic reconstruction only. Do not use for FSC or final quantitative validation."
        ),
    }
    atomic_json(plan.result_manifest, result)
    latest = plan.attempt_dir.parent / "latest_success"
    temporary = latest.with_name(latest.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(plan.attempt_dir.name, target_is_directory=True)
    os.replace(temporary, latest)

    public = plan.public_reconstruction_dir / format_angpix(output_angpix)
    public.mkdir(parents=True, exist_ok=True)
    published_map = public / volume.name
    if published_map.is_symlink() or published_map.exists():
        published_map.unlink()
    published_map.symlink_to(os.path.relpath(volume, start=public))
    for preview in volume.parent.glob("*.png"):
        destination = public / preview.name
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        destination.symlink_to(os.path.relpath(preview, start=public))
    public_manifest = dict(result)
    public_manifest.update({
        "dataset_id": plan.dataset_id,
        "reconstruction": str(published_map),
        "internal_attempt": str(plan.attempt_dir),
    })
    atomic_json(public / "manifest.json", public_manifest)
    result["published_reconstruction"] = str(published_map)
    result["public_manifest"] = str(public / "manifest.json")
    atomic_json(plan.result_manifest, result)

    validation = record_reconstruction_validation(
        layout_for(load_settings(plan.settings_path), plan.dataset_id),
        level="technical",
        note=(
            "Automatically validated after WarpTools completed successfully and "
            "the reconstruction output and manifests passed filesystem checks."
        ),
    )
    result = dict(result)
    result["acceptance_state"] = "technically_validated"
    result["acceptance"] = validation
    return result


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--project-settings", type=Path, required=True)
    run.add_argument("--output-angpix", type=float, default=0.0)
    run.add_argument("--dataset", default=None)
    run.add_argument("--device-list", default="0")
    run.add_argument("--perdevice", type=int, default=1)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        plan = build_plan(
            args.project_settings,
            output_angpix=args.output_angpix,
            device_list=args.device_list,
            perdevice=args.perdevice,
            dataset_id=args.dataset,
        )
        print(f"[pre-conversion] source:  {plan.source_dir}")
        print(f"[pre-conversion] attempt: {plan.attempt_dir}")
        result = run_reconstruction(plan)
        print(f"[pre-conversion] map: {result['reconstruction']}")
        print(f"[pre-conversion] manifest: {plan.result_manifest}")
        return 0
    except WarpToolsReconstructionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
