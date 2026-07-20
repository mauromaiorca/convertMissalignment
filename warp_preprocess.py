#!/usr/bin/env python3
"""Create a resampled Warp tilt-series dataset and its dataset-specific batches.

The public command plans a derived dataset. The generated Slurm preprocessing
batch calls the private ``run`` command, which performs anti-aliased detector
resampling with IMOD ``newstack -shrink``. Geometry and physical coordinates are
preserved; the source dataset is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline import jobs as JOBS  # noqa: E402
from pipeline import project_config as PC  # noqa: E402
from pipeline.project_publish import atomic_json, hash_file, publish_warp_dataset  # noqa: E402
from pipeline.runlayout import (  # noqa: E402
    RunLayout,
    dataset_id_from_config,
    format_angpix,
    parse_angpix_id,
)


def _load_settings(project: Path) -> tuple[Path, dict[str, Any]]:
    project = project.expanduser().resolve()
    settings = project if project.name.endswith(".toml") else project / "project_settings.toml"
    if not settings.is_file():
        raise FileNotFoundError(f"project settings not found: {settings}")
    with settings.open("rb") as handle:
        cfg = tomllib.load(handle)
    if not bool((cfg.get("provenance", {}) or {}).get("resolved")):
        raise ValueError("warp_preprocess requires a resolved v8 project_settings.toml")
    return settings, cfg


def _identity(cfg: dict[str, Any]) -> tuple[str, str, str, Path]:
    project = cfg.get("project", {}) or {}
    conversion = cfg.get("conversion", {}) or {}
    ma = cfg.get("missalignment", {}) or {}
    paths = cfg.get("paths", {}) or {}
    conditions = conversion.get("initial_conditions") or ["ali_identity"]
    if len(conditions) != 1:
        raise ValueError("v8 requires exactly one conversion condition per project")
    return (
        str(project.get("basename") or project.get("name") or "series"),
        str(conditions[0]),
        str(ma.get("refinement_mode") or "standard"),
        Path(paths.get("output_dir") or ".").resolve(),
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {path}")
    return json.loads(path.read_text())


def _update_project_records(project_root: Path, dataset_id: str, manifest: Path, status: str) -> None:
    registry_path = project_root / "provenance" / "artifact_registry.json"
    registry: dict[str, Any] = {"schema_version": 1, "artifacts": {}}
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text())
    raw_artifacts = registry.get("artifacts") or {}
    if isinstance(raw_artifacts, list):
        # Transitional compatibility with early v8 development projects.
        artifacts = {
            str(item.get("dataset_id") or item.get("artifact_id") or f"artifact_{index}"): item
            for index, item in enumerate(raw_artifacts)
            if isinstance(item, dict)
        }
    else:
        artifacts = dict(raw_artifacts)
    artifacts[dataset_id] = {
        "artifact_type": "warp_tilt_series_dataset",
        "dataset_id": dataset_id,
        "manifest": str(manifest),
        "status": status,
    }
    registry.update({"schema_version": 1, "artifacts": artifacts})
    atomic_json(registry_path, registry)

    status_path = project_root / "project_status.json"
    project_status = {"schema_version": 1, "layout_version": 8, "status": "prepared", "datasets": {}}
    if status_path.is_file():
        project_status = json.loads(status_path.read_text())
    datasets = dict(project_status.get("datasets") or {})
    datasets[dataset_id] = {"status": status, "manifest": str(manifest)}
    project_status["datasets"] = datasets
    if status in {"complete", "validated"}:
        project_status["selected_dataset_id"] = dataset_id
    atomic_json(status_path, project_status)


def _stack_from_project(project: Path) -> Path:
    stacks = sorted((project / "tiltstack").glob("*/*.st"))
    if len(stacks) != 1:
        raise RuntimeError(f"expected one tilt stack in {project / 'tiltstack'}, found {len(stacks)}")
    return stacks[0]


def _mrc_info(path: Path) -> dict[str, Any]:
    import mrcfile
    with mrcfile.mmap(path, mode="r", permissive=True) as handle:
        if handle.data.ndim != 3:
            raise ValueError(f"expected a 3-D tilt stack, got shape {handle.data.shape}")
        nz, ny, nx = map(int, handle.data.shape)
        px = float(handle.voxel_size.x)
        py = float(handle.voxel_size.y)
    if px <= 0 or py <= 0:
        raise ValueError(f"invalid MRC pixel size for {path}: {px}, {py}")
    return {"shape_zyx": [nz, ny, nx], "pixel_size_A": px, "pixel_size_y_A": py}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_dataset_toml(layout: RunLayout, *, source_id: str, factor: float,
                        source_pixel: float, target_pixel: float, status: str) -> None:
    layout.dataset_config.parent.mkdir(parents=True, exist_ok=True)
    layout.dataset_config.write_text(
        "[dataset]\n"
        f'id = "{layout.dataset_id}"\n'
        f'status = "{status}"\n'
        f'pixel_size_A = {target_pixel!r}\n'
        f'source_id = "{source_id}"\n'
        "\n[preprocessing]\n"
        'operation = "detector_resampling"\n'
        'method = "imod_newstack_shrink"\n'
        f'factor = {factor!r}\n'
        f'source_pixel_size_A = {source_pixel!r}\n'
        f'target_pixel_size_A = {target_pixel!r}\n'
        'coordinate_policy = "physical coordinates preserved"\n'
    )


def _missalignment_commands(cfg: dict[str, Any], layout: RunLayout) -> tuple[str, str]:
    from pipeline.orchestrate import _load_03run
    r3 = _load_03run()
    layout.config_dir.mkdir(parents=True, exist_ok=True)
    smoke_yaml = layout.config_dir / "config.smoke.yaml"
    full_yaml = layout.config_yaml
    smoke_yaml.write_text(r3.config_text(layout.smoke_warp_dir, "smoke"))
    full_yaml.write_text(r3.config_text(layout.full_warp_dir, layout.refinement_mode))
    ma = cfg.get("missalignment", {}) or {}
    executable = str(ma.get("executable", "miss-alignment"))
    training_devices = str(ma.get("training_devices", "0"))
    reconstruction_devices = str(ma.get("reconstruction_devices", "0"))
    dataloaders = int(ma.get("dataloaders_per_trainer", 1))
    smoke = r3.shell_quote(r3.missalignment_command(
        config_path=smoke_yaml,
        training_devices=training_devices,
        reconstruction_devices=reconstruction_devices,
        dataloaders_per_trainer=dataloaders,
        prepare_stacks=None,
        start_at_iteration=0,
        executable=executable,
    ))
    full = r3.shell_quote(r3.missalignment_command(
        config_path=full_yaml,
        training_devices=training_devices,
        reconstruction_devices=reconstruction_devices,
        dataloaders_per_trainer=dataloaders,
        prepare_stacks=None,
        start_at_iteration=0,
        executable=executable,
    ))
    return smoke, full


def plan(args: argparse.Namespace) -> int:
    settings, cfg = _load_settings(args.project)
    basename, condition, mode, project_root = _identity(cfg)
    source_id = args.source or dataset_id_from_config(cfg)
    source_layout = RunLayout.from_settings(
        out_dir=project_root, basename=basename, condition=condition,
        refinement_mode=mode, dataset_id=source_id,
    )
    source_manifest = _read_manifest(source_layout.dataset_manifest)
    if source_manifest.get("status") not in {"complete", "validated"}:
        raise RuntimeError(f"source dataset {source_id} is not complete")
    source_pixel = float(source_manifest.get("pixel_size_A") or parse_angpix_id(source_id))
    source_shape = source_manifest.get("stack_shape_zyx") or []

    if args.bin is not None:
        factor = float(args.bin)
        if factor < 2 or not factor.is_integer():
            raise ValueError("--bin must be an integer >= 2")
        if len(source_shape) == 3:
            _, ny, nx = map(int, source_shape)
            if nx % int(factor) or ny % int(factor):
                raise ValueError(
                    f"source detector dimensions {nx}x{ny} are not divisible by {int(factor)}; "
                    "crop explicitly or choose another factor to avoid coordinate-centre drift"
                )
        target_pixel = source_pixel * factor
    else:
        target_pixel = float(args.target_angpix)
        if target_pixel <= source_pixel:
            raise ValueError("--target-angpix must be larger than the source pixel size")
        factor = target_pixel / source_pixel
        nearest = round(factor)
        if nearest < 2 or abs(factor - nearest) > 1e-6:
            raise ValueError(
                "version 8 alpha1 supports exact integer detector reductions only; "
                "choose --bin N or a target pixel size equal to source_pixel_size × N"
            )
        factor = float(nearest)
        target_pixel = source_pixel * factor

    target_id = format_angpix(target_pixel)
    if target_id == source_id:
        raise ValueError("target dataset equals source dataset")
    target = RunLayout.from_settings(
        out_dir=project_root, basename=basename, condition=condition,
        refinement_mode=mode, dataset_id=target_id,
    ).create()

    existing = None
    if target.dataset_manifest.is_file():
        existing = json.loads(target.dataset_manifest.read_text())
        old = ((existing.get("preprocessing") or {}).get("factor"))
        if existing.get("source_artifact_id") != source_manifest.get("artifact_id") or (
            old is not None and abs(float(old) - factor) > 1e-8
        ):
            raise RuntimeError(
                f"{target.warp_dataset_dir} already describes a different dataset; choose another target"
            )

    plan_path = target.state_dir / f"preprocess_{target_id}.json"
    record = {
        "schema_version": 1,
        "status": "planned",
        "operation": "detector_resampling",
        "method": "imod_newstack_shrink",
        "source_dataset_id": source_id,
        "source_dataset_manifest": str(source_layout.dataset_manifest),
        "source_artifact_id": source_manifest.get("artifact_id"),
        "source_pixel_size_A": source_pixel,
        "source_shape_zyx": source_shape,
        "factor": factor,
        "target_dataset_id": target_id,
        "target_pixel_size_A": target_pixel,
        "target_dataset_manifest": str(target.dataset_manifest),
        "settings": str(settings),
        "coordinate_policy": "detector pixels resampled; physical coordinates and Warp volume extents preserved",
        "half_set_policy": "unchanged",
    }
    atomic_json(plan_path, record)
    atomic_json(target.dataset_manifest, {
        "schema_version": 1,
        "artifact_type": "warp_tilt_series_dataset",
        "artifact_id": None,
        "source_artifact_id": source_manifest.get("artifact_id"),
        "dataset_id": target_id,
        "pixel_size_A": target_pixel,
        "preprocessing": record,
        "status": "planned",
    })
    _write_dataset_toml(target, source_id=source_id, factor=factor,
                        source_pixel=source_pixel, target_pixel=target_pixel, status="planned")

    smoke_cmd, full_cmd = _missalignment_commands(cfg, target)
    cluster = PC.from_dict(cfg).cluster
    reconstruction = dict(cfg.get("reconstruction", {}) or {})
    reconstruction["cluster"] = ((cfg.get("cluster", {}) or {}).get("reconstruction_cluster") or {})
    reconstruction["warptools_cluster"] = ((cfg.get("cluster", {}) or {}).get("warptools_reconstruction_cluster") or {})
    run_command = (
        f'"$PIPELINE_PYTHON" {shlex.quote(str(Path(__file__).resolve()))} run '
        f"--project {shlex.quote(str(settings))} --source {shlex.quote(source_id)} "
        f"--target {shlex.quote(target_id)} --factor {factor!r}"
    )
    written = JOBS.generate_jobs(
        target,
        profile=str((cfg.get("cluster", {}) or {}).get("profile", "maxwell")),
        ma_command=full_cmd,
        smoke_command=smoke_cmd,
        run_script=str(target.config_dir / "run_missalignment.sh"),
        settings_path=str(settings),
        cluster=cluster,
        reconstruction_config=reconstruction,
        include_import=False,
        preprocess_command=run_command,
    )
    atomic_json(target.missalignment_run_dir / "input" / "selected_dataset.json", {
        "schema_version": 1,
        "dataset_id": target_id,
        "dataset_manifest": str(target.dataset_manifest),
        "source_dataset_id": source_id,
        "selection_policy": "explicit pixel-size dataset",
        "status": "planned",
    })
    atomic_json(target.manifest("result_manifest.json"), {
        "schema_version": 1,
        "result_backend": PC.from_dict(cfg).result_backend,
        "condition": condition,
        "refinement_mode": mode,
        "dataset_id": target_id,
        "training_directory": str(target.full_warp_dir),
        "pre_missalign_directory": str(target.pre_missalign_dir),
        "smoke_directory": str(target.smoke_warp_dir),
        "initial_xml": None,
        "final_xml": None,
        "final_iteration": None,
    })
    atomic_json(target.missalignment_run_dir / "generated_batches.json", written)
    _update_project_records(project_root, target_id, target.dataset_manifest, "planned")

    print(f"[preprocess] planned dataset: {target_id}")
    print(f"[preprocess] source: {source_id} ({source_pixel:g} A/px)")
    print(f"[preprocess] target: {target_pixel:g} A/px")
    print(f"[preprocess] run: sbatch {target.batch_path('warp_data', 'preprocess.sbatch')}")
    print(f"[preprocess] then inspect: sbatch {target.batch_path('warp_data', 'reconstruct.sbatch')}")
    return 0


def _copy_project_metadata(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in {"tiltstack", "_converted.marker", "conversion_validation.json"}:
            continue
        destination = target / item.name
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if item.is_dir():
            shutil.copytree(item, destination, symlinks=True)
        elif item.is_symlink():
            destination.symlink_to(os.readlink(item))
        else:
            shutil.copy2(item, destination)


def execute(args: argparse.Namespace) -> int:
    settings, cfg = _load_settings(args.project)
    basename, condition, mode, project_root = _identity(cfg)
    source = RunLayout.from_settings(
        out_dir=project_root, basename=basename, condition=condition,
        refinement_mode=mode, dataset_id=args.source,
    )
    target = RunLayout.from_settings(
        out_dir=project_root, basename=basename, condition=condition,
        refinement_mode=mode, dataset_id=args.target,
    ).create()
    plan_path = target.state_dir / f"preprocess_{target.dataset_id}.json"
    plan_record = _read_manifest(plan_path)
    if abs(float(plan_record["factor"]) - float(args.factor)) > 1e-8:
        raise RuntimeError("requested factor does not match the recorded preprocessing plan")
    source_manifest = _read_manifest(source.dataset_manifest)
    source_project = source.training_dir.resolve()
    target_project = target.training_dir.resolve()
    if not source_project.is_dir() or not (source_project / "_converted.marker").is_file():
        raise RuntimeError(f"source Warp dataset is not converted: {source_project}")
    if (target_project / "_converted.marker").is_file() and not args.force:
        print(f"[preprocess] already complete: {target.dataset_id}")
        return 0

    source_stack = _stack_from_project(source_project)
    source_info = _mrc_info(source_stack)
    ntilts, ny, nx = source_info["shape_zyx"]
    factor = float(args.factor)
    if factor.is_integer() and (nx % int(factor) or ny % int(factor)):
        raise RuntimeError(f"source detector dimensions {nx}x{ny} are not divisible by {int(factor)}")

    if target_project.exists():
        for child in target_project.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    target_project.mkdir(parents=True, exist_ok=True)
    _copy_project_metadata(source_project, target_project)

    relative_stack = source_stack.relative_to(source_project / "tiltstack")
    target_stack = target_project / "tiltstack" / relative_stack
    target_stack.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["newstack", "-input", str(source_stack), "-output", str(target_stack),
           "-shrink", str(factor), "-float", "0"]
    log_path = target.log_dir("warp_data") / "preprocess_newstack.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(cp.stdout or "")
    if cp.returncode != 0:
        raise RuntimeError(f"newstack failed with return code {cp.returncode}; see {log_path}")

    target_info = _mrc_info(target_stack)
    target_pixel = float(target_info["pixel_size_A"])
    expected = float(source_info["pixel_size_A"]) * factor
    if abs(target_pixel - expected) > max(1e-4, expected * 1e-4):
        raise RuntimeError(f"newstack output pixel size {target_pixel:g} != expected {expected:g} A/px")
    if target_info["shape_zyx"][0] != ntilts:
        raise RuntimeError("preprocessing changed the number of tilt images")

    conversion_files = sorted(target_project.glob("*.conversion.json"))
    if len(conversion_files) == 1:
        conversion = json.loads(conversion_files[0].read_text())
        conversion.update({
            "derived_dataset": True,
            "source_dataset_id": source.dataset_id,
            "source_conversion_manifest": str(source_project / conversion_files[0].name),
            "output_stack": str(target_stack),
            "output_pixel_size_A": target_pixel,
            "image_shape_zyx": target_info["shape_zyx"],
        })
        history = list(conversion.get("resampling_history") or [])
        history.append({
            "operation": "detector_resampling",
            "method": "imod_newstack_shrink",
            "factor": factor,
            "input_pixel_size_A": source_info["pixel_size_A"],
            "output_pixel_size_A": target_pixel,
            "output_shape_yx": target_info["shape_zyx"][1:],
        })
        conversion["resampling_history"] = history
        atomic_json(conversion_files[0], conversion)

    source_stack_digest, source_stack_hash_mode = hash_file(source_stack)
    preprocess = {
        "operation": "detector_resampling",
        "method": "imod_newstack_shrink",
        "factor": factor,
        "command": cmd,
        "source_dataset_id": source.dataset_id,
        "source_artifact_id": source_manifest.get("artifact_id"),
        "source_stack": str(source_stack),
        "source_stack_sha256": source_stack_digest,
        "source_stack_hash_mode": source_stack_hash_mode,
        "source_shape_zyx": source_info["shape_zyx"],
        "source_pixel_size_A": source_info["pixel_size_A"],
        "target_stack": str(target_stack),
        "target_shape_zyx": target_info["shape_zyx"],
        "target_pixel_size_A": target_pixel,
        "centre_policy": "IMOD newstack centred shrink; source dimensions validated for exact integer divisibility",
        "coordinate_transform": {
            "units": "detector_pixels",
            "target_to_source_xy": [[factor, 0.0, (factor - 1.0) / 2.0],
                                    [0.0, factor, (factor - 1.0) / 2.0],
                                    [0.0, 0.0, 1.0]],
        },
        "physical_coordinates_preserved": True,
        "half_set_policy": "unchanged",
        "software": {"newstack": shutil.which("newstack")},
    }
    atomic_json(target_project / "preprocessing.json", preprocess)
    atomic_json(target_project / "conversion_validation.json", {
        "schema_version": 3,
        "derived_dataset": True,
        "source_dataset_id": source.dataset_id,
        "target_dataset_id": target.dataset_id,
        "pixel_size_A": target_pixel,
        "image_shape_zyx": target_info["shape_zyx"],
        "geometry_source": str(source.dataset_manifest),
        "volume_frame_contract_version": 2,
    })
    (target_project / "_converted.marker").write_text("ok\n")
    publish_warp_dataset(
        target,
        source_artifact_id=source_manifest.get("artifact_id"),
        preprocessing=preprocess,
    )
    _write_dataset_toml(target, source_id=source.dataset_id, factor=factor,
                        source_pixel=float(source_info["pixel_size_A"]),
                        target_pixel=target_pixel, status="complete")
    plan_record.update({"status": "complete", "command": cmd,
                        "target_shape_zyx": target_info["shape_zyx"],
                        "actual_target_pixel_size_A": target_pixel})
    atomic_json(plan_path, plan_record)
    selected = target.missalignment_run_dir / "input" / "selected_dataset.json"
    if selected.is_file():
        data = json.loads(selected.read_text())
        data["status"] = "complete"
        data["artifact_id"] = json.loads(target.dataset_manifest.read_text()).get("artifact_id")
        atomic_json(selected, data)
    _update_project_records(project_root, target.dataset_id, target.dataset_manifest, "complete")
    print(f"[preprocess] complete: {target.warp_dataset_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan a resampled Warp dataset and generate its Slurm batches.")
    sub = parser.add_subparsers(dest="command")

    public = sub.add_parser("plan", help="plan a derived pixel-size dataset")
    public.add_argument("--project", type=Path, required=True,
                        help="project directory or project_settings.toml")
    public.add_argument("--source", help="source dataset ID, e.g. 5.45Apx")
    group = public.add_mutually_exclusive_group(required=True)
    group.add_argument("--bin", type=int, help="integer detector shrink factor")
    group.add_argument("--target-angpix", type=float, help="target detector pixel size in A/px")
    public.set_defaults(func=plan)

    run = sub.add_parser("run", help="internal executor used by generated batches")
    run.add_argument("--project", type=Path, required=True)
    run.add_argument("--source", required=True)
    run.add_argument("--target", required=True)
    run.add_argument("--factor", type=float, required=True)
    run.add_argument("--force", action="store_true")
    run.set_defaults(func=execute)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # User-facing shorthand requested for v8: warp_preprocess.py --project ... --bin 3
    if argv and argv[0] not in {"plan", "run", "-h", "--help"}:
        argv.insert(0, "plan")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
