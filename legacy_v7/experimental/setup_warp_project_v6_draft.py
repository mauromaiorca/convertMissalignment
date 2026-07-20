#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from v6 import SCHEMA_VERSION, SOFTWARE_VERSION  # noqa: E402
from v6.alignment_import import importer_for  # noqa: E402
from v6.config import (  # noqa: E402
    ProjectConfig,
    TiltSeriesConfig,
    resolve_cluster_profile,
    sha256_file,
    to_plain,
    write_toml,
)
from v6.jobs import generate_stage_jobs  # noqa: E402
from v6.relion_m import RelionExportContract, m_contract_for_source  # noqa: E402
from v6.sources import SourceDiscoveryError, TiltStackSourceAdapter, resolve_source  # noqa: E402
from v6.stages import StagePlanningError, plan_stages  # noqa: E402
from v6.warptools import WarpToolsAdapter  # noqa: E402
from v6.warp_project import SnapshotManager, V6Layout, WarpProjectRef, write_project_ref  # noqa: E402


def _atomic_json(path: Path, obj) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def _build_config(args, source) -> ProjectConfig:
    cluster, software, profile_notes = resolve_cluster_profile(args.cluster_profile)
    ts = TiltSeriesConfig(id=args.basename, basename=args.basename)
    ts.source.mode = source.mode
    ts.source.selected_reason = source.selected_reason
    ts.source.movies.directory = str(args.data_dir.resolve()) if source.mode == "movies" else ""
    ts.source.movies.pattern = "*"
    ts.source.movies.mdoc = source.mdoc
    ts.source.movies.gain = source.gain
    ts.source.stack.path = source.stack_path
    ts.source.stack.tilt_file = source.tilt_file
    ts.source.stack.mdoc = source.mdoc
    ts.source.stack.section_to_angle_known = bool(source.identity_table)
    ts.source.stack.acquisition_order_known = bool(source.observations.get("acquisition_order_known"))
    ts.imod.raw_stack = source.stack_path
    ts.imod.tlt = source.tilt_file
    if source.mode == "movies":
        try:
            stack_ref = TiltStackSourceAdapter().discover(args.data_dir.resolve(), args.basename)
            ts.source.stack.path = stack_ref.stack_path
            ts.source.stack.tilt_file = stack_ref.tilt_file
            ts.imod.raw_stack = stack_ref.stack_path
            ts.imod.tlt = stack_ref.tilt_file
            ts.capabilities.imod_alignment_available = True
        except SourceDiscoveryError:
            pass
    ts.binning.extra_projection_binning = args.extra_binning
    ts.warp.alignment_backend = args.alignment_backend
    ts.capabilities = source.capabilities
    if source.mode == "movies" and ts.imod.raw_stack:
        ts.capabilities.imod_alignment_available = True
    if args.condition == "raw_xf_affine_fixed" and args.alignment_backend == "legacy_affine" and ts.imod.raw_stack:
        ts.capabilities.imod_alignment_available = True
    ts.imod.raw_stack = source.raw_stack or source.stack_path
    ts.imod.aligned_stack = source.aligned_stack
    ts.imod.xf = source.final_xf
    ts.imod.tlt = source.tilt_file
    ts.imod.align_com = source.align_com
    ts.imod.raw_dimensions_xyz = list(source.raw_header.get("shape_xyz", []))
    ts.imod.aligned_dimensions_xyz = list(source.aligned_header.get("shape_xyz", []))
    ts.imod.raw_pixel_size_A = source.raw_header.get("pixel_size_A")
    ts.imod.aligned_pixel_size_A = source.aligned_header.get("pixel_size_A")
    ts.imod.tilt_count = len(source.identity_table)
    ts.imod.tilt_axis_angle_deg = source.tilt_axis_angle_deg
    ts.imod.tilt_axis_source = source.tilt_axis_source
    ts.imod.source_reconstruction = source.source_reconstruction
    ts.imod.target_volume_dimensions_xyz = list(source.target_geometry.get("shape_xyz", []))
    ts.imod.target_voxel_size_A = source.target_geometry.get("pixel_size_A")
    ts.imod.target_physical_dimensions_A = list(source.target_geometry.get("physical_size_A", []))
    ts.imod.target_geometry_source = source.target_geometry.get("source", "")
    ts.microscope.pixel_size_A = ts.imod.raw_pixel_size_A
    return ProjectConfig(
        schema_version=SCHEMA_VERSION,
        project={
            "name": args.basename,
            "output_dir": str(args.out_dir.resolve()),
            "condition": args.condition,
            "software_version": SOFTWARE_VERSION,
        },
        cluster=cluster,
        software=software,
        tilt_series=[ts],
        provenance={"resolved": True, "schema_version": SCHEMA_VERSION,
                    "software_version": SOFTWARE_VERSION,
                    "cluster_profile_notes": profile_notes},
    )


def _config_to_toml_dict(cfg: ProjectConfig) -> dict:
    data = to_plain(cfg)
    data["tilt_series"] = [to_plain(ts) for ts in cfg.tilt_series]
    return data


def _write_contracts(layout: V6Layout, cfg: ProjectConfig) -> None:
    ts = cfg.tilt_series[0]
    relion = RelionExportContract().to_dict()
    m = m_contract_for_source(ts.source.mode).to_dict()
    _atomic_json(layout.relion / "export_contract.json", relion)
    _atomic_json(layout.m / "input_contract.json", m)


def _tool_available(executable: str) -> bool:
    if not executable:
        return False
    path = Path(executable)
    return (path.is_file() and path.stat().st_size >= 0) or shutil.which(executable) is not None


def _stage_is_executable(stage, cfg: ProjectConfig, layout: V6Layout) -> tuple[bool, str]:
    if stage.stage_id == "10_warp_ingest":
        ok, reason, _syntax = WarpToolsAdapter(cfg.software.warptools_executable).supports_stack_only_ingest(
            layout.manifests / "warptools_probe.json"
        )
        return ok, reason
    return False, f"{stage.stage_id} is not generated until its parent snapshot is validated"


def setup(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Create a v6 canonical WarpTools project plan.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--basename", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--source-mode", choices=("auto", "movies", "tilt_stack"), default="auto")
    parser.add_argument("--condition", choices=("raw_xf_affine_fixed", "ali_identity", "raw_xf", "raw_identity"),
                        default="raw_xf_affine_fixed")
    parser.add_argument("--alignment-backend", choices=("legacy_affine", "warptools_native"),
                        default="legacy_affine")
    parser.add_argument("--extra-binning", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--cluster-profile", default="maxwell")
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not data_dir.is_dir():
        parser.error(f"--data-dir is not a directory: {data_dir}")

    try:
        source = resolve_source(data_dir, args.basename, args.source_mode, condition=args.condition)
        cfg = _build_config(args, source)
        stages = plan_stages(cfg)
    except (SourceDiscoveryError, StagePlanningError, ValueError) as exc:
        print(f"ERROR: setup failed: {exc}", file=sys.stderr)
        return 2

    layout = V6Layout(out_dir).create()
    settings = out_dir / "project_settings.toml"
    write_toml(settings, _config_to_toml_dict(cfg))
    toml_hash = sha256_file(settings)

    _atomic_json(layout.manifests / "config_provenance.json", {
        "absolute_toml_path": str(settings),
        "toml_sha256": toml_hash,
        "schema_version": SCHEMA_VERSION,
        "software_version": SOFTWARE_VERSION,
    })
    _atomic_json(layout.manifests / "source_inventory.json", source.to_dict())
    _atomic_json(layout.manifests / "capability_manifest.json", asdict(source.capabilities))
    _atomic_json(layout.manifests / "tilt_identity_table.json",
                 [asdict(x) for x in source.identity_table])

    ts = cfg.tilt_series[0]
    project_ref = WarpProjectRef(
        project_id=cfg.project["name"],
        tilt_series_id=ts.id,
        frame_series_settings_file=str(layout.warp / "ingest" / "frame_series" / "frame_series.settings"),
        frame_series_raw_directory=str(layout.warp / "ingest" / "frame_series"),
        tilt_series_settings_file=str(layout.warp / "base" / "tilt_series.settings"),
        tomostar_directory=str(layout.warp / "ingest" / "tomostar"),
        frame_processing_directory=str(layout.warp / "ingest" / "frame_processing"),
        tilt_series_processing_directory=str(layout.warp / "base" / "processing"),
        source_mode=ts.source.mode,
        capabilities=ts.capabilities,
        geometry_id="geometry_initial",
        ctf_id="ctf_unset",
        selection_id="selection_unset",
        parent_snapshot_id=None,
        toml_hash=toml_hash,
    )
    write_project_ref(layout, project_ref)
    SnapshotManager(layout, toml_hash).declare_snapshots(
        ["base", "alignment_initial", "pre_missalign", "missalign_smoke", "missalign_full"])

    importer = importer_for(args.alignment_backend)
    import_plan = importer.plan(toml=settings, project_dir=out_dir, source_xf=ts.imod.xf)
    _atomic_json(layout.manifests / "alignment_import_plan.json", import_plan.to_dict())
    _atomic_json(layout.manifests / "stage_plan.json", [stage.to_dict() for stage in stages])
    _write_contracts(layout, cfg)
    blocked: dict[str, str] = {}
    executable_stages = []
    for stage in stages:
        if stage.stage_id != "10_warp_ingest":
            continue
        ok, reason = _stage_is_executable(stage, cfg, layout)
        if not ok:
            blocked[stage.stage_id] = reason
        else:
            executable_stages.append(stage)
    written = generate_stage_jobs(
        jobs_dir=layout.jobs, run_dir=out_dir, settings_path=settings,
        toml_hash=toml_hash, cluster=cfg.cluster, stages=executable_stages)
    _atomic_json(layout.manifests / "job_graph.json", {
        "jobs": written,
        "blocked": blocked,
    })
    print(f"[setup] v6 project: {out_dir}")
    print(f"[setup] resolved TOML: {settings}")
    print(f"[setup] source mode: {ts.source.mode} ({ts.source.selected_reason})")
    if "10_warp_ingest" in written:
        print("[next] submit Warp ingest:")
        print(f"  sbatch {written['10_warp_ingest']}")
    else:
        print("[setup] project plan created, but Warp ingest is blocked:")
        print(f"  {blocked.get('10_warp_ingest', 'stage is not executable with the current resolved software/capability profile')}")
        print("[setup] no Slurm job should be submitted yet.")
    return 0


def main() -> int:
    return setup(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
