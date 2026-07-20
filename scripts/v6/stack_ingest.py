from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .config import ProjectConfig, sha256_file
from .mrc import validate_stack
from .stage_result import atomic_json, write_result
from .warptools import WarpToolsAdapter
from .warp_project import SnapshotManager, V6Layout, WarpProjectRef, write_project_ref


def run(cfg: ProjectConfig, *, settings: Path, run_dir: Path, toml_hash: str) -> None:
    ts = cfg.tilt_series[0]
    if ts.source.mode != "tilt_stack":
        raise RuntimeError("10_warp_ingest currently implements stack-only input only")
    source = Path(ts.source.stack.path)
    tilt_file = Path(ts.source.stack.tilt_file)
    if not tilt_file.is_file():
        raise RuntimeError(f"tilt file missing: {tilt_file}")
    angles = [float(x.split()[0]) for x in tilt_file.read_text().splitlines() if x.strip()]
    header = validate_stack(source, expected_tilts=len(angles))
    layout = V6Layout(run_dir).create()
    working_stack, working_header, binning_record = _working_stack(
        source=source,
        header=header,
        factor=int(ts.binning.extra_projection_binning),
        run_dir=run_dir,
        series_id=ts.id,
        expected_tilts=len(angles),
    )
    adapter = WarpToolsAdapter(cfg.software.warptools_executable)
    ok, reason, syntax = adapter.supports_stack_only_ingest(layout.manifests / "warptools_probe.json")
    if not ok:
        raise RuntimeError(reason)

    project_dir = layout.warp / "base"
    ingest = adapter.run_stack_only_ingest(
        input_stack=working_stack,
        tilt_file=tilt_file,
        output_dir=project_dir,
        series_id=ts.id,
        pixel_size_A=working_header.pixel_size_A,
    )
    tilt_settings = layout.warp / "base" / "tilt_series.settings"
    frame_settings = layout.warp / "base" / "frame_series.settings"
    tomostar = layout.warp / "base" / "tomostar" / f"{ts.id}.tomostar"
    xml = layout.warp / "base" / f"{ts.id}.xml"
    processing = layout.warp / "base" / "processing"
    _validate_warp_outputs(
        frame_settings=frame_settings,
        tilt_settings=tilt_settings,
        tomostar=tomostar,
        xml=xml,
        processing=processing,
    )
    mapping = [
        {
            "tilt_id": f"tilt_{i:04d}",
            "source_section": i,
            "tilt_angle_deg": angle,
        }
        for i, angle in enumerate(angles)
    ]
    atomic_json(layout.manifests / "stack_ingest_section_mapping.json", mapping)
    atomic_json(layout.manifests / "working_stack_manifest.json", binning_record)

    ref = WarpProjectRef(
        project_id=cfg.project["name"],
        tilt_series_id=ts.id,
        frame_series_settings_file=str(frame_settings),
        frame_series_raw_directory=str(working_stack.parent),
        tilt_series_settings_file=str(tilt_settings),
        tomostar_directory=str(tomostar.parent),
        frame_processing_directory=str(layout.warp / "ingest" / "frame_processing"),
        tilt_series_processing_directory=str(processing),
        source_mode="tilt_stack",
        capabilities=ts.capabilities,
        geometry_id="geometry_source_stack",
        ctf_id="ctf_unset",
        selection_id="selection_unset",
        parent_snapshot_id=None,
        toml_hash=toml_hash,
    )
    write_project_ref(layout, ref)
    snap = SnapshotManager(layout, toml_hash).create_snapshot(
        "base",
        parent_snapshot_id=None,
        copy_files=[frame_settings, tilt_settings, tomostar, xml],
        link_files=[source],
        geometry_id="geometry_source_stack",
    )
    details = {
        "snapshot_id": snap.snapshot_id,
        "validated": True,
        "expected_outputs": [str(frame_settings), str(tilt_settings), str(tomostar), str(xml)],
        "tilt_count": len(angles),
        "warptools": syntax.to_dict(),
        "warptools_commands": ingest["commands"],
        "working_stack": binning_record,
    }
    write_result(run_dir=run_dir, stage_id="10_warp_ingest", status="validated", details=details)
    _maybe_generate_alignment_job(cfg, settings=settings, run_dir=run_dir, toml_hash=toml_hash)


def _working_stack(
    *,
    source: Path,
    header,
    factor: int,
    run_dir: Path,
    series_id: str,
    expected_tilts: int,
) -> tuple[Path, object, dict]:
    if factor not in (1, 2, 4, 8):
        raise RuntimeError(f"extra_projection_binning must be one of 1,2,4,8; got {factor}")
    if factor == 1:
        return source, header, {
            "factor": 1,
            "source_stack": str(source),
            "source_stack_hash": sha256_file(source),
            "working_stack": str(source),
            "working_stack_hash": sha256_file(source),
            "source_dimensions_xyz": header.shape_xyz,
            "working_dimensions_xyz": header.shape_xyz,
            "source_pixel_size_A": header.pixel_size_A,
            "working_pixel_size_A": header.pixel_size_A,
            "newstack_command": None,
            "newstack_version": None,
        }
    if shutil.which("newstack") is None:
        raise RuntimeError("newstack is required for extra_projection_binning > 1 but is not on PATH")
    work_dir = Path(run_dir) / "working_stacks"
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{series_id}_bin{factor}.mrc"
    cmd = ["newstack", "-input", str(source), "-output", str(out), "-shrink", str(float(factor)), "-float", "0"]
    version_cp = subprocess.run(["newstack", "-version"], text=True, capture_output=True, check=False)
    version = (version_cp.stdout or version_cp.stderr).strip()
    cp = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if cp.returncode != 0:
        raise RuntimeError(f"newstack -shrink failed rc={cp.returncode}: {' '.join(cmd)}\n{cp.stderr or cp.stdout}")
    working_header = validate_stack(out, expected_tilts=expected_tilts)
    expected_x = header.nx // factor
    expected_y = header.ny // factor
    if working_header.nx != expected_x or working_header.ny != expected_y:
        raise RuntimeError(
            f"newstack output dimensions {working_header.nx}x{working_header.ny} "
            f"!= expected {expected_x}x{expected_y}"
        )
    expected_pixel = header.pixel_size_A * factor
    if abs(working_header.pixel_size_A - expected_pixel) > max(1e-3, expected_pixel * 1e-5):
        raise RuntimeError(f"newstack output pixel size {working_header.pixel_size_A} != expected {expected_pixel}")
    return out, working_header, {
        "factor": factor,
        "source_stack": str(source),
        "source_stack_hash": sha256_file(source),
        "working_stack": str(out),
        "working_stack_hash": sha256_file(out),
        "source_dimensions_xyz": header.shape_xyz,
        "working_dimensions_xyz": working_header.shape_xyz,
        "source_pixel_size_A": header.pixel_size_A,
        "working_pixel_size_A": working_header.pixel_size_A,
        "newstack_command": cmd,
        "newstack_version": version,
    }


def _validate_warp_outputs(*, frame_settings: Path, tilt_settings: Path, tomostar: Path, xml: Path, processing: Path) -> None:
    for label, path in (
        ("frame-series settings", frame_settings),
        ("tilt-series settings", tilt_settings),
        ("tomostar", tomostar),
        ("Warp XML", xml),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"WarpTools did not create a non-empty {label}: {path}")
    if frame_settings.read_text(errors="ignore").lstrip().startswith("{"):
        raise RuntimeError(f"frame-series settings look like private JSON, not WarpTools settings: {frame_settings}")
    if tilt_settings.read_text(errors="ignore").lstrip().startswith("{"):
        raise RuntimeError(f"tilt-series settings look like private JSON, not WarpTools settings: {tilt_settings}")
    if "<WarpProject" in xml.read_text(errors="ignore"):
        raise RuntimeError(f"Warp XML uses the private v6 <WarpProject> schema: {xml}")
    if not processing.is_dir():
        raise RuntimeError(f"WarpTools did not materialise the processing directory: {processing}")


def _legacy_converter_available() -> tuple[bool, str]:
    return False, (
        "20_initial_alignment_and_qc is blocked: v6 must use the real v5 Warp XML converter, "
        "not the previous private XML mutation path"
    )


def _maybe_generate_alignment_job(cfg: ProjectConfig, *, settings: Path, run_dir: Path, toml_hash: str) -> None:
    from .jobs import generate_stage_jobs
    from .stages import plan_stages

    layout = V6Layout(run_dir).create()
    graph_path = layout.manifests / "job_graph.json"
    graph = json.loads(graph_path.read_text()) if graph_path.is_file() else {"jobs": {}, "blocked": {}}
    ok, reason = _legacy_converter_available()
    if not ok:
        graph.setdefault("blocked", {})["20_initial_alignment_and_qc"] = reason
        atomic_json(graph_path, graph)
        return
    stages = [stage for stage in plan_stages(cfg) if stage.stage_id == "20_initial_alignment_and_qc"]
    written = generate_stage_jobs(
        jobs_dir=layout.jobs,
        run_dir=run_dir,
        settings_path=settings,
        toml_hash=toml_hash,
        cluster=cfg.cluster,
        stages=stages,
    )
    graph.setdefault("jobs", {}).update(written)
    graph.setdefault("blocked", {}).pop("20_initial_alignment_and_qc", None)
    atomic_json(graph_path, graph)
