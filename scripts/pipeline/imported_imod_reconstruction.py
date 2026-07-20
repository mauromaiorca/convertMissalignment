#!/usr/bin/env python3
"""Regenerate the imported IMOD reconstruction without modifying source files."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .imod_reconstruction import (
        ReconstructionError,
        atomic_json,
        controlled_update,
        executable,
        mrc_header,
        run_imod_command_file,
        sha256_file,
        validate_command_file,
    )
    from .runlayout import RunLayout, dataset_id_from_config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.imod_reconstruction import (
        ReconstructionError,
        atomic_json,
        controlled_update,
        executable,
        mrc_header,
        run_imod_command_file,
        sha256_file,
        validate_command_file,
    )
    from pipeline.runlayout import RunLayout, dataset_id_from_config


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _required_file(value: Any, label: str) -> Path:
    path = Path(str(value or ""))
    if not path.is_file() or path.stat().st_size <= 0:
        raise ReconstructionError(f"{label} is missing or empty: {path}")
    return path.resolve()


def _publish(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(os.path.relpath(source, destination.parent))


def run(settings: Path) -> int:
    settings = settings.resolve()
    cfg = _load(settings)
    project = cfg.get("project", {}) or {}
    paths = cfg.get("paths", {}) or {}
    conversion = cfg.get("conversion", {}) or {}
    ma = cfg.get("missalignment", {}) or {}
    inp = cfg.get("input", {}) or {}
    rec = cfg.get("reconstruction", {}) or {}
    imod = rec.get("imod", {}) or {}
    basename = str(project.get("basename") or "series")
    conditions = conversion.get("initial_conditions") or ["ali_identity"]
    if len(conditions) != 1:
        raise ReconstructionError("version 8 requires one condition per project")
    layout = RunLayout.from_settings(
        out_dir=Path(paths.get("output_dir") or "."),
        basename=basename,
        condition=str(conditions[0]),
        refinement_mode=str(ma.get("refinement_mode") or "standard"),
        dataset_id=dataset_id_from_config(cfg),
    ).create()

    raw_stack = _required_file(inp.get("raw_stack"), "raw stack")
    xf = _required_file(inp.get("final_xf_file"), "final IMOD transform")
    tilt_file = _required_file(inp.get("final_tilt_file"), "tilt-angle file")
    newst_template = _required_file(imod.get("newst_template") or inp.get("newst_com"), "newst.com")
    tilt_template = _required_file(imod.get("tilt_template") or inp.get("tilt_com"), "tilt.com")
    submfg = executable(str(imod.get("submfg_executable", "submfg")), "submfg")

    job_id = os.environ.get("SLURM_JOB_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = layout.attempts_dir / "reconstruction" / "imported_imod" / f"attempt_{job_id}"
    work.mkdir(parents=True, exist_ok=False)
    local_xf = work / f"{basename}.xf"
    local_tlt = work / f"{basename}.tlt"
    aligned = work / f"{basename}_imported_ali.mrc"
    reconstruction = work / f"{basename}_imported.rec"
    newst_com = work / "newst.com"
    tilt_com = work / "tilt.com"
    shutil.copy2(xf, local_xf)
    shutil.copy2(tilt_file, local_tlt)

    newst_text, newst_updates = controlled_update(
        newst_template.read_text(),
        {
            "InputFile": str(raw_stack),
            "TransformFile": local_xf.name,
            "OutputFile": aligned.name,
        },
        label="newst.com",
    )
    tilt_text, tilt_updates = controlled_update(
        tilt_template.read_text(),
        {
            "InputProjections": aligned.name,
            "OutputFile": reconstruction.name,
            "TiltFile": local_tlt.name,
        },
        label="tilt.com",
    )
    newst_com.write_text(newst_text)
    tilt_com.write_text(tilt_text)
    validate_command_file(newst_com, expected="newstack", forbidden="tilt")
    validate_command_file(tilt_com, expected="tilt", forbidden="newstack")

    newstack_run = run_imod_command_file(
        submfg=submfg, command_file=newst_com, cwd=work,
        consolidated_log=work / "newstack.log",
    )
    tilt_run = run_imod_command_file(
        submfg=submfg, command_file=tilt_com, cwd=work,
        consolidated_log=work / "tilt.log",
    )
    for output, label in ((aligned, "aligned stack"), (reconstruction, "reconstruction")):
        if not output.is_file() or output.stat().st_size <= 0:
            raise ReconstructionError(f"{label} was not produced: {output}")

    public = layout.imported_imod_dir / "reconstructions" / "regenerated"
    _publish(aligned, public / aligned.name)
    _publish(reconstruction, public / reconstruction.name)
    manifest = {
        "schema_version": 1,
        "artifact_type": "imported_imod_reconstruction",
        "status": "completed",
        "settings": str(settings),
        "source_raw_stack": str(raw_stack),
        "source_xf": str(xf),
        "source_tilt_file": str(tilt_file),
        "aligned_stack": str(public / aligned.name),
        "reconstruction": str(public / reconstruction.name),
        "aligned_header": mrc_header(aligned),
        "reconstruction_header": mrc_header(reconstruction),
        "source_hashes": {
            "xf": sha256_file(xf),
            "tilt_file": sha256_file(tilt_file),
            "newst_template": sha256_file(newst_template),
            "tilt_template": sha256_file(tilt_template),
        },
        "command_updates": {"newst": newst_updates, "tilt": tilt_updates},
        "newstack_run": newstack_run,
        "tilt_run": tilt_run,
        "attempt_directory": str(work),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "hostname": socket.getfqdn() or socket.gethostname(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(work / "manifest.json", manifest)
    atomic_json(public / "manifest.json", manifest)
    print(f"[reconstruct-imported-imod] {public / reconstruction.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-settings", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        return run(args.project_settings)
    except (ReconstructionError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
