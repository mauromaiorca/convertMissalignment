#!/usr/bin/env python3
"""TOML-driven IMOD reconstruction for v8 MissAlignment snapshots."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import finalize as FIN
    from . import project_config as PC
    from .runlayout import RunLayout, dataset_id_from_config
except ImportError:  # direct script execution from generated Slurm files
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline import finalize as FIN
    from pipeline import project_config as PC
    from pipeline.runlayout import RunLayout, dataset_id_from_config

SNAPSHOTS = ("pre_missalign", "smoke", "full")


class ReconstructionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImodReconstructionPlan:
    settings_path: Path
    snapshot: str
    run_dir: Path
    work_dir: Path
    xml: Path
    warp_dir: Path
    raw_stack: Path
    tlt_file: Path
    xtilt_file: Path | None
    newst_template: Path
    tilt_template: Path
    exported_xf: Path
    aligned_stack: Path
    reconstruction: Path
    generated_newst: Path
    generated_tilt: Path
    manifest: Path
    validation: Path
    pre_validation: Path
    command_diff: Path
    public_dir: Path
    config: dict[str, Any]


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w") as handle:
        json.dump(obj, handle, indent=2, default=str)
        handle.write("\n")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tail_text(text: str, max_lines: int = 80) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def load_settings(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ReconstructionError(f"project settings not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ReconstructionError(f"project settings TOML is invalid: {path}: {exc}") from exc


def layout_for(cfg: dict[str, Any], dataset_id: str | None = None) -> RunLayout:
    rc = PC.from_dict(cfg).require_resolved()
    condition = rc.conditions[0]
    return RunLayout.from_settings(
        out_dir=Path(rc.output_dir),
        basename=rc.basename,
        condition=condition,
        refinement_mode=rc.refinement_mode,
        dataset_id=dataset_id or dataset_id_from_config(cfg),
    )


def exactly_one_xml(directory: Path, label: str) -> Path:
    candidates = sorted(p for p in directory.glob("*.xml") if p.is_file() and p.stat().st_size > 0)
    if len(candidates) != 1:
        raise ReconstructionError(
            f"{label}: expected exactly one non-empty XML in {directory}, found {len(candidates)}"
        )
    return candidates[0]


def resolve_snapshot_xml(cfg: dict[str, Any], layout: RunLayout, snapshot: str) -> tuple[Path, Path]:
    result_manifest = layout.manifest("result_manifest.json")
    result = json.loads(result_manifest.read_text()) if result_manifest.is_file() else {}
    if snapshot == "pre_missalign":
        warp_dir = Path(result.get("pre_missalign_directory") or layout.pre_missalign_dir)
        return exactly_one_xml(warp_dir, "pre_missalign"), warp_dir
    if snapshot == "smoke":
        verdict_path = layout.results_dir / "smoke_verdict.json"
        if not verdict_path.is_file():
            raise ReconstructionError(f"smoke snapshot requires smoke verdict: {verdict_path}")
        verdict = json.loads(verdict_path.read_text())
        if verdict.get("smoke") != "ok":
            raise ReconstructionError(f"smoke verdict is not ok: {verdict_path}")
        xml = Path(verdict.get("xml") or "")
        if not xml.is_file() or xml.stat().st_size <= 0:
            raise ReconstructionError(f"smoke verdict does not name a valid XML: {verdict_path}")
        warp_dir = Path(result.get("smoke_directory") or layout.smoke_warp_dir)
        return xml, warp_dir
    if snapshot == "full":
        xml = Path(result.get("final_xml") or "")
        if not xml.is_file() or xml.stat().st_size <= 0:
            raise ReconstructionError(
                f"full snapshot requires non-empty result_manifest.final_xml in {result_manifest}"
            )
        warp_dir = Path(result.get("training_directory") or layout.full_warp_dir)
        return xml, warp_dir
    raise ReconstructionError(f"unknown reconstruction snapshot {snapshot!r}; expected {SNAPSHOTS}")


def _nonempty_path(value: Any, label: str) -> Path:
    path = Path(str(value or ""))
    if not path.is_file() or path.stat().st_size <= 0:
        raise ReconstructionError(f"{label} is missing or empty: {path}")
    return path


def build_plan(settings_path: Path, snapshot: str, dataset_id: str | None = None) -> ImodReconstructionPlan:
    cfg = load_settings(settings_path)
    rec = cfg.get("reconstruction", {}) or {}
    if not rec.get("enabled", True):
        raise ReconstructionError("[reconstruction].enabled is false")
    if snapshot not in rec.get("snapshots", list(SNAPSHOTS)):
        raise ReconstructionError(f"snapshot {snapshot!r} is not enabled in [reconstruction].snapshots")
    layout = layout_for(cfg, dataset_id)
    xml, warp_dir = resolve_snapshot_xml(cfg, layout, snapshot)
    rc = PC.from_dict(cfg)
    inp = cfg.get("input", {}) or {}
    imod = rec.get("imod", {}) or {}
    root = layout.attempts_dir / "reconstruction" / layout.dataset_id / "imod" / snapshot
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in attempts.glob("attempt_*") if p.is_dir())
    next_id = len(existing) + 1
    work_dir = attempts / f"attempt_{next_id:03d}" / rc.basename
    xf = work_dir / "transforms" / f"{rc.basename}_{snapshot}_raw_to_final.xf"
    return ImodReconstructionPlan(
        settings_path=Path(settings_path).resolve(),
        snapshot=snapshot,
        run_dir=layout.run_dir,
        work_dir=work_dir,
        xml=xml.resolve(),
        warp_dir=warp_dir.resolve(),
        raw_stack=_nonempty_path(inp.get("raw_stack"), "[input].raw_stack").resolve(),
        tlt_file=_nonempty_path(inp.get("final_tilt_file"), "[input].final_tilt_file").resolve(),
        xtilt_file=Path(inp["xtilt_file"]).resolve() if inp.get("xtilt_file") else None,
        newst_template=_nonempty_path(imod.get("newst_template") or inp.get("newst_com"), "newst.com template").resolve(),
        tilt_template=_nonempty_path(imod.get("tilt_template") or inp.get("tilt_com"), "tilt.com template").resolve(),
        exported_xf=xf,
        aligned_stack=work_dir / f"{rc.basename}_{snapshot}_ali.mrc",
        reconstruction=work_dir / f"{rc.basename}_{snapshot}.rec",
        generated_newst=work_dir / "newst.com",
        generated_tilt=work_dir / "tilt.com",
        manifest=work_dir / "reconstruction_manifest.json",
        validation=work_dir / "reconstruction_validation.json",
        pre_validation=work_dir / "pre_execution_validation.json",
        command_diff=work_dir / "command_file_updates.json",
        public_dir=(
            layout.export_dir / "imod" / "reconstructions" / "final"
            if snapshot == "full"
            else layout.results_dir / "reconstructions" / ("before" if snapshot == "pre_missalign" else "smoke")
        ),
        config=cfg,
    )


def _directive_key(line: str) -> str | None:
    match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*)\b", line)
    if not match:
        return None
    return match.group(1).lower()


def controlled_update(text: str, replacements: dict[str, str], *, label: str) -> tuple[str, dict[str, Any]]:
    lines = text.splitlines()
    positions: dict[str, list[int]] = {key.lower(): [] for key in replacements}
    for idx, line in enumerate(lines):
        key = _directive_key(line)
        if key in positions:
            positions[key].append(idx)
    missing = [key for key, idxs in positions.items() if not idxs]
    duplicate = {key: idxs for key, idxs in positions.items() if len(idxs) > 1}
    if missing or duplicate:
        raise ReconstructionError(
            f"{label}: cannot apply controlled updates; missing={missing}, duplicate={duplicate}"
        )
    updates = []
    for key, value in replacements.items():
        idx = positions[key.lower()][0]
        before = lines[idx]
        lines[idx] = f"{key} {value}"
        updates.append({"key": key, "line": idx + 1, "before": before, "after": lines[idx]})
    return "\n".join(lines) + "\n", {"label": label, "updates": updates}


def materialize_command_files(plan: ImodReconstructionPlan) -> dict[str, Any]:
    cfg = plan.config
    rec = cfg.get("reconstruction", {}) or {}
    imod = rec.get("imod", {}) or {}
    rc = PC.from_dict(cfg)
    newst_repl = {
        "InputFile": str(plan.raw_stack),
        "TransformFile": f"{rc.basename}.xf",
        "OutputFile": plan.aligned_stack.name,
    }
    if int(imod.get("newst_bin", 0)) > 0:
        newst_repl["BinByFactor"] = str(int(imod.get("newst_bin", 0)))
    tilt_repl = {
        "InputProjections": plan.aligned_stack.name,
        "OutputFile": plan.reconstruction.name,
        "TiltFile": f"{rc.basename}.tlt",
    }
    if bool(imod.get("use_gpu", False)):
        tilt_repl["UseGPU"] = str(int(imod.get("gpu_id", 0)))
    newst_text, newst_report = controlled_update(plan.newst_template.read_text(), newst_repl, label="newst.com")
    tilt_text, tilt_report = controlled_update(plan.tilt_template.read_text(), tilt_repl, label="tilt.com")
    plan.work_dir.mkdir(parents=True, exist_ok=True)
    plan.generated_newst.write_text(newst_text)
    plan.generated_tilt.write_text(tilt_text)
    (plan.work_dir / f"{rc.basename}.xf").write_text(plan.exported_xf.read_text())
    (plan.work_dir / f"{rc.basename}.tlt").write_text(plan.tlt_file.read_text())
    if plan.xtilt_file and plan.xtilt_file.is_file():
        (plan.work_dir / f"{rc.basename}.xtilt").write_text(plan.xtilt_file.read_text())
    report = {"newst": newst_report, "tilt": tilt_report}
    atomic_json(plan.command_diff, report)
    return report


def active_imod_commands(path: Path) -> list[dict[str, Any]]:
    commands = []
    for idx, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or not stripped.startswith("$"):
            continue
        token = stripped[1:].split(None, 1)[0].lower()
        commands.append({"line": idx, "program": token, "text": stripped})
    return commands


def validate_command_file(path: Path, *, expected: str, forbidden: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ReconstructionError(f"command file is missing or empty: {path}")
    commands = active_imod_commands(path)
    expected_hits = [c for c in commands if c["program"] == expected]
    forbidden_hits = [c for c in commands if c["program"] == forbidden]
    if len(expected_hits) != 1 or forbidden_hits:
        raise ReconstructionError(
            f"{path.name}: invalid IMOD command file; expected one ${expected}, "
            f"found {len(expected_hits)}, forbidden ${forbidden} hits={len(forbidden_hits)}"
        )
    return {"path": str(path), "expected": expected, "commands": commands}


def _count_rows(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#"))


def _read_numeric_rows(path: Path, columns: int | None = None) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        vals = [float(x) for x in stripped.split()]
        if columns is not None and len(vals) < columns:
            raise ReconstructionError(f"{path} row has {len(vals)} columns; expected at least {columns}: {line}")
        if not all(math.isfinite(v) for v in vals):
            raise ReconstructionError(f"{path} contains non-finite values: {line}")
        rows.append(vals)
    return rows


def validate_inputs(plan: ImodReconstructionPlan) -> dict[str, Any]:
    source_header = mrc_header(plan.raw_stack)
    tilt_rows = _read_numeric_rows(plan.tlt_file, 1)
    xf_rows = _read_numeric_rows(plan.exported_xf, 6)
    tlt_count = len(tilt_rows)
    xf_count = len(xf_rows)
    if tlt_count != xf_count:
        raise ReconstructionError(f"tilt count {tlt_count} != transform count {xf_count}")
    if source_header["shape_xyz"][2] != tlt_count:
        raise ReconstructionError(
            f"source stack sections {source_header['shape_xyz'][2]} != tilt count {tlt_count}"
        )
    determinants = []
    for row in xf_rows:
        det = row[0] * row[3] - row[1] * row[2]
        if not math.isfinite(det) or abs(det) < 1e-12:
            raise ReconstructionError(f"singular/non-finite XF matrix in {plan.exported_xf}: {row}")
        determinants.append(det)
    result = {
        "status": "validated",
        "source_stack": str(plan.raw_stack),
        "source_header": source_header,
        "tilt_count": tlt_count,
        "transform_count": xf_count,
        "xf_determinant_min_abs": min(abs(d) for d in determinants) if determinants else None,
        "settings_sha256": sha256_file(plan.settings_path),
    }
    atomic_json(plan.pre_validation, result)
    return result


def mrc_header(path: Path) -> dict[str, Any]:
    try:
        import mrcfile
    except ModuleNotFoundError as exc:
        raise ReconstructionError("mrcfile is required for reconstruction header validation") from exc
    with mrcfile.open(path, permissive=True, header_only=True) as handle:
        vox = handle.voxel_size
        return {
            "shape_xyz": [int(handle.header.nx), int(handle.header.ny), int(handle.header.nz)],
            "mode": int(handle.header.mode),
            "voxel_size_A": [float(vox.x), float(vox.y), float(vox.z)],
        }


def validate_outputs(plan: ImodReconstructionPlan, input_counts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"inputs": input_counts}
    for path, label in ((plan.aligned_stack, "aligned_stack"), (plan.reconstruction, "reconstruction")):
        if not path.is_file() or path.stat().st_size <= 0:
            raise ReconstructionError(f"{label} was not produced or is empty: {path}")
        out[label] = {"path": str(path), "sha256": sha256_file(path), "header": mrc_header(path)}
    if out["aligned_stack"]["header"]["shape_xyz"][2] != input_counts["tilt_count"]:
        raise ReconstructionError("newstack output section count does not match tilt count")
    geom = plan.config.get("geometry", {}) or {}
    expected_aligned = geom.get("aligned_shape_xyz")
    if expected_aligned and out["aligned_stack"]["header"]["shape_xyz"][:2] != list(expected_aligned[:2]):
        raise ReconstructionError(
            f"aligned stack dimensions {out['aligned_stack']['header']['shape_xyz']} "
            f"do not match configured aligned dimensions {expected_aligned}"
        )
    expected_pixel = geom.get("aligned_pixel_size_A") or geom.get("target_pixel_size_A")
    if expected_pixel:
        got = out["aligned_stack"]["header"]["voxel_size_A"][0]
        if abs(float(got) - float(expected_pixel)) > max(1e-3, 1e-4 * float(expected_pixel)):
            raise ReconstructionError(f"aligned stack pixel size {got} != expected {expected_pixel}")
    atomic_json(plan.validation, {"status": "completed", **out})
    return out


def executable(value: str, label: str) -> str:
    path = Path(value)
    resolved = str(path) if path.is_absolute() else shutil.which(value)
    if not resolved:
        raise ReconstructionError(
            f"{label} not found: {value}. Load IMOD or put the IMOD bin directory on PATH "
            f"before running the reconstruction batch."
        )
    return resolved


def run_imod_command_file(*, submfg: str, command_file: Path, cwd: Path,
                          consolidated_log: Path) -> dict[str, Any]:
    command_file = Path(command_file)
    command = [submfg, command_file.name]
    stdout_path = consolidated_log.with_suffix(consolidated_log.suffix + ".stdout")
    stderr_path = consolidated_log.with_suffix(consolidated_log.suffix + ".stderr")
    cp = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    stdout_path.write_text(cp.stdout)
    stderr_path.write_text(cp.stderr)
    native = cwd / (command_file.stem + ".log")
    native_text = native.read_text(errors="replace") if native.is_file() else ""
    consolidated_log.write_text(
        f"$ {' '.join(map(shlex.quote, command))}\n"
        f"cwd: {cwd}\n"
        "\n--- submfg stdout ---\n" + cp.stdout +
        "\n--- submfg stderr ---\n" + cp.stderr +
        f"\n--- native IMOD log: {native.name} ---\n" + native_text
    )
    if cp.returncode != 0:
        raise ReconstructionError(
            "IMOD command failed\n"
            f"command: {' '.join(map(shlex.quote, command))}\n"
            f"working directory: {cwd}\n"
            f"exit code: {cp.returncode}\n"
            f"tail:\n{tail_text(consolidated_log.read_text(errors='replace'))}"
        )
    return {
        "command": command,
        "cwd": str(cwd),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "native_log": str(native) if native.is_file() else None,
        "consolidated_log": str(consolidated_log),
    }


def export_xf(plan: ImodReconstructionPlan) -> list[str]:
    cfg = plan.config
    rc = PC.from_dict(cfg)
    condition = rc.conditions[0]
    params = FIN._synthesize_missalign_params(cfg, condition, plan.work_dir / "missalign_params.json")
    exporter = Path(__file__).resolve().parents[1] / "export_condition_results.py"
    cmd = [
        sys.executable, str(exporter),
        "--params", str(params),
        "--warp-dir", str(plan.warp_dir),
        "--condition", condition,
        "--out-dir", str(plan.exported_xf.parent),
        "--xml", str(plan.xml),
    ]
    plan.exported_xf.parent.mkdir(parents=True, exist_ok=True)
    for stale in plan.exported_xf.parent.glob("*_raw_to_final.xf"):
        stale.unlink()
    cp = subprocess.run(cmd, text=True, capture_output=True)
    (plan.work_dir / "export_warp_to_imod.log").write_text(cp.stdout + cp.stderr)
    if cp.returncode != 0:
        raise ReconstructionError(f"Warp XML export failed ({cp.returncode}); see export_warp_to_imod.log")
    produced = sorted(plan.exported_xf.parent.glob("*_raw_to_final.xf"))
    if len(produced) != 1 or produced[0].stat().st_size <= 0:
        raise ReconstructionError(f"Warp XML export did not produce exactly one non-empty XF in {plan.exported_xf.parent}")
    if produced[0] != plan.exported_xf:
        produced[0].replace(plan.exported_xf)
    return cmd


def verify_runtime_hashes(plan: ImodReconstructionPlan, *, expected_executor_sha: str | None,
                          expected_settings_sha: str | None, allow_mismatch: bool) -> dict[str, Any]:
    executor = Path(__file__).resolve()
    current_executor = sha256_file(executor)
    current_settings = sha256_file(plan.settings_path)
    report = {
        "executor": str(executor),
        "executor_sha256": current_executor,
        "expected_executor_sha256": expected_executor_sha,
        "settings": str(plan.settings_path),
        "settings_sha256": current_settings,
        "expected_settings_sha256": expected_settings_sha,
        "allow_mismatch": allow_mismatch,
    }
    errors = []
    if expected_executor_sha and expected_executor_sha != "unavailable" and expected_executor_sha != current_executor:
        errors.append(("reconstruction executor changed after project setup", expected_executor_sha, current_executor))
    if expected_settings_sha and expected_settings_sha != "unavailable" and expected_settings_sha != current_settings:
        errors.append(("project settings changed after job generation", expected_settings_sha, current_settings))
    if errors and not allow_mismatch:
        message = ["ERROR: " + errors[0][0]]
        for _, exp, cur in errors:
            message.append(f"Expected SHA-256: {exp}")
            message.append(f"Current SHA-256: {cur}")
        message.append("Regenerate the project jobs or use --allow-provenance-mismatch explicitly.")
        raise ReconstructionError("\n".join(message))
    return report




def _publish_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(os.path.relpath(source, destination.parent))


def publish_result(plan: ImodReconstructionPlan, manifest: dict[str, Any]) -> Path:
    """Publish validated products from the hidden attempt into the v8 tree."""
    public = plan.public_dir
    public.mkdir(parents=True, exist_ok=True)
    _publish_file(plan.reconstruction, public / plan.reconstruction.name)
    _publish_file(plan.aligned_stack, public / plan.aligned_stack.name)
    _publish_file(plan.exported_xf, public / plan.exported_xf.name)
    published = dict(manifest)
    published.update({
        "published_directory": str(public),
        "published_reconstruction": str(public / plan.reconstruction.name),
        "published_aligned_stack": str(public / plan.aligned_stack.name),
        "published_transform": str(public / plan.exported_xf.name),
        "attempt_manifest": str(plan.manifest),
    })
    target = public / "manifest.json"
    atomic_json(target, published)
    return target

def run_plan(plan: ImodReconstructionPlan, *, expected_executor_sha: str | None = None,
             expected_settings_sha: str | None = None, allow_provenance_mismatch: bool = False) -> int:
    cfg = plan.config
    imod = ((cfg.get("reconstruction", {}) or {}).get("imod", {}) or {})
    if imod.get("execution_mode", "submfg_command_file") != "submfg_command_file":
        raise ReconstructionError("only reconstruction.imod.execution_mode='submfg_command_file' is supported")
    submfg = executable(str(imod.get("submfg_executable", "submfg")), "submfg")
    plan.work_dir.mkdir(parents=True, exist_ok=True)
    provenance = verify_runtime_hashes(
        plan,
        expected_executor_sha=expected_executor_sha,
        expected_settings_sha=expected_settings_sha,
        allow_mismatch=allow_provenance_mismatch,
    )
    print(f"[reconstruct:{plan.snapshot}] executor_sha256={provenance['executor_sha256']}")
    print(f"[reconstruct:{plan.snapshot}] settings_sha256={provenance['settings_sha256']}")
    print(f"[reconstruct:{plan.snapshot}] START resolve/export")
    export_cmd = export_xf(plan)
    updates = materialize_command_files(plan)
    command_files = {
        "newst": validate_command_file(plan.generated_newst, expected="newstack", forbidden="tilt"),
        "tilt": validate_command_file(plan.generated_tilt, expected="tilt", forbidden="newstack"),
    }
    input_counts = validate_inputs(plan)
    print(f"[reconstruct:{plan.snapshot}] DONE resolve/export")
    print(f"[reconstruct:{plan.snapshot}] START newstack")
    newstack_run = run_imod_command_file(
        submfg=submfg, command_file=plan.generated_newst,
        cwd=plan.work_dir, consolidated_log=plan.work_dir / "newstack.log")
    print(f"[reconstruct:{plan.snapshot}] DONE newstack")
    print(f"[reconstruct:{plan.snapshot}] START tilt")
    tilt_run = run_imod_command_file(
        submfg=submfg, command_file=plan.generated_tilt,
        cwd=plan.work_dir, consolidated_log=plan.work_dir / "tilt.log")
    print(f"[reconstruct:{plan.snapshot}] DONE tilt")
    output_validation = validate_outputs(plan, input_counts)
    manifest = {
        "status": "completed",
        "snapshot": plan.snapshot,
        "settings": str(plan.settings_path),
        "xml": str(plan.xml),
        "warp_dir": str(plan.warp_dir),
        "raw_stack": str(plan.raw_stack),
        "raw_stack_sha256": sha256_file(plan.raw_stack),
        "tlt_file": str(plan.tlt_file),
        "xtilt_file": str(plan.xtilt_file) if plan.xtilt_file else None,
        "newst_template": str(plan.newst_template),
        "tilt_template": str(plan.tilt_template),
        "exported_xf": str(plan.exported_xf),
        "aligned_stack": str(plan.aligned_stack),
        "reconstruction": str(plan.reconstruction),
        "execution_mode": "submfg_command_file",
        "newst_run": newstack_run,
        "tilt_run": tilt_run,
        "export_command": export_cmd,
        "runtime_provenance": provenance,
        "command_file_validation": command_files,
        "command_file_updates": updates,
        "validation": output_validation,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "hostname": socket.getfqdn() or socket.gethostname(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(plan.manifest, manifest)
    published = publish_result(plan, manifest)
    print(f"[reconstruct:{plan.snapshot}] manifest: {published}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0].startswith("--"):
        argv = ["run", *argv]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--project-settings", required=True, type=Path)
    run.add_argument("--snapshot", required=True, choices=SNAPSHOTS)
    run.add_argument("--dataset", default=None)
    run.add_argument("--expected-executor-sha", default=None)
    run.add_argument("--expected-settings-sha", default=None)
    run.add_argument("--allow-provenance-mismatch", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.project_settings, args.snapshot, args.dataset)
        return run_plan(
            plan,
            expected_executor_sha=args.expected_executor_sha,
            expected_settings_sha=args.expected_settings_sha,
            allow_provenance_mismatch=args.allow_provenance_mismatch,
        )
    except ReconstructionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
