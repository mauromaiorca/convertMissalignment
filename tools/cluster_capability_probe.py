#!/usr/bin/env python3
"""Cluster capability probe (spec §19).

Self-contained (no repo imports) so it works when copied into a RUN_DIR/jobs dir
and executed on a compute node. Writes
``diagnostics/preflight/cluster_capabilities.{json,txt}`` and exits NONZERO when a
required capability is absent, so the full MissAlignment job can depend on it.

Never run or trusted on the local Mac for cluster claims — it REPORTS the real
environment it is executed in.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

REDACT = ("TOKEN", "KEY", "SECRET", "PASSWORD", "AUTH", "COOKIE")


def _redact_env() -> dict:
    return {k: ("<redacted>" if any(p in k.upper() for p in REDACT) else v)
            for k, v in os.environ.items()}


def _mod_path(name: str):
    try:
        spec = importlib.util.find_spec(name)
        return spec.origin if spec else None
    except Exception as exc:
        return f"<error: {exc}>"


def _git_rev(path) -> str | None:
    try:
        d = Path(path).resolve().parent
        cp = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"],
                            text=True, capture_output=True, timeout=10)
        return cp.stdout.strip() or None
    except Exception:
        return None


def _run(cmd, timeout=30):
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return {"rc": cp.returncode, "stdout": cp.stdout, "stderr": cp.stderr}
    except Exception as exc:
        return {"rc": None, "error": str(exc)}


def _supported_modes() -> dict:
    """Try to discover which alignment modes the installed MissAlignment fork supports."""
    out = {"queried": False, "modes": None, "error": None}
    try:
        import miss_alignment  # noqa: F401
        out["queried"] = True
        for attr in ("SUPPORTED_ALIGNMENTS", "ALIGNMENT_MODES", "SUPPORTED_ALIGNMENT_MODES"):
            mod = sys.modules.get("miss_alignment")
            if hasattr(mod, attr):
                out["modes"] = list(getattr(mod, attr))
                break
    except Exception as exc:
        out["error"] = str(exc)
    return out


def probe(run_dir: Path, training_dir: Path | None) -> dict:
    rep: dict = {
        "hostname": socket.gethostname(),
        "kernel": platform.release(), "architecture": platform.machine(),
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        "path": os.environ.get("PATH"),
        "ld_library_path": ("<redacted-present>" if os.environ.get("LD_LIBRARY_PATH") else None),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python_executable": sys.executable, "python_version": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "env_redacted": _redact_env(),
        "modules": _run(["bash", "-lc", "module list 2>&1"]),
    }
    # torch / CUDA
    try:
        import torch
        rep["torch"] = {
            "version": torch.__version__,
            "cuda_build": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [],
            "capabilities": [list(torch.cuda.get_device_capability(i))
                             for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [],
        }
    except Exception as exc:
        rep["torch"] = {"error": str(exc)}
    rep["nvidia_smi"] = _run(["nvidia-smi"])
    rep["nvidia_smi_L"] = _run(["nvidia-smi", "-L"])
    # packages
    rep["packages"] = {
        "miss_alignment": _mod_path("miss_alignment"),
        "warpylib": _mod_path("warpylib"),
        "torch_projectors": _mod_path("torch_projectors"),
        "mrcfile": _mod_path("mrcfile"),
    }
    rep["miss_alignment_git"] = _git_rev(rep["packages"]["miss_alignment"]) \
        if isinstance(rep["packages"]["miss_alignment"], str) else None
    rep["miss_alignment_exe"] = shutil.which("miss-alignment")
    rep["supported_alignment_modes"] = _supported_modes()
    # IMOD
    rep["imod"] = {exe: shutil.which(exe) for exe in ("newstack", "tilt", "ctfphaseflip", "submfg")}
    # filesystem tests
    rep["fs"] = {}
    try:
        t = run_dir / "diagnostics" / "preflight" / ".write_test"
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text("ok"); t.unlink()
        rep["fs"]["run_dir_writable"] = True
    except Exception as exc:
        rep["fs"]["run_dir_writable"] = False; rep["fs"]["run_dir_error"] = str(exc)
    if training_dir:
        xmls = list(Path(training_dir).glob("*.xml")) if Path(training_dir).exists() else []
        stacks = list((Path(training_dir) / "tiltstack").glob("*/*.st")) \
            if (Path(training_dir) / "tiltstack").exists() else []
        rep["warp_project"] = {"training_dir": str(training_dir),
                               "exists": Path(training_dir).exists(),
                               "xml_count": len(xmls), "stack_count": len(stacks),
                               "valid": bool(xmls and stacks)}
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--training-dir", type=Path, default=None)
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--require-missalignment", action="store_true")
    ap.add_argument("--require-modes", nargs="*", default=[])
    args = ap.parse_args()

    rep = probe(args.run_dir, args.training_dir)
    outd = args.run_dir / "diagnostics" / "preflight"
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "cluster_capabilities.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
    txt = [f"hostname: {rep['hostname']}", f"python: {rep['python_executable']}",
           f"torch: {rep.get('torch', {})}", f"miss_alignment: {rep['packages']['miss_alignment']}",
           f"warpylib: {rep['packages']['warpylib']}", f"imod: {rep['imod']}",
           f"warp_project: {rep.get('warp_project')}"]
    (outd / "cluster_capabilities.txt").write_text("\n".join(txt) + "\n")

    failures = []
    if args.require_cuda and not rep.get("torch", {}).get("cuda_available"):
        failures.append("CUDA not available")
    if args.require_missalignment and not (rep["packages"]["miss_alignment"] or rep["miss_alignment_exe"]):
        failures.append("miss-alignment not importable/on PATH")
    if args.require_modes:
        supported = rep["supported_alignment_modes"].get("modes")
        if supported is not None:
            missing = [m for m in args.require_modes if m not in supported and m not in ("smoke", "standard", "affine2d")]
            if missing:
                failures.append(f"unsupported alignment modes: {missing} (fork exposes {supported})")
    if args.training_dir and not rep.get("warp_project", {}).get("valid"):
        failures.append(f"Warp project invalid at {args.training_dir} (need *.xml + tiltstack/*/*.st)")

    verdict = {"ok": not failures, "failures": failures}
    (outd / "preflight_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    if failures:
        sys.stderr.write("PREFLIGHT FAILED:\n  " + "\n  ".join(failures) + "\n")
        return 1
    print("PREFLIGHT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
