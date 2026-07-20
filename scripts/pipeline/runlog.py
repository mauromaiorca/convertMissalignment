#!/usr/bin/env python3
"""Central structured logging + command capture + postmortem + debug bundle.

The single diagnosis backend for the three-phase workflow. Everything a remote
user needs to debug a cluster failure without the developer present is produced
here: one JSON event per line in ``logs/events.jsonl``, full (untruncated)
stdout/stderr/result.json per external command, a postmortem on any phase/step
failure, and a redacted, image-free ``debug_bundle_<run_id>.tar.gz``.

No stdlib ``logging`` global state, no secrets: environment values whose KEY
matches TOKEN/KEY/SECRET/PASSWORD/AUTH/COOKIE are redacted everywhere.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import tarfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REDACT_PATTERNS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "AUTH", "COOKIE")


def _utc() -> str:
    # new datetime via timezone-aware now; Date.now-free environments use this at
    # runtime only (never imported at workflow-script time).
    return datetime.now(timezone.utc).isoformat()


def redact_env(env: dict | None = None) -> dict:
    """Return a copy of ``env`` with secret-looking values redacted."""
    env = dict(os.environ if env is None else env)
    out = {}
    for k, v in env.items():
        if any(p in k.upper() for p in REDACT_PATTERNS):
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


def _safe_name(step: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(step))


@dataclass
class RunLogger:
    """Owns a RUN_DIR's structured logs. Construct once per phase invocation."""
    run_dir: Path
    run_id: str
    phase: str
    events_path: Path = field(init=False)
    _last_step: str = field(default="", init=False)
    _last_command: str = field(default="", init=False)

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        for sub in ("logs", "logs/commands", "logs/environment", "logs/resources",
                    f"logs/{self.phase}", ".internal/diagnostics",
                    ".internal/diagnostics/postmortem", "provenance"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "logs" / "events.jsonl"

    # -- events ------------------------------------------------------------
    def log_event(self, *, step: str, event: str, status: str = "info",
                  message: str = "", data: Any = None) -> dict:
        rec = {
            "timestamp_utc": _utc(), "run_id": self.run_id, "phase": self.phase,
            "step": step, "event": event, "status": status,
            "hostname": socket.gethostname(), "pid": os.getpid(), "cwd": os.getcwd(),
            "message": message, "data": data if data is not None else {},
        }
        with self.events_path.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        self._last_step = step
        return rec

    # -- environment snapshot ---------------------------------------------
    def write_environment(self, *, name: str = "environment", extra: dict | None = None) -> Path:
        report = {
            "timestamp_utc": _utc(), "run_id": self.run_id, "phase": self.phase,
            "sys_executable": sys.executable, "python_version": sys.version,
            "sys_path": sys.path, "platform": platform.platform(),
            "machine": platform.machine(), "hostname": socket.gethostname(),
            "cwd": os.getcwd(), "env_redacted": redact_env(),
            "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        if extra:
            report.update(extra)
        p = self.run_dir / "logs" / "environment" / f"{_safe_name(name)}.json"
        _atomic_write_text(p, json.dumps(report, indent=2, default=str) + "\n")
        return p

    # -- external commands -------------------------------------------------
    def run_command(self, argv: Sequence[str], *, step: str, env: dict | None = None,
                    cwd: str | Path | None = None, input_paths: Sequence[Path] = (),
                    output_paths: Sequence[Path] = (), check: bool = False,
                    stdin_text: str | None = None, timeout: float | None = None) -> dict:
        """Run an external command, capturing FULL stdout/stderr + a result.json.

        Never truncates stdout/stderr to a tail; the full streams are written to
        ``logs/commands/<step>.{stdout,stderr}.log`` and the structured outcome to
        ``logs/commands/<step>.result.json``. Returns the result dict.
        """
        name = _safe_name(step)
        cdir = self.run_dir / "logs" / "commands"
        cmd_path = cdir / f"{name}.command.txt"
        out_path = cdir / f"{name}.stdout.log"
        err_path = cdir / f"{name}.stderr.log"
        res_path = cdir / f"{name}.result.json"
        run_env = dict(os.environ if env is None else env)
        run_env.setdefault("IMOD_DIR", os.environ.get("IMOD_DIR", "/Applications/IMOD"))
        shell_render = " ".join(_shquote(a) for a in argv)
        cmd_path.write_text(shell_render + "\n")
        self._last_command = shell_render
        start = _utc()
        self.log_event(step=step, event="command_start", status="running",
                       message=shell_render, data={"argv": list(argv)})
        try:
            cp = subprocess.run(list(argv), env=run_env, cwd=str(cwd) if cwd else None,
                                text=True, capture_output=True, input=stdin_text, timeout=timeout)
            rc, stdout, stderr, signal = cp.returncode, cp.stdout, cp.stderr, None
        except subprocess.TimeoutExpired as exc:
            rc, stdout, stderr, signal = 124, exc.stdout or "", (exc.stderr or "") + "\nTIMEOUT", "TIMEOUT"
        except FileNotFoundError as exc:
            rc, stdout, stderr, signal = 127, "", f"executable not found: {exc}", "ENOENT"
        end = _utc()
        out_path.write_text(stdout or "")
        err_path.write_text(stderr or "")
        result = {
            "step": step, "argv": list(argv), "shell": shell_render,
            "cwd": str(cwd) if cwd else os.getcwd(),
            "env_selected": {k: run_env.get(k) for k in ("IMOD_DIR", "PATH", "CUDA_VISIBLE_DEVICES",
                                                         "OMP_NUM_THREADS") if k in run_env},
            "start_utc": start, "end_utc": end, "return_code": rc, "signal": signal,
            "stdout_path": str(out_path), "stderr_path": str(err_path),
            "stdout_bytes": len(stdout or ""), "stderr_bytes": len(stderr or ""),
            "input_paths": [str(p) for p in input_paths],
            "output_paths": [str(p) for p in output_paths],
            "outputs": [{"path": str(p), "exists": Path(p).exists(),
                         "size": Path(p).stat().st_size if Path(p).exists() else 0}
                        for p in output_paths],
        }
        _atomic_write_text(res_path, json.dumps(result, indent=2, default=str) + "\n")
        self.log_event(step=step, event="command_end",
                       status="ok" if rc == 0 else "error",
                       message=f"rc={rc}", data={"result_json": str(res_path), "return_code": rc})
        if check and rc != 0:
            tail = (stderr or "")[-400:]
            raise CommandError(f"{step}: command failed rc={rc} (full log: {err_path}). "
                               f"stderr tail: {tail}", result=result)
        return result

    # -- postmortem --------------------------------------------------------
    def write_postmortem(self, exc: BaseException, *, step: str = "",
                         expected_outputs: Sequence[Path] = (),
                         input_files: Sequence[Path] = (),
                         versions: dict | None = None,
                         suggestions: Sequence[str] = ()) -> Path:
        pm = self.run_dir / ".internal" / "diagnostics" / "postmortem"
        tb_path = pm / "traceback.txt"
        tb_path.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        # last events
        last = []
        if self.events_path.exists():
            last = self.events_path.read_text().splitlines()[-50:]
        (pm / "last_events.jsonl").write_text("\n".join(last) + ("\n" if last else ""))
        # filesystem snapshot (size-limited)
        (pm / "filesystem_snapshot.txt").write_text(_dir_listing(self.run_dir, limit=2000))
        (pm / "environment_summary.txt").write_text(
            "\n".join(f"{k}={v}" for k, v in sorted(redact_env().items())))
        failure = {
            "timestamp_utc": _utc(), "run_id": self.run_id, "phase": self.phase,
            "step": step or self._last_step,
            "exception_class": type(exc).__name__, "exception_message": str(exc),
            "traceback_path": str(tb_path), "last_successful_step": self._last_step,
            "last_command": self._last_command,
            "return_code": getattr(getattr(exc, "result", None), "get", lambda *_: None)("return_code")
            if hasattr(exc, "result") else None,
            "input_files": [str(p) for p in input_files],
            "expected_outputs": [{"path": str(p), "exists": Path(p).exists()} for p in expected_outputs],
            "software_versions": versions or {},
            "suggested_next_diagnostic_commands": list(suggestions) or [
                f"python prepare_imod_to_warp.py collect-debug <settings>",
                f"cat {self.run_dir}/logs/events.jsonl | tail -20",
                f"ls -R {self.run_dir}/logs/commands",
            ],
        }
        fpath = pm / "failure.json"
        _atomic_write_text(fpath, json.dumps(failure, indent=2, default=str) + "\n")
        self.log_event(step=step or self._last_step, event="postmortem", status="error",
                       message=str(exc), data={"failure_json": str(fpath)})
        return fpath


class CommandError(RuntimeError):
    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result or {}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _shquote(s: str) -> str:
    import shlex
    return shlex.quote(str(s))


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _dir_listing(root: Path, limit: int = 2000) -> str:
    root = Path(root)
    lines = []
    for p in sorted(root.rglob("*")):
        try:
            size = p.stat().st_size if p.is_file() else 0
        except OSError:
            size = -1
        lines.append(f"{size:>12}  {p.relative_to(root)}")
        if len(lines) >= limit:
            lines.append(f"... (truncated at {limit} entries)")
            break
    return "\n".join(lines) + "\n"


# Image/large-data extensions excluded from debug bundles.
_BUNDLE_EXCLUDE_SUFFIX = (".mrc", ".st", ".rec", ".ali", ".pt", ".ckpt", ".npy", ".map")


def collect_debug_bundle(run_dir: Path, run_id: str, *, include_checkpoints: bool = False) -> Path:
    """Build ``debug_bundle_<run_id>.tar.gz``: configs/manifests/logs/diagnostics,
    MRC header reports, transform summaries, result JSON, tracebacks, a size-limited
    listing. Excludes full MRC stacks, secrets, and (unless asked) large checkpoints.
    """
    run_dir = Path(run_dir)
    bundle_dir = run_dir / ".internal" / "debug_bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle = bundle_dir / f"debug_bundle_{run_id}.tar.gz"
    listing = run_dir / ".internal" / "diagnostics" / "bundle_listing.txt"
    _atomic_write_text(listing, _dir_listing(run_dir, limit=5000))

    def _ok(p: Path) -> bool:
        suf = p.suffix.lower()
        if suf in _BUNDLE_EXCLUDE_SUFFIX and not (include_checkpoints and suf in (".pt", ".ckpt")):
            return False
        try:
            if p.is_file() and p.stat().st_size > 25 * 1024 * 1024:  # skip > 25 MB
                return False
        except OSError:
            return False
        return True

    include_dirs = ["provenance", "logs", ".internal/diagnostics", "batches",
                    "missalignment", "warp_data"]
    with tarfile.open(bundle, "w:gz") as tar:
        for rel in include_dirs:
            d = run_dir / rel
            if not d.exists():
                continue
            for p in sorted(d.rglob("*")):
                if p.is_file() and _ok(p):
                    tar.add(p, arcname=str(p.relative_to(run_dir)))
        # top-level small config/manifest files
        for p in sorted(run_dir.glob("*.json")):
            if _ok(p):
                tar.add(p, arcname=p.name)
    return bundle
