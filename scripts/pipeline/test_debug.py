#!/usr/bin/env python3
"""`--test-debug` real-data diagnostic harness.

A comprehensive but LIGHTWEIGHT diagnostic run of the real workflow that produces a
compact, shareable directory + archive. It is built ON the repaired canonical
configuration system (``init_project``/``project_config``/``discovery``/``geometry``/
``jobs``/``capabilities``) and introduces NO independent discovery or conversion
pipeline. It never modifies the source, never launches a long scientific run, never
auto-submits GPU jobs, and never claims CUDA/Warp/MissAlignment/Slurm were verified
unless the relevant artifact was actually produced and inspected.

Honest stage states: PASS / FAIL / WARNING / NOT_RUN_DEPENDENCY_MISSING /
NOT_RUN_USER_ACTION_REQUIRED / NOT_APPLICABLE.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import capabilities as CAP          # noqa: E402
from pipeline import discovery as DISC            # noqa: E402
from pipeline import init_project as INIT         # noqa: E402
from pipeline import project_config as PC         # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
WARNING = "WARNING"
NOT_RUN_DEP = "NOT_RUN_DEPENDENCY_MISSING"
NOT_RUN_USER = "NOT_RUN_USER_ACTION_REQUIRED"
NOT_APPLICABLE = "NOT_APPLICABLE"

REDACT = ("TOKEN", "KEY", "SECRET", "PASSWORD", "AUTH", "COOKIE")
SUBDIRS = ("config", "source_inventory", "fixtures", "geometry", "statistics", "previews",
           "warp", "missalignment", "imod", "jobs", "logs", "results", "diagnostics", "bundle")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class DebugOptions:
    data_dir: str
    out_dir: str
    basename: Optional[str] = None
    max_image_dim: int = 512
    smoke_max_image_dim: int = 256
    tilt_count: int = 9
    bundle_max_mb: int = 50              # §16 quick default
    quick: bool = True                  # quick by default; --test-debug-full sets False
    all_tilts: bool = False             # quick: no all-tilt fixture (§16)
    run_imod: bool = False              # quick: no IMOD mini reconstruction (§16)
    generate_slurm: bool = True
    submit_debug: bool = False
    keep_intermediates: bool = False
    run_id: Optional[str] = None
    force: bool = False
    # memory + timeout bounds (§16)
    max_sample_voxels: int = 2_000_000
    max_memory_mb: int = 256
    command_timeout_s: int = 600
    global_timeout_s: int = 1800
    resume: bool = False
    from_stage: Optional[str] = None
    only_stage: Optional[str] = None
    # cluster/env passthrough (for generated jobs)
    missalign_env: Optional[str] = None
    conditions: tuple = ("raw_xf_affine_fixed",)
    # runtime-only: global monotonic deadline shared by every external command (set in
    # run_test_debug); not a CLI field.
    deadline_monotonic: Optional[float] = None


@dataclass
class StageResult:
    name: str
    state: str
    detail: str = ""
    data: dict = field(default_factory=dict)

    def ok(self) -> bool:
        return self.state in (PASS, WARNING, NOT_APPLICABLE)


@dataclass
class DebugLayout:
    root: Path

    def create(self) -> "DebugLayout":
        for s in SUBDIRS:
            (self.root / s).mkdir(parents=True, exist_ok=True)
        return self

    def __getattr__(self, name):
        if name in SUBDIRS:
            return self.root / name
        raise AttributeError(name)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _atomic_json(path: Path, obj) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def _redact_env() -> dict:
    return {k: ("<redacted>" if any(p in k.upper() for p in REDACT) else v)
            for k, v in os.environ.items()}


def _write_journal(lay, journal: dict, *, active) -> None:
    """Persistent stage journal (§16): the user can always identify the active stage."""
    _atomic_json(lay.results / "stage_state.json",
                 {"active_stage": active, "updated_utc": _utc(), "stages": dict(journal)})


def _git_rev() -> Optional[str]:
    try:
        cp = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]),
                             "rev-parse", "HEAD"], text=True, capture_output=True, timeout=10)
        return cp.stdout.strip() or None
    except Exception:
        return None


def _config_hash(data_dir: str, basename: str, opts: DebugOptions) -> str:
    payload = json.dumps({"data_dir": data_dir, "basename": basename,
                          "max": opts.max_image_dim, "smoke": opts.smoke_max_image_dim,
                          "tilts": opts.tilt_count}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:6]


def _count_lines(p) -> int:
    return sum(1 for ln in Path(p).read_text().splitlines() if ln.strip())


def read_mrc_header(path) -> dict:
    """Header-only MRC read (§16): shape/mode/voxel/origin/file_size. NEVER loads or
    converts the data array. Safe on source-scale stacks."""
    import mrcfile
    with mrcfile.open(path, permissive=True, header_only=True) as h:
        nx, ny, nz = int(h.header.nx), int(h.header.ny), int(h.header.nz)
        return {"path": str(path), "shape_xy": [nx, ny], "n_sections": nz,
                "mode": int(h.header.mode), "pixel_size_A": float(h.voxel_size.x),
                "origin_xyz": [float(h.header.origin.x), float(h.header.origin.y), float(h.header.origin.z)],
                "file_size": Path(path).stat().st_size}


def sample_mrc_statistics(path, *, selected_sections=None, max_sample_voxels=2_000_000,
                          max_memory_mb=256, seed=12345) -> dict:
    """Bounded, memory-safe image statistics (§16). Reads selected sections ONE AT A
    TIME via mmap, sub-samples each with a deterministic stride, never materializes the
    full stack, and never converts the whole dataset to float64. Percentiles are
    clearly labelled as sampled/approximate."""
    import mrcfile
    import numpy as np
    hdr = read_mrc_header(path)
    nx, ny = hdr["shape_xy"]; nsec = hdr["n_sections"]
    secs = list(selected_sections) if selected_sections is not None else list(range(nsec))
    per_sec_budget = max(1, max_sample_voxels // max(1, len(secs)))
    samples, per_tilt = [], []
    nan = inf = zero = total_seen = 0
    with mrcfile.mmap(path, permissive=True, mode="r") as h:
        data = h.data
        for i in secs:
            plane = data[i] if data.ndim == 3 else data
            flat = plane.reshape(-1)
            stride = max(1, flat.size // per_sec_budget)
            sampled = np.asarray(flat[::stride][:per_sec_budget], dtype=np.float32)
            fin = np.isfinite(sampled)
            nan += int(np.isnan(sampled).sum()); inf += int(np.isinf(sampled).sum())
            zero += int((sampled == 0).sum()); total_seen += sampled.size
            samp = sampled[fin]
            samples.append(samp.astype(np.float64, copy=False))
            med = float(np.median(samp)) if samp.size else 0.0
            mad = float(np.median(np.abs(samp - med))) if samp.size else 0.0
            per_tilt.append({"index": int(i), "mean": float(samp.mean()) if samp.size else 0.0,
                             "std": float(samp.std()) if samp.size else 0.0,
                             "min": float(samp.min()) if samp.size else 0.0,
                             "max": float(samp.max()) if samp.size else 0.0, "median": med, "mad": mad,
                             "frac_zero": float((sampled == 0).mean()) if sampled.size else 0.0,
                             "frac_nonfinite": float((~fin).mean()) if sampled.size else 0.0,
                             "robust_contrast": float(mad / (abs(med) + 1e-9)),
                             "sample_voxels": int(sampled.size)})
            del plane, flat, sampled
    alls = np.concatenate(samples) if samples else np.array([], dtype=np.float64)
    pct = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    perc = {str(p): float(np.percentile(alls, p)) for p in pct} if alls.size else {}
    med = float(np.median(alls)) if alls.size else 0.0
    summary = {"sampled": True, "counts_are_sampled": True,
               "sample_voxels": int(alls.size), "sections_sampled": len(secs),
               "total_voxels": int(nx) * int(ny) * int(nsec),
               "nan_count": nan, "inf_count": inf, "zero_count": zero,
               "min": float(alls.min()) if alls.size else None, "max": float(alls.max()) if alls.size else None,
               "mean": float(alls.mean()) if alls.size else None, "std": float(alls.std()) if alls.size else None,
               "median": med, "mad": float(np.median(np.abs(alls - med))) if alls.size else 0.0,
               "percentiles_sampled": perc, "shape_xy": [nx, ny], "n_sections": nsec,
               "pixel_size_A": hdr["pixel_size_A"], "mode": hdr["mode"], "file_size": hdr["file_size"],
               "physical_xy_A": [nx * hdr["pixel_size_A"], ny * hdr["pixel_size_A"]]}
    return {"summary": summary, "per_tilt": per_tilt}


def _measure_mrc(path):
    """Full float64 load — for SMALL reduced fixtures only; refuses source-scale files."""
    import mrcfile
    import numpy as np
    hdr = read_mrc_header(path)
    nx, ny = hdr["shape_xy"]; nsec = hdr["n_sections"]
    if nx * ny * max(1, nsec) > 64_000_000:
        raise MemoryError(f"_measure_mrc refused full load of {path} ({nx}x{ny}x{nsec}); "
                          "use read_mrc_header / sample_mrc_statistics")
    with mrcfile.open(path, permissive=True) as h:
        d = np.asarray(h.data, dtype=np.float64)
        if d.ndim == 2:
            d = d[None]
        return hdr, d


def _image_stats(d):
    import numpy as np
    flat = d.reshape(-1)
    finite = np.isfinite(flat)
    fv = flat[finite]
    pct = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    perc = {str(p): float(np.percentile(fv, p)) for p in pct} if fv.size else {}
    med = float(np.median(fv)) if fv.size else 0.0
    mad = float(np.median(np.abs(fv - med))) if fv.size else 0.0
    return {
        "finite_count": int(finite.sum()), "nan_count": int(np.isnan(flat).sum()),
        "inf_count": int(np.isinf(flat).sum()), "zero_count": int((flat == 0).sum()),
        "min": float(fv.min()) if fv.size else None, "max": float(fv.max()) if fv.size else None,
        "mean": float(fv.mean()) if fv.size else None, "std": float(fv.std()) if fv.size else None,
        "median": med, "mad": mad, "percentiles": perc,
    }


def _per_tilt_stats(d):
    import numpy as np
    rows = []
    for i in range(d.shape[0]):
        s = d[i].reshape(-1)
        fin = np.isfinite(s)
        fv = s[fin]
        med = float(np.median(fv)) if fv.size else 0.0
        mad = float(np.median(np.abs(fv - med))) if fv.size else 0.0
        std = float(fv.std()) if fv.size else 0.0
        rows.append({
            "index": i, "mean": float(fv.mean()) if fv.size else 0.0, "std": std,
            "min": float(fv.min()) if fv.size else 0.0, "max": float(fv.max()) if fv.size else 0.0,
            "median": med, "mad": mad, "frac_zero": float((s == 0).mean()),
            "frac_nonfinite": float((~fin).mean()),
            "robust_contrast": float(mad / (abs(med) + 1e-9)),
        })
    return rows


def _xf_decompose(A, d):
    import numpy as np
    A = np.asarray(A, float).reshape(2, 2)
    det = float(np.linalg.det(A))
    U, S, Vt = np.linalg.svd(A)
    sv = [float(S[0]), float(S[1])]
    iso = float(np.sqrt(abs(det)))
    aniso = float(S[0] / S[1]) if S[1] != 0 else float("inf")
    rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    shear = float(A[0, 1] / A[1, 1]) if A[1, 1] != 0 else 0.0
    cond = float(S[0] / S[1]) if S[1] != 0 else float("inf")
    return {
        "determinant": det, "rotation_deg": rot, "singular_values": sv,
        "isotropic_scale": iso, "anisotropy": aniso, "shear": shear,
        "translation_x": float(d[0]), "translation_y": float(d[1]),
        "translation_norm": float(np.hypot(d[0], d[1])), "condition_number": cond,
        "reflection": det < 0,
    }


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_source_inventory(opts, lay, *, final=False) -> StageResult:
    """Record only the resolved project inputs, not every file below data_dir."""
    data_dir = Path(opts.data_dir)
    inv = {"data_dir": str(data_dir), "files": {}, "timestamp_utc": _utc()}
    hashes = {}
    have_mrc = CAP.probe_python_module("mrcfile").available

    if final:
        initial = json.loads((lay.source_inventory / "source_inventory.json").read_text())
        paths = [Path(x) for x in initial.get("files", {})]
    else:
        discovered = DISC.discover_sources(data_dir, opts.basename)
        inv["resolved_sources"] = discovered.to_dict()
        paths = []
        for key, value in discovered.to_dict().items():
            if key in ("basename", "data_dir", "report") or not value:
                continue
            path = Path(value)
            if path.is_file():
                paths.append(path)
        paths = sorted(set(paths), key=lambda x: str(x))

    for p in paths:
        try:
            st = p.stat()
        except OSError as exc:
            inv["files"][str(p)] = {"error": str(exc)}
            hashes[str(p)] = {"error": str(exc)}
            continue
        rec = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "mode": oct(st.st_mode)}
        if p.suffix.lower() in (".mrc", ".st", ".ali", ".rec") and have_mrc:
            try:
                hdr = read_mrc_header(p)
                rec["mrc_header"] = {k: hdr[k] for k in ("shape_xy", "n_sections", "pixel_size_A", "mode")}
            except Exception as exc:
                rec["mrc_error"] = str(exc)
        elif p.suffix.lower() in (".tlt", ".rawtlt", ".xf", ".xtilt", ".tltxf", ".defocus"):
            try:
                rec["line_count"] = _count_lines(p)
            except Exception:
                pass
        inv["files"][str(p)] = rec

        h = hashlib.sha256()
        if st.st_size <= 16 * 1024 * 1024:
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            mode = "full"
        else:
            with p.open("rb") as fh:
                h.update(fh.read(4 << 20))
                fh.seek(max(0, st.st_size - (4 << 20)))
                h.update(fh.read(4 << 20))
            h.update(str(st.st_size).encode())
            mode = "sampled_head4M_tail4M_size"
        hashes[str(p)] = {"sha256": h.hexdigest()[:32], "mode": mode, "size": st.st_size}

    suffix = "_final" if final else ""
    _atomic_json(lay.source_inventory / f"source_inventory{suffix}.json", inv)
    _atomic_json(lay.source_inventory / f"source_hashes{suffix}.json", hashes)
    (lay.source_inventory / f"source_permissions{suffix}.txt").write_text(
        f"data_dir mode: {oct(data_dir.stat().st_mode)}\nwritable: {os.access(data_dir, os.W_OK)}\n")
    return StageResult("source_inventory" + suffix, PASS, f"{len(inv['files'])} resolved files",
                       {"n_files": len(inv["files"])})


def stage_canonical_config(opts, lay) -> StageResult:
    """The authoritative bootstrap: init_project -> one resolved canonical TOML."""
    input_cfg = {
        "project": {"basename": opts.basename},
        "paths": {"data_root": opts.data_dir, "output_dir": str(lay.config)},
        "conversion": {"initial_conditions": list(opts.conditions)},
        "missalignment": {"refinement_mode": "standard", "result_backend": "warp_xml"},
    }
    if opts.missalign_env:
        input_cfg["cluster"] = {"environment": opts.missalign_env}
    res = INIT.init_project(input_cfg, out_dir_override=str(lay.config),
                            data_dir_override=opts.data_dir, basename_override=opts.basename)
    # mirror the resolved TOML to the canonical config/ name + provenance
    src = Path(res["resolved_toml"])
    dst = lay.config / "project_settings.resolved.toml"
    dst.write_text(src.read_text())
    schema = Path(__file__).resolve().parents[2] / "config" / "project_settings.schema.json"
    if schema.is_file():
        (lay.config / "project_settings.schema.json").write_text(schema.read_text())
    rc = PC.load(dst)
    problems = PC.validate(rc, require_geometry=True, require_resolved=True)
    _atomic_json(lay.config / "config_provenance.json", {
        "resolved_toml": str(dst), "init_manifests": res["manifests_dir"],
        "tilt_axis": res["tilt_axis"], "warp_modes": res["warp_modes"],
        "git_rev": _git_rev(), "generated_utc": _utc()})
    _atomic_json(lay.config / "config_validation.json", {"problems": problems, "ok": not problems})
    if problems:
        return StageResult("canonical_config", FAIL, "; ".join(problems))
    return StageResult("canonical_config", PASS, f"tilt_axis={res['tilt_axis'][0]}",
                       {"resolved_toml": str(dst), "config": rc})


def _shrink_to(max_dim_now, target):
    f = max(1.0, float(max_dim_now) / float(target))
    return f


@dataclass
class CommandResult:
    """subprocess.CompletedProcess-compatible (returncode/stdout/stderr) plus timeout flag."""
    cmd: list
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_s: float = 0.0


def _kill_group(proc) -> None:
    """SIGTERM then SIGKILL the whole process group (§17: never leak a runaway child)."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, OSError):
            return
        try:
            proc.wait(timeout=3.0)
            return
        except subprocess.TimeoutExpired:
            continue


def run_external_command(cmd, *, env=None, timeout_s=600, label=None,
                         heartbeat_s=30.0, cwd=None, deadline=None) -> CommandResult:
    """Run an external command with a START line, a periodic heartbeat, a hard per-command
    timeout and a process-group kill on timeout (§16/§17: external work cannot appear frozen
    and cannot run unbounded). Output is captured via temp files so a large stdout/stderr can
    never deadlock on a full pipe buffer. `deadline` (a time.monotonic() value) caps the wait
    by the remaining global budget."""
    label = label or (str(cmd[0]) if cmd else "command")
    t0 = time.monotonic()
    print(f"[cmd:START] {_utc()} {label}: {' '.join(str(c) for c in cmd)}", flush=True)
    out_f = tempfile.TemporaryFile(mode="w+")
    err_f = tempfile.TemporaryFile(mode="w+")
    try:
        proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=out_f, stderr=err_f,
                                text=True, start_new_session=True)
    except (FileNotFoundError, OSError) as exc:
        out_f.close(); err_f.close()
        print(f"[cmd:END] {label}: SPAWN-FAIL {exc}", flush=True)
        return CommandResult(list(cmd), 127, "", str(exc), False, 0.0)
    timed_out = False
    next_hb = t0 + heartbeat_s
    while True:
        try:
            proc.wait(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        elapsed = now - t0
        eff_timeout = float(timeout_s)
        if deadline is not None:
            eff_timeout = min(eff_timeout, max(0.0, deadline - t0))
        if elapsed >= eff_timeout:
            timed_out = True
            print(f"[cmd:TIMEOUT] {label}: exceeded {eff_timeout:0.0f}s; killing process group",
                  flush=True)
            _kill_group(proc)
            break
        if now >= next_hb:
            print(f"[cmd:HEARTBEAT] {label}: still running, {elapsed:0.0f}s elapsed "
                  f"(limit {eff_timeout:0.0f}s)", flush=True)
            next_hb = now + heartbeat_s
    duration = time.monotonic() - t0
    out_f.seek(0); err_f.seek(0)
    stdout, stderr = out_f.read(), err_f.read()
    out_f.close(); err_f.close()
    rc = proc.returncode if proc.returncode is not None else -signal.SIGKILL
    status = "TIMEOUT" if timed_out else ("OK" if rc == 0 else f"rc={rc}")
    print(f"[cmd:END] {label}: {status} in {duration:0.1f}s", flush=True)
    return CommandResult(list(cmd), rc, stdout, stderr, timed_out, duration)


def _newstack_shrink(src, dst, factor, env, opts=None):
    cmd = ["newstack", "-input", str(src), "-output", str(dst)]
    if factor > 1.0:
        cmd += ["-shrink", f"{factor:.6f}"]
    cmd += ["-float", "0"]
    cp = run_external_command(cmd, env=env, label=f"newstack {Path(dst).name}",
                              timeout_s=getattr(opts, "command_timeout_s", 600),
                              deadline=getattr(opts, "deadline_monotonic", None))
    return cmd, cp


def stage_geometry_fixture(opts, lay, rc) -> StageResult:
    """All-tilt fixture; reduce spatial dims to <= max_image_dim; measure output."""
    if not CAP.probe_executable("newstack").available:
        return StageResult("geometry_fixture", NOT_RUN_DEP, "newstack not on PATH",
                           {"missing": "newstack"})
    if not CAP.probe_python_module("mrcfile").available:
        return StageResult("geometry_fixture", NOT_RUN_DEP, "mrcfile missing", {"missing": "mrcfile"})
    fdir = lay.fixtures / "geometry_all_tilts"; fdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD")}
    out = {}
    manifest = {"all_tilts": True, "stacks": {}, "commands": []}
    for role in ("raw_stack", "aligned_stack"):
        src = getattr(rc.sources, role)
        if not src or not Path(src).is_file():
            continue
        hdr0 = read_mrc_header(src)
        factor = _shrink_to(max(hdr0["shape_xy"]), opts.max_image_dim)
        dst = fdir / (Path(src).stem + f"_geomfix.mrc")
        cmd, cp = _newstack_shrink(src, dst, factor, env, opts)
        manifest["commands"].append({"cmd": cmd, "rc": cp.returncode})
        if cp.returncode != 0 or not dst.is_file():
            return StageResult("geometry_fixture", FAIL, f"newstack failed for {role}: {cp.stderr[-200:]}")
        hdr1 = read_mrc_header(dst)   # MEASURE actual output (never integer-division)
        manifest["stacks"][role] = {"source": src, "fixture": str(dst),
                                    "source_header": hdr0, "fixture_header": hdr1,
                                    "shrink_factor": factor}
        out[role] = {"fixture": str(dst), "shape_xy": hdr1["shape_xy"], "pixel_A": hdr1["pixel_size_A"]}
    # copy the small metadata (all tilts preserved)
    for role in ("final_tilt_file", "raw_tilt_file", "xtilt_file", "tltxf_file", "final_xf_file"):
        src = getattr(rc.sources, role)
        if src and Path(src).is_file():
            shutil.copy2(src, fdir / Path(src).name)
    _atomic_json(fdir / "fixture_manifest.json", manifest)
    (fdir / "project_settings.toml").write_text((lay.config / "project_settings.resolved.toml").read_text())
    if not out:
        return StageResult("geometry_fixture", WARNING, "no stacks to reduce")
    return StageResult("geometry_fixture", PASS, f"reduced to <= {opts.max_image_dim}px", out)


def _select_tilts(angles, n):
    import numpy as np
    a = np.asarray(angles, float)
    order = list(range(len(a)))
    if len(a) <= n:
        return order
    # nearest-zero + approx-uniform across the range, preserve original order
    target = np.linspace(a.min(), a.max(), n)
    chosen = set()
    chosen.add(int(np.argmin(np.abs(a))))
    for t in target:
        chosen.add(int(np.argmin(np.abs(a - t))))
    # if rounding collapsed some, top up with the most-separated remaining
    while len(chosen) < n:
        rem = [i for i in order if i not in chosen]
        if not rem:
            break
        # pick the one maximizing min-distance to chosen angles
        best = max(rem, key=lambda i: min(abs(a[i] - a[j]) for j in chosen))
        chosen.add(best)
    return sorted(chosen)[:n]


def stage_compute_smoke_fixture(opts, lay, rc) -> StageResult:
    if not CAP.probe_executable("newstack").available:
        return StageResult("compute_smoke_fixture", NOT_RUN_DEP, "newstack missing", {"missing": "newstack"})
    if not (rc.sources.final_tilt_file or rc.sources.raw_tilt_file):
        return StageResult("compute_smoke_fixture", FAIL, "no tilt file for selection")
    import numpy as np
    tlt = rc.sources.final_tilt_file or rc.sources.raw_tilt_file
    angles = [float(x) for x in Path(tlt).read_text().splitlines() if x.strip()]
    idx = _select_tilts(angles, opts.tilt_count)
    fdir = lay.fixtures / "compute_smoke"; fdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD")}
    # selected_tilts.tsv
    rows = ["orig_index\tangle"] + [f"{i}\t{angles[i]:.3f}" for i in idx]
    (fdir / "selected_tilts.tsv").write_text("\n".join(rows) + "\n")
    manifest = {"tilt_count": len(idx), "selected_indices": idx,
                "selected_angles": [angles[i] for i in idx], "subsets": {}, "commands": []}
    # subset the line-based metadata
    for role, p in (("tilt_file", rc.sources.final_tilt_file), ("raw_tilt_file", rc.sources.raw_tilt_file),
                    ("xtilt_file", rc.sources.xtilt_file), ("tltxf_file", rc.sources.tltxf_file),
                    ("final_xf", rc.sources.final_xf_file), ("defocus_file", rc.sources.defocus_file)):
        if p and Path(p).is_file():
            lines = [ln for ln in Path(p).read_text().splitlines() if ln.strip()]
            if len(lines) == len(angles):
                sub = fdir / Path(p).name
                sub.write_text("\n".join(lines[i] for i in idx) + "\n")
                manifest["subsets"][role] = {"path": str(sub), "rows": len(idx)}
    # subset + reduce the stacks
    out = {}
    for role in ("raw_stack", "aligned_stack"):
        src = getattr(rc.sources, role)
        if not src or not Path(src).is_file():
            continue
        hdr0 = read_mrc_header(src)
        secs = ",".join(str(i) for i in idx)
        sub = fdir / (Path(src).stem + "_sel.mrc")
        cmd = ["newstack", "-input", str(src), "-output", str(sub), "-secs", secs, "-float", "0"]
        cp = run_external_command(cmd, env=env, label=f"newstack -secs {Path(sub).name}",
                                  timeout_s=opts.command_timeout_s,
                                  deadline=opts.deadline_monotonic)
        manifest["commands"].append({"cmd": cmd, "rc": cp.returncode})
        if cp.returncode != 0 or not sub.is_file():
            return StageResult("compute_smoke_fixture", FAIL, f"newstack -secs failed: {cp.stderr[-200:]}")
        factor = _shrink_to(max(hdr0["shape_xy"]), opts.smoke_max_image_dim)
        red = fdir / (Path(src).stem + "_smoke.mrc")
        cmd2, cp2 = _newstack_shrink(sub, red, factor, env, opts)
        manifest["commands"].append({"cmd": cmd2, "rc": cp2.returncode})
        if cp2.returncode != 0 or not red.is_file():
            return StageResult("compute_smoke_fixture", FAIL, f"smoke reduce failed: {cp2.stderr[-200:]}")
        hdr1 = read_mrc_header(red)
        if not opts.keep_intermediates:
            sub.unlink(missing_ok=True)
        manifest["subsets"][role] = {"path": str(red), "header": hdr1, "shrink_factor": factor}
        out[role] = {"shape_xy": hdr1["shape_xy"], "n_sections": hdr1["n_sections"]}
        # section-correspondence invariant
        if hdr1["n_sections"] != len(idx):
            return StageResult("compute_smoke_fixture", FAIL,
                               f"{role}: sections {hdr1['n_sections']} != selected {len(idx)}")
    _atomic_json(fdir / "fixture_manifest.json", manifest)
    return StageResult("compute_smoke_fixture", PASS, f"{len(idx)} tilts <= {opts.smoke_max_image_dim}px",
                       {"selected": idx, "stacks": out})


def stage_statistics(opts, lay, rc) -> StageResult:
    if not CAP.probe_python_module("mrcfile").available:
        return StageResult("statistics", NOT_RUN_DEP, "mrcfile missing", {"missing": "mrcfile"})
    image_summary = {}
    per_tilt = []
    histos = {}
    # original + reduced stacks
    targets = []
    for role in ("raw_stack", "aligned_stack"):
        s = getattr(rc.sources, role)
        if s and Path(s).is_file():
            targets.append((role + "_source", s))
    for sub in ("geometry_all_tilts", "compute_smoke"):
        d = lay.fixtures / sub
        if d.is_dir():
            for mrc in sorted(d.glob("*.mrc")):
                targets.append((f"{sub}:{mrc.stem}", str(mrc)))
    import numpy as np
    # statistics are computed by BOUNDED SAMPLING (memory-safe on source-scale stacks);
    # the full data array is never loaded (§16). Each source stack is sampled ONCE.
    cache = lay.root / "diagnostics" / "_stat_cache"
    for label, path in targets:
        try:
            res = sample_mrc_statistics(
                path, max_sample_voxels=opts.max_sample_voxels, max_memory_mb=opts.max_memory_mb)
        except Exception as exc:
            image_summary[label] = {"error": str(exc)}; continue
        st = res["summary"]
        image_summary[label] = st
        # histogram from the per-section sampled percentiles (no full-array histogram)
        perc = st.get("percentiles_sampled", {})
        if perc:
            histos[label] = {"percentiles": perc, "min": st.get("min"), "max": st.get("max")}
        if "source" in label:
            for r in res["per_tilt"]:
                r["stack"] = label; per_tilt.append(r)
    _atomic_json(lay.statistics / "image_summary.json", image_summary)
    _atomic_json(lay.statistics / "intensity_histograms.json", histos)
    cols = ["stack", "index", "mean", "std", "min", "max", "median", "mad",
            "frac_zero", "frac_nonfinite", "robust_contrast"]
    lines = ["\t".join(cols)] + ["\t".join(str(r.get(c, "")) for c in cols) for r in per_tilt]
    (lay.statistics / "per_tilt_image_statistics.tsv").write_text("\n".join(lines) + "\n")

    # tilt statistics
    tlt = rc.sources.final_tilt_file or rc.sources.raw_tilt_file
    tilt_stats = {}
    if tlt and Path(tlt).is_file():
        a = sorted([float(x) for x in Path(tlt).read_text().splitlines() if x.strip()])
        steps = [round(a[i + 1] - a[i], 4) for i in range(len(a) - 1)]
        tilt_stats = {
            "count": len(a), "min_angle": a[0], "max_angle": a[-1],
            "median_step": float(np.median(steps)) if steps else None,
            "largest_gap": max(steps) if steps else None,
            "nearest_zero": min(a, key=abs), "positive": sum(1 for x in a if x > 0),
            "negative": sum(1 for x in a if x < 0),
            "duplicate_angles": len(a) - len(set(a)),
            "non_monotonic": any(a[i + 1] < a[i] for i in range(len(a) - 1)),
        }
    _atomic_json(lay.statistics / "tilt_statistics.json", tilt_stats)

    # xf statistics
    xf_stats = {"rows": []}
    if rc.sources.final_xf_file and Path(rc.sources.final_xf_file).is_file():
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from imod_affine import read_xf
        A, d = read_xf(rc.sources.final_xf_file)
        decs = [_xf_decompose(A[i], d[i]) for i in range(len(A))]
        xf_stats["rows"] = decs
        keys = ["determinant", "rotation_deg", "isotropic_scale", "anisotropy", "shear",
                "translation_norm", "condition_number"]
        agg = {}
        for k in keys:
            vals = np.array([row[k] for row in decs if np.isfinite(row[k])])
            if vals.size:
                med = float(np.median(vals))
                agg[k] = {"min": float(vals.min()), "max": float(vals.max()),
                          "mean": float(vals.mean()), "median": med,
                          "mad": float(np.median(np.abs(vals - med)))}
        xf_stats["aggregate"] = agg
        # point round-trip validation
        from imod_affine import xf_to_homogeneous
        nx, ny = rc.geometry.raw_shape_xyz[:2] if rc.geometry.raw_shape_xyz else (1024, 1024)
        pts = np.array([[nx / 2, ny / 2], [0, 0], [nx - 1, 0], [0, ny - 1], [nx - 1, ny - 1]], float)
        H = xf_to_homogeneous(A[0], d[0], (nx, ny), (nx, ny))
        ph = np.c_[pts, np.ones(len(pts))]
        rt = (np.linalg.inv(H) @ (H @ ph.T)).T[:, :2]
        xf_stats["roundtrip_max_err_px"] = float(np.abs(rt - pts).max())
        tsv = ["index\t" + "\t".join(keys)]
        for i, row in enumerate(decs):
            tsv.append(str(i) + "\t" + "\t".join(f"{row[k]:.5g}" for k in keys))
        (lay.statistics / "xf_per_tilt.tsv").write_text("\n".join(tsv) + "\n")
    _atomic_json(lay.statistics / "xf_statistics.json", xf_stats)
    return StageResult("statistics", PASS, f"{len(image_summary)} stacks, {len(per_tilt)} tilt rows",
                       {"stacks": len(image_summary)})


def stage_geometry_invariants(opts, lay, rc) -> StageResult:
    g = rc.geometry
    frames = {}

    def frame(name, shape_xy, pix, parent):
        frames[name] = {"shape_xy": list(shape_xy) if shape_xy else None, "pixel_size_A": pix,
                        "physical_A": [shape_xy[0] * pix, shape_xy[1] * pix] if (shape_xy and pix) else None,
                        "centre_convention": "(n-1)/2", "parent_frame": parent}
    if g.raw_shape_xyz:
        frame("source_raw_detector", g.raw_shape_xyz[:2], g.raw_pixel_size_A, None)
    if g.aligned_shape_xyz:
        frame("source_aligned_detector", g.aligned_shape_xyz[:2], g.aligned_pixel_size_A, "source_raw_detector")
    gfix = lay.fixtures / "geometry_all_tilts"
    if gfix.is_dir() and CAP.probe_python_module("mrcfile").available:
        for role, fr in (("raw", "debug_raw_detector"), ("ali", "debug_aligned_detector")):
            for mrc in gfix.glob(f"*{role}*_geomfix.mrc"):
                try:
                    h = read_mrc_header(mrc)
                    frame(fr, h["shape_xy"], h["pixel_size_A"],
                          "source_raw_detector" if role == "raw" else "source_aligned_detector")
                except Exception:
                    pass
    target_shape = g.target_volume_shape_xyz
    target_pix = g.target_pixel_size_A
    if target_shape and target_pix:
        frames["target_reconstruction_volume"] = {
            "shape_xyz": list(target_shape), "pixel_size_A": target_pix,
            "physical_A": [s * target_pix for s in target_shape], "parent_frame": "source_aligned_detector"}
    _atomic_json(lay.geometry / "frame_graph.json", {"frames": frames})
    _atomic_json(lay.geometry / "grids.json", {"frames": frames})
    _atomic_json(lay.geometry / "physical_dimensions.json",
                 {k: v.get("physical_A") for k, v in frames.items()})

    # validation incl. the doubled-volume failure mode
    checks = []
    state = PASS

    def check(name, ok, detail=""):
        nonlocal state
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            state = FAIL

    if g.raw_pixel_size_A and g.aligned_pixel_size_A:
        ratio = g.aligned_pixel_size_A / g.raw_pixel_size_A
        check("aligned_pixel_is_integer_multiple_of_raw", abs(ratio - round(ratio)) < 0.02,
              f"ratio={ratio:.4f}")
    # Warp volume invariant (2.10): if a produced XML exists, assert it; else check the
    # KNOWN failure mode on the planned geometry (raw_voxels x output_pixel == 2x target).
    if target_shape and target_pix:
        produced = list((lay.warp).rglob("*.xml"))
        import re
        if produced:
            for xmlp in produced:
                m = re.search(r'VolumeDimensionsAngstrom="([^"]+)"', xmlp.read_text())
                if m:
                    vol = [float(x) for x in m.group(1).split(",")]
                    ok = PC.volume_invariant_ok(vol, target_shape, target_pix)
                    check(f"warp_volume_invariant[{xmlp.name}]", ok,
                          f"xml={vol} target={[round(s*target_pix,1) for s in target_shape]}")
        # detect the doubled-volume mistake explicitly (raw shape x output pixel)
        if g.raw_shape_xyz:
            wrong = [g.raw_shape_xyz[0] * target_pix, g.raw_shape_xyz[1] * target_pix,
                     g.raw_shape_xyz[2] * target_pix] if len(g.raw_shape_xyz) == 3 else None
            if wrong:
                doubled = not PC.volume_invariant_ok(wrong, target_shape, target_pix)
                check("doubled_volume_mistake_is_detectable", doubled,
                      "raw_voxels x output_pixel differs from target (as it must)")
    if g.tilt_axis_angle_deg in (None, 0, 0.0):
        check("tilt_axis_nonzero", False, "tilt axis is 0/none")
    else:
        check("tilt_axis_nonzero", True, f"{g.tilt_axis_angle_deg} ({g.tilt_axis_source})")
    md = [f"# Geometry validation\n", f"state: {state}\n"]
    for c in checks:
        md.append(f"- [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    (lay.geometry / "geometry_validation.md").write_text("\n".join(md) + "\n")
    _atomic_json(lay.geometry / "geometry_validation.json", {"state": state, "checks": checks})
    return StageResult("geometry_invariants", state,
                       f"{sum(1 for c in checks if c['ok'])}/{len(checks)} checks", {"checks": checks})


def stage_previews(opts, lay, rc) -> StageResult:
    if not CAP.probe_python_module("matplotlib").available:
        return StageResult("previews", NOT_RUN_DEP, "matplotlib missing", {"missing": "matplotlib"})
    if not CAP.probe_python_module("mrcfile").available:
        return StageResult("previews", NOT_RUN_DEP, "mrcfile missing", {"missing": "mrcfile"})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    made = []
    smoke = lay.fixtures / "compute_smoke"
    sel = []
    selfile = smoke / "selected_tilts.tsv"
    if selfile.is_file():
        for ln in selfile.read_text().splitlines()[1:]:
            i, a = ln.split("\t"); sel.append((int(i), float(a)))

    def _disp(arr):  # consistent percentile display
        lo, hi = np.percentile(arr[np.isfinite(arr)], [1, 99]) if np.isfinite(arr).any() else (0, 1)
        return lo, hi

    # match reduced fixture stacks by whether they are the aligned (_ali) or raw stack
    smoke_mrcs = sorted(smoke.glob("*_smoke.mrc"))
    role_files = {"aligned": next((m for m in smoke_mrcs if "_ali" in m.name), None),
                  "raw": next((m for m in smoke_mrcs if "_ali" not in m.name), None)}
    for role, fname in (("raw", "raw_selected_tilts.png"), ("aligned", "aligned_selected_tilts.png")):
        mrc = role_files.get(role)
        if not mrc:
            continue
        try:
            h, arr = _measure_mrc(mrc)
        except Exception:
            continue
        n = min(arr.shape[0], 5)
        picks = np.linspace(0, arr.shape[0] - 1, n).astype(int)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
        if n == 1:
            axes = [axes]
        lo, hi = _disp(arr)
        for ax, k in zip(axes, picks):
            ax.imshow(arr[k], cmap="gray", vmin=lo, vmax=hi)
            ang = sel[k][1] if k < len(sel) else "?"
            ax.set_title(f"{role} #{k} {ang}", fontsize=8); ax.axis("off")
        fig.suptitle(f"{role} {h['shape_xy']} @ {h['pixel_size_A']:.3f} A/px", fontsize=9)
        fig.tight_layout(); fig.savefig(lay.previews / fname, dpi=80); plt.close(fig)
        made.append(fname)

    # per-tilt mean/std
    pst = lay.statistics / "per_tilt_image_statistics.tsv"
    if pst.is_file():
        rows = [ln.split("\t") for ln in pst.read_text().splitlines()[1:]]
        if rows:
            idx = [int(r[1]) for r in rows]; mean = [float(r[2]) for r in rows]; std = [float(r[3]) for r in rows]
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(idx, mean, ".-", label="mean"); ax.plot(idx, std, ".-", label="std")
            ax.set_xlabel("tilt index"); ax.legend(fontsize=8); ax.set_title("per-tilt mean/std", fontsize=9)
            fig.tight_layout(); fig.savefig(lay.previews / "per_tilt_mean_std.png", dpi=80); plt.close(fig)
            made.append("per_tilt_mean_std.png")

    # xf translation/rotation
    xfs = lay.statistics / "xf_statistics.json"
    if xfs.is_file():
        decs = json.loads(xfs.read_text()).get("rows", [])
        if decs:
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3))
            a1.plot([r["translation_x"] for r in decs], label="tx")
            a1.plot([r["translation_y"] for r in decs], label="ty"); a1.legend(fontsize=8)
            a1.set_title("xf translation", fontsize=9)
            a2.plot([r["rotation_deg"] for r in decs]); a2.set_title("xf rotation (deg)", fontsize=9)
            fig.tight_layout(); fig.savefig(lay.previews / "xf_translation_rotation.png", dpi=80); plt.close(fig)
            made.append("xf_translation_rotation.png")

    # sampled-percentile profiles (the statistics are sampled, not full histograms)
    hj = lay.statistics / "intensity_histograms.json"
    if hj.is_file():
        H = json.loads(hj.read_text())
        rows = [(lbl, hh.get("percentiles", {})) for lbl, hh in H.items() if hh.get("percentiles")]
        if rows:
            fig, ax = plt.subplots(figsize=(6, 3))
            for label, perc in rows[:4]:
                items = sorted(perc.items(), key=lambda kv: float(kv[0]))
                ax.plot([float(k) for k, _ in items], [v for _, v in items], ".-", label=label[:24])
            ax.set_xlabel("percentile"); ax.set_ylabel("sampled intensity")
            ax.legend(fontsize=7); ax.set_title("sampled intensity percentiles", fontsize=9)
            fig.tight_layout(); fig.savefig(lay.previews / "intensity_histograms.png", dpi=80); plt.close(fig)
            made.append("intensity_histograms.png")
    if not made:
        return StageResult("previews", WARNING, "no previews generated (no fixture stacks)")
    return StageResult("previews", PASS, f"{len(made)} PNGs", {"files": made})


def stage_exercise_prep(opts, lay, rc) -> StageResult:
    """Exercise the production preparation functions on the resolved TOML (no duplication)."""
    sub = {}
    # canonical TOML load
    try:
        rc2 = PC.load(lay.config / "project_settings.resolved.toml")
        rc2.require_resolved()
        sub["canonical_toml_load"] = PASS
    except Exception as exc:
        sub["canonical_toml_load"] = f"{FAIL}: {exc}"
    # MissAlignment YAML generation (stock; real generator)
    try:
        from pipeline.orchestrate import _load_03run
        r3 = _load_03run()
        smoke_yaml = lay.missalignment / "config.smoke.yaml"
        smoke_yaml.write_text(r3.config_text(lay.warp / "warp_smoke", "smoke"))
        std_yaml = lay.missalignment / "config.standard.yaml"
        std_yaml.write_text(r3.config_text(lay.warp / "warp_std", "standard"))
        sub["missalignment_yaml"] = PASS if "alignment: global" in smoke_yaml.read_text() else FAIL
    except Exception as exc:
        sub["missalignment_yaml"] = f"{FAIL}: {exc}"
    # result-adapter preflight (backend recognized)
    sub["result_backend"] = PASS if rc.result_backend in PC.RESULT_BACKENDS else f"{FAIL}: {rc.result_backend}"
    # Warp conversion availability
    wc = CAP.can_convert_warp()
    sub["warp_conversion"] = PASS if wc.available else NOT_RUN_DEP
    # IMOD reconstruction-plan generation (real reconstruction library)
    try:
        from reconstruction.command_files import build_tilt_com
        nx, ny = (rc.geometry.aligned_shape_xyz[:2] if rc.geometry.aligned_shape_xyz else (256, 256))
        com = build_tilt_com(in_stack="ali.mrc", out_rec="rec.mrc", tilt_file="x.tlt",
                             fullimage_xy=(nx, ny), thickness=max(1, nx // 4))
        sub["imod_recon_plan"] = PASS if "IMAGEBINNED 1" in com else FAIL
    except Exception as exc:
        sub["imod_recon_plan"] = f"{FAIL}: {exc}"
    _atomic_json(lay.diagnostics / "exercise_prep.json", sub)
    bad = [k for k, v in sub.items() if isinstance(v, str) and v.startswith(FAIL)]
    state = FAIL if bad else (WARNING if any(v == NOT_RUN_DEP for v in sub.values()) else PASS)
    return StageResult("exercise_prep", state, f"{len(sub)} substages", sub)


def stage_imod_mini_recon(opts, lay, rc) -> StageResult:
    if not opts.run_imod:
        return StageResult("imod_mini_recon", NOT_RUN_USER, "--no-debug-run-imod")
    if not CAP.probe_executable("tilt").available:
        return StageResult("imod_mini_recon", NOT_RUN_DEP, "IMOD tilt missing", {"missing": "tilt"})
    smoke = lay.fixtures / "compute_smoke"
    stacks = list(smoke.glob("*ali*_smoke.mrc")) or list(smoke.glob("*_smoke.mrc"))
    tlt = smoke / Path(rc.sources.final_tilt_file).name if rc.sources.final_tilt_file else None
    if not stacks or not (tlt and tlt.is_file()):
        return StageResult("imod_mini_recon", NOT_APPLICABLE, "no compute-smoke stack/tilt")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reconstruction.model import ReconstructionRequest
    from reconstruction.prepare import prepare_imod_reconstruction
    odir = lay.imod / "compute_smoke"; odir.mkdir(parents=True, exist_ok=True)
    h = read_mrc_header(stacks[0])
    nx, ny = h["shape_xy"]
    thickness = max(8, nx // 4)
    req = ReconstructionRequest(output_dir=str(odir), input_mode="aligned_stack",
                                aligned_stack=str(stacks[0]), tilt_file=str(tlt),
                                fullimage_xy=(nx, ny), thickness=thickness, execution="local",
                                basename="debug")
    try:
        res = prepare_imod_reconstruction(req)
    except Exception as exc:
        return StageResult("imod_mini_recon", FAIL, str(exc))
    val = {"input_sections": h["n_sections"], "fullimage_xy": [nx, ny], "thickness": thickness,
           "imagebinned": 1, "output_rec": res.output_rec, "executed": res.executed}
    if res.executed and Path(res.output_rec).is_file():
        oh, oarr = _measure_mrc(res.output_rec)
        import numpy as np
        val["output_header"] = oh
        val["finite"] = bool(np.isfinite(oarr).all())
        # central slice preview
        if CAP.probe_python_module("matplotlib").available:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            mid = oarr.shape[0] // 2
            fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(oarr[mid], cmap="gray"); ax.axis("off")
            ax.set_title("debug recon central slice", fontsize=9)
            fig.tight_layout(); fig.savefig(lay.previews / "debug_recon_central_slice.png", dpi=80); plt.close(fig)
    _atomic_json(odir / "reconstruction_validation.json", val)
    state = PASS if (res.executed and val.get("finite")) else WARNING
    return StageResult("imod_mini_recon", state, f"thickness={thickness}", val)


def stage_warp_diagnostics(opts, lay, rc) -> StageResult:
    wc = CAP.can_convert_warp()
    if not wc.available:
        return StageResult("warp_diagnostics", NOT_RUN_DEP, f"warpylib {wc.state}", {"missing": "warpylib"})
    # warpylib present: real conversion on both fixtures (cluster path; not exercised locally)
    return StageResult("warp_diagnostics", NOT_RUN_USER,
                       "warpylib present but full conversion deferred (run on the cluster preflight)")


def stage_debug_jobs(opts, lay, rc) -> StageResult:
    if not opts.generate_slurm:
        return StageResult("debug_jobs", NOT_RUN_USER, "--no-debug-generate-slurm")
    # stock mode only unless the constrained fork is available (2.15)
    fork = PC.constrained_fork_status()
    mode = rc.refinement_mode if rc.refinement_mode in PC.STOCK_REFINEMENT_MODES else "standard"
    cl = rc.cluster
    env_line = (f'export PATH="{cl.environment}/bin:$PATH"' if cl.environment else
                '# (no [cluster].environment; relying on a pre-activated shell)')
    smoke_yaml = lay.missalignment / "config.smoke.yaml"
    gres = cl.gres
    if cl.profile == "maxwell" and cl.partition == "vds" and gres == "gpu:1":
        gres = None
    gres_line = f"#SBATCH --gres={gres}\n" if gres else ""
    common = (f"#!/usr/bin/env bash\nset -Eeuo pipefail\n#SBATCH --partition={cl.partition}\n"
              f"#SBATCH --constraint={cl.constraint or 'V100'}\n{gres_line}"
              f"#SBATCH --time=0-01:00:00\n#SBATCH --cpus-per-task=8\n"
              f"#SBATCH --output={lay.logs}/%x_%j.log\n{env_line}\n"
              "date --iso-8601=seconds; hostname -f 2>/dev/null || hostname; which miss-alignment || true\n")
    (lay.jobs / "debug_preflight.sbatch").write_text(
        common + f"python {lay.root}/jobs/_probe.py --run-dir {lay.root} || exit 1\necho preflight-ok\n")
    probe_src = Path(__file__).resolve().parents[2] / "tools" / "cluster_capability_probe.py"
    if probe_src.is_file():
        (lay.jobs / "_probe.py").write_text(probe_src.read_text())
    # smoke uses the REDUCED smoke YAML + the NORMAL command (no --smoke flag)
    (lay.jobs / "debug_missalignment_smoke.sbatch").write_text(
        common + f'# compute-smoke fixture; stock mode={mode}; reduced YAML; normal command (no unsupported smoke flag)\n'
        f'miss-alignment --config-file {smoke_yaml} --training-devices 0 '
        f'--reconstruction-devices 0 --dataloaders-per-trainer 1 --start-at-iteration 0\n'
        f'python {lay.root}/jobs/_verdict.py --run-dir {lay.root}\n')
    cpu_part = f"#SBATCH --partition={cl.cpu_partition}\n" if cl.cpu_partition else ""
    (lay.jobs / "debug_result_validation.sbatch").write_text(
        "#!/usr/bin/env bash\nset -Eeuo pipefail\n" + cpu_part +
        f"#SBATCH --output={lay.logs}/%x_%j.log\n"
        f'python {lay.root}/jobs/_verdict.py --run-dir {lay.root}\n')
    (lay.jobs / "_verdict.py").write_text(_VERDICT_PY)
    (lay.jobs / "submit_debug.sh").write_text(
        "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
        f"# MANUAL submission. MissAlignment GPU smoke is NOT auto-submitted by --test-debug.\n"
        f"PRE=$(sbatch --parsable {lay.jobs}/debug_preflight.sbatch); echo preflight=$PRE\n"
        f"SMOKE=$(sbatch --parsable --dependency=afterok:$PRE {lay.jobs}/debug_missalignment_smoke.sbatch); echo smoke=$SMOKE\n"
        f"sbatch --dependency=afterok:$SMOKE {lay.jobs}/debug_result_validation.sbatch\n")
    for f in ("debug_preflight.sbatch", "debug_missalignment_smoke.sbatch",
              "debug_result_validation.sbatch", "submit_debug.sh"):
        (lay.jobs / f).chmod(0o755)
    (lay.jobs / "README_DEBUG.md").write_text(
        f"# Debug jobs\n\nNOT auto-submitted. To submit the GPU smoke manually:\n\n"
        f"    bash {lay.jobs}/submit_debug.sh\n\n"
        f"Stock mode `{mode}` (constrained fork: {'available' if fork['available'] else 'unavailable'}).\n"
        f"After the jobs finish, collect logs/results with:\n\n"
        f"    ./setup_missalign_project.py --test-debug-collect --debug-run {lay.root}\n")
    return StageResult("debug_jobs", PASS, f"stock mode={mode}; not submitted",
                       {"mode": mode, "submitted": False})


_VERDICT_PY = '''#!/usr/bin/env python3
import argparse, json, pathlib, re
ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); a = ap.parse_args()
rd = pathlib.Path(a.run_dir); ma = rd / "missalignment"
res = list(ma.rglob("*")) if ma.exists() else []
nonempty = [p for p in res if p.is_file() and p.stat().st_size > 0]
iters = list(ma.glob("**/iter*"))
bad = any(re.search(r"\\b(nan|inf)\\b", p.read_text(errors="ignore"), re.I)
          for p in (rd / "logs").rglob("*") if p.is_file() and p.suffix in (".log", ".txt"))
ok = (len(nonempty) > 0 or len(iters) > 0) and not bad
v = {"smoke": "ok" if ok else "failed",
     "verified": {"command_exit_zero": True, "output_files_nonempty": len(nonempty) > 0,
                  "iteration_outputs": len(iters) > 0, "no_nan_inf_in_logs": not bad},
     "note": "stock smoke establishes exit/output/iter/logs only; NOT constrained gradients."}
out = rd / "results" / "debug_smoke_verdict.json"; out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(v, indent=2)); print("verdict", v["smoke"]); raise SystemExit(0 if ok else 1)
'''


# --------------------------------------------------------------------------- #
# summary + bundle
# --------------------------------------------------------------------------- #
def _software_versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "mrcfile", "matplotlib", "torch", "warpylib"):
        c = CAP.probe_python_module(mod)
        out[mod] = c.version or ("present" if c.available else "absent")
    for exe in ("newstack", "tilt", "ctfphaseflip", "miss-alignment", "sbatch"):
        out[exe] = shutil.which(exe) or "absent"
    return out


def write_summary(opts, lay, rc, results, started, source_changed) -> dict:
    counts = {s: 0 for s in (PASS, FAIL, WARNING, NOT_RUN_DEP, NOT_RUN_USER, NOT_APPLICABLE)}
    for r in results:
        counts[r.state] = counts.get(r.state, 0) + 1
    failures = [{"stage": r.name, "detail": r.detail} for r in results if r.state == FAIL]
    summary = {
        "run_id": lay.root.name, "run_dir": str(lay.root),
        "start_utc": started, "end_utc": _utc(),
        "source_project": opts.data_dir, "basename": opts.basename,
        "config_hash": _config_hash(opts.data_dir, opts.basename, opts),
        "git_rev": _git_rev(), "software_versions": _software_versions(),
        "capabilities": CAP.probe_all(),
        "stages": [{"name": r.name, "state": r.state, "detail": r.detail} for r in results],
        "counts": counts, "failures": failures,
        "source_changed": source_changed,
        "resolved_toml": str(lay.config / "project_settings.resolved.toml"),
        "geometry": asdict(rc.geometry) if rc else {},
        "previews": [p.name for p in lay.previews.glob("*.png")],
        "recommended_next_actions": (
            ["FIX failing stages (see DEBUG_FAILURES.json)"] if failures else
            ["submit GPU smoke: bash %s/jobs/submit_debug.sh" % lay.root,
             "then: --test-debug-collect --debug-run %s" % lay.root]),
    }
    _atomic_json(lay.results / "DEBUG_SUMMARY.json", summary)
    _atomic_json(lay.results / "DEBUG_FAILURES.json", {"failures": failures})
    # markdown
    md = [f"# Debug summary — {lay.root.name}", "",
          f"- source: `{opts.data_dir}`", f"- basename: `{opts.basename}`",
          f"- resolved config: `config/project_settings.resolved.toml`",
          f"- source changed during run: **{source_changed}**", "",
          f"## Result counts", ""]
    for s in (PASS, WARNING, FAIL, NOT_RUN_DEP, NOT_RUN_USER, NOT_APPLICABLE):
        md.append(f"- {s}: {counts.get(s,0)}")
    md += ["", "## Stages", ""]
    for r in results:
        md.append(f"- **{r.name}**: {r.state} — {r.detail}")
    md += ["", "## Previews", ""]
    for p in sorted(lay.previews.glob("*.png")):
        md.append(f"- [{p.name}](../previews/{p.name})")
    md += ["", "## Cluster", "",
           "GPU smoke is NOT auto-submitted. To submit:",
           f"```\nbash {lay.root}/jobs/submit_debug.sh\n```",
           "After completion:",
           f"```\n./setup_missalign_project.py --test-debug-collect --debug-run {lay.root}\n```"]
    (lay.results / "DEBUG_SUMMARY.md").write_text("\n".join(md) + "\n")
    # stats tsv + file index
    (lay.results / "DEBUG_STATISTICS.tsv").write_text(
        "stage\tstate\tdetail\n" + "\n".join(f"{r.name}\t{r.state}\t{r.detail}" for r in results) + "\n")
    idx = ["path\tsize"]
    for p in sorted(lay.root.rglob("*")):
        if p.is_file():
            idx.append(f"{p.relative_to(lay.root)}\t{p.stat().st_size}")
    (lay.results / "DEBUG_FILE_INDEX.tsv").write_text("\n".join(idx) + "\n")
    return summary


_BUNDLE_EXCLUDE_SUFFIX = (".rec",)


def build_bundle(opts, lay, rc) -> Path:
    max_bytes = opts.bundle_max_mb * 1024 * 1024
    internal, shareable = {}, {}
    src_prefix = str(Path(opts.data_dir))

    def _redact_path(s):
        return s.replace(src_prefix, "<SOURCE>")

    bundle = lay.bundle / f"{opts.basename}_{lay.root.name}_debug_bundle.tar.gz"
    # candidate files: everything under the run dir except the bundle dir and big stacks
    candidates = []
    total = 0
    for p in sorted(lay.root.rglob("*")):
        if not p.is_file() or lay.bundle in p.parents:
            continue
        suf = p.suffix.lower()
        size = p.stat().st_size
        # exclude full reconstructions; include reduced fixture stacks only if small
        if suf in _BUNDLE_EXCLUDE_SUFFIX:
            continue
        if suf == ".mrc":
            # only reduced fixture stacks, and only if they fit the budget
            if "fixtures" not in str(p) or size > 20 * 1024 * 1024:
                continue
        if total + size > max_bytes:
            continue
        candidates.append(p); total += size
        rel = str(p.relative_to(lay.root))
        internal[rel] = str(p)
        shareable[rel] = _redact_path(str(p))
    _atomic_json(lay.bundle / "internal_paths.json", internal)
    _atomic_json(lay.bundle / "shareable_paths_redacted.json", shareable)
    with tarfile.open(bundle, "w:gz") as tar:
        for p in candidates:
            tar.add(p, arcname=str(p.relative_to(lay.root)))
        for extra in ("internal_paths.json", "shareable_paths_redacted.json"):
            ep = lay.bundle / extra
            if ep.is_file():
                tar.add(ep, arcname=f"bundle/{extra}")
    return bundle


# --------------------------------------------------------------------------- #
# main entry points
# --------------------------------------------------------------------------- #
def run_test_debug(opts: DebugOptions) -> int:
    started = _utc()
    # global wall-clock budget shared by every external command and the stage loop (§16/§17)
    opts.deadline_monotonic = time.monotonic() + float(opts.global_timeout_s)
    data_dir = Path(opts.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: --data-dir is not a directory: {data_dir}")
        return 2
    # basename inference (canonical discovery; fail on ambiguity)
    if not opts.basename:
        try:
            opts.basename, brep = DISC.infer_basename(data_dir)
            print(f"[test-debug] inferred basename: {opts.basename}")
        except DISC.DiscoveryError as exc:
            print(f"ERROR: {exc}")
            return 2
    run_id = opts.run_id or f"{_stamp()}_{_config_hash(str(data_dir), opts.basename, opts)}"
    root = Path(opts.out_dir) / "test_debug" / f"{opts.basename}_{run_id}"
    if root.exists() and not opts.force:
        print(f"ERROR: debug run already exists: {root} (use --force-debug to overwrite)")
        return 2
    lay = DebugLayout(root).create()
    print(f"[test-debug] run dir: {root}")

    results = []
    print("[test-debug] source inventory (resolved files only)", flush=True)
    results.append(stage_source_inventory(opts, lay))
    print("[test-debug] canonical config", flush=True)
    cfg_res = stage_canonical_config(opts, lay)
    results.append(cfg_res)
    if not cfg_res.ok():
        write_summary(opts, lay, None, results, started, source_changed=False)
        print("ERROR: canonical config failed; see results/DEBUG_FAILURES.json")
        return 1
    rc = cfg_res.data["config"]
    # quick mode (default): no all-tilt fixture, no IMOD reconstruction. --test-debug-full
    # enables them. Stages run under a persistent journal so the active stage is always
    # visible (§16: cannot appear frozen).
    stages = [stage_compute_smoke_fixture, stage_geometry_invariants,
              stage_exercise_prep, stage_warp_diagnostics, stage_debug_jobs]
    if not opts.quick:
        stages.insert(1, stage_statistics)
        stages.insert(3, stage_previews)
    if opts.all_tilts:
        stages.insert(0, stage_geometry_fixture)
    if opts.run_imod:
        stages.insert(len(stages) - 1, stage_imod_mini_recon)
    journal = {s.__name__: "PENDING" for s in stages}
    journal["source_inventory"] = journal["canonical_config"] = "PASS"
    _write_journal(lay, journal, active=None)
    started_stages = False
    for fn in stages:
        # --debug-from-stage: skip every stage until the named one (resume support, §17)
        if opts.from_stage and not started_stages:
            if fn.__name__ == opts.from_stage:
                started_stages = True
            else:
                journal[fn.__name__] = "SKIPPED"; _write_journal(lay, journal, active=None); continue
        if opts.only_stage and fn.__name__ != opts.only_stage:
            journal[fn.__name__] = "SKIPPED"; _write_journal(lay, journal, active=None); continue
        # global budget: never start a new stage past the deadline (§16)
        if time.monotonic() >= opts.deadline_monotonic:
            print(f"[test-debug] global timeout ({opts.global_timeout_s}s) reached; "
                  f"skipping remaining stages", flush=True)
            journal[fn.__name__] = "TIMEOUT"; _write_journal(lay, journal, active=None)
            results.append(StageResult(fn.__name__, FAIL, "global timeout before stage start"))
            continue
        journal[fn.__name__] = "RUNNING"; _write_journal(lay, journal, active=fn.__name__)
        stage_started = time.monotonic()
        print(f"[test-debug] {fn.__name__} ...", flush=True)
        try:
            r = fn(opts, lay, rc)
            results.append(r)
            journal[fn.__name__] = r.state
        except Exception as exc:
            import traceback
            (lay.diagnostics / f"{fn.__name__}_traceback.txt").write_text(traceback.format_exc())
            results.append(StageResult(fn.__name__, FAIL, str(exc)))
            journal[fn.__name__] = FAIL
        print(f"[test-debug] {fn.__name__}: {journal[fn.__name__]} "
              f"({time.monotonic() - stage_started:.1f}s)", flush=True)
        _write_journal(lay, journal, active=None)

    # final source verification (hard fail on any change)
    stage_source_inventory(opts, lay, final=True)
    h0 = json.loads((lay.source_inventory / "source_hashes.json").read_text())
    h1 = json.loads((lay.source_inventory / "source_hashes_final.json").read_text())
    source_changed = (set(h0) != set(h1) or any(
        h0.get(k, {}).get("sha256") != h1.get(k, {}).get("sha256") for k in set(h0) | set(h1)))
    results.append(StageResult("source_unchanged", FAIL if source_changed else PASS,
                               "source modified!" if source_changed else "source read-only OK"))

    summary = write_summary(opts, lay, rc, results, started, source_changed)
    bundle = build_bundle(opts, lay, rc)
    (Path(opts.out_dir) / "test_debug" / "LATEST_DEBUG_RUN").write_text(str(root) + "\n")

    # console report
    c = summary["counts"]
    print("\nDEBUG RUN COMPLETE")
    print(f"Run directory: {root}")
    print(f"Resolved config: {lay.config / 'project_settings.resolved.toml'}")
    print(f"Summary: {lay.results / 'DEBUG_SUMMARY.md'}")
    print(f"Preview directory: {lay.previews}")
    print(f"Debug bundle: {bundle}")
    print(f"\nPASS: {c.get(PASS,0)}\nWARNING: {c.get(WARNING,0)}\nFAIL: {c.get(FAIL,0)}\n"
          f"NOT RUN: {c.get(NOT_RUN_DEP,0)+c.get(NOT_RUN_USER,0)}")
    print("\nCluster smoke not submitted.\nTo submit:")
    print(f"  bash {lay.jobs / 'submit_debug.sh'}")
    print("\nAfter completion:")
    print(f"  ./setup_missalign_project.py --test-debug-collect --debug-run {root}")
    # nonzero only when a REQUIRED local diagnostic failed (warnings/not-run don't fail)
    required_fail = any(r.state == FAIL for r in results)
    return 1 if required_fail else 0


def collect_test_debug(debug_run: str) -> int:
    root = Path(debug_run)
    if not root.is_dir():
        print(f"ERROR: --debug-run not found: {root}")
        return 2
    lay = DebugLayout(root)
    collected = {"logs": [], "results": [], "checkpoints": [], "missalignment_logs": []}
    for log in (lay.logs).glob("*") if lay.logs.exists() else []:
        if log.is_file():
            collected["logs"].append(log.name)
    for r in (lay.results).glob("*.json") if lay.results.exists() else []:
        collected["results"].append(r.name)
    verdict = lay.results / "debug_smoke_verdict.json"
    coll = {"collected_utc": _utc(), "inventory": collected,
            "smoke_verdict": json.loads(verdict.read_text()) if verdict.is_file() else None}
    _atomic_json(lay.diagnostics / "collect_report.json", coll)
    # update the summary's stage list with the collected verdict (honest)
    sj = lay.results / "DEBUG_SUMMARY.json"
    if sj.is_file():
        s = json.loads(sj.read_text())
        s["collected"] = coll
        _atomic_json(sj, s)
    # rebuild the bundle (no source MRC required to collect logs)
    try:
        rc = PC.load(lay.config / "project_settings.resolved.toml")
        opts = DebugOptions(data_dir=rc.data_root, out_dir=str(root.parents[1]),
                            basename=rc.basename)
        bundle = build_bundle(opts, lay, rc)
        print(f"[collect] rebuilt bundle: {bundle}")
    except Exception as exc:
        print(f"[collect] bundle rebuild skipped: {exc}")
    print(f"[collect] updated {sj}")
    print(f"[collect] smoke verdict: {coll['smoke_verdict']}")
    return 0
