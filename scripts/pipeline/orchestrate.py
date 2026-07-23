#!/usr/bin/env python3
"""Resumable step-state orchestration for the binning + CTF + recon + affine2d
preparation. This is what ``prepare_imod_to_warp.py`` actually executes (it no
longer merely prints a delegation line).

Each step records input/command/output hashes in ``step_state.json``; a step is
skipped when its inputs+command are unchanged and its outputs still exist
(stale detection). ``--resume`` / ``--from-step`` / ``--only-step`` are honoured.

Real IMOD is used for binning (``newstack``), working CTF (``ctfphaseflip``), and
the working reconstruction (``tilt``). The MissAlignment ``affine2d`` RUN and the
Warp XML evaluation are NOT executed locally (no MissAlignment/warpylib/GPU) --
only their CPU-side configs/scripts are generated.
"""
from __future__ import annotations

import hashlib
import faulthandler
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imod_affine import (  # noqa: E402
    WARP_AXIS_ANGLE_CONVENTION_VERSION, read_xf, write_xf, xf_to_homogeneous, homogeneous_to_xf)
from multiresolution import Grid2D, build_plan  # noqa: E402
from multiresolution import transfer as _T  # noqa: E402

from . import ctf as C
from . import datastate as DS
from . import geometry as G

# project preparation (PREPARE) step order. NOTE: final_ctf is intentionally ABSENT here --
# final CTF depends on the post-MissAlignment refined alignment and therefore
# belongs to export/finalization, not preparation (see LOCAL_THREE_PHASE_*; §1.1).
# warp_convert/warp_validate populate and gate the MissAlignment training dir (§1.2).
STEP_ORDER = [
    "discover", "workspace", "working_raw", "working_xf", "working_aligned",
    "working_ctf", "working_selected", "working_reconstruction",
    "warp_convert", "warp_validate",
    "missalign_config", "slurm", "submission",
]


def _load_03run():
    """Load the real config/command generators from scripts/03_run_missalignment.py
    (the filename is not a valid module name, so use importlib)."""
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "03_run_missalignment.py"
    spec = importlib.util.spec_from_file_location("run_missalignment_03", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Above this size, step-staleness uses a bounded sampled fingerprint instead of a full
# read: a multi-GB tilt stack must never be streamed end-to-end just to decide whether a
# step is current (§17). Small files (com files, .xf, .tlt) keep an exact content hash.
_FULL_HASH_MAX_BYTES = 64 << 20   # 64 MiB
_SAMPLE_WINDOW = 1 << 20          # 1 MiB head/middle/tail windows
_STAGE_TIMEOUT_SECONDS = 30


class _StageWatchdog:
    def __init__(self, name: str, timeout: int = _STAGE_TIMEOUT_SECONDS):
        self.name = name
        self.timeout = timeout
        self._timer: threading.Timer | None = None

    def __enter__(self):
        print(f"[prepare] start: {self.name}", flush=True)
        self._timer = threading.Timer(self.timeout, self._dump)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._timer:
            self._timer.cancel()
        status = "failed" if exc_type else "done"
        print(f"[prepare] {status}: {self.name}", flush=True)
        return False

    def _dump(self):
        print(
            f"[watchdog] stage '{self.name}' exceeded {self.timeout}s; dumping Python stack",
            file=sys.stderr,
            flush=True,
        )
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


def _link_stack(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    try:
        dst.symlink_to(Path(src).resolve())
    except OSError as exc:
        raise RuntimeError(f"could not symlink source stack {src} -> {dst}: {exc}") from exc


def _hash_file(p: Path) -> str | None:
    p = Path(p)
    if not p.is_file():
        return None
    size = p.stat().st_size
    h = hashlib.sha256()
    if size <= _FULL_HASH_MAX_BYTES:
        with p.open("rb") as fh:
            for c in iter(lambda: fh.read(1 << 16), b""):
                h.update(c)
        return h.hexdigest()[:16]
    # large file: deterministic sampled fingerprint over (size, head, middle, tail).
    # Reads at most ~3 MiB regardless of file size; the 's:' prefix marks it as sampled.
    w = _SAMPLE_WINDOW
    h.update(str(size).encode())
    with p.open("rb") as fh:
        h.update(fh.read(w))                       # head
        fh.seek(max(0, size // 2 - w // 2))
        h.update(fh.read(w))                       # middle
        fh.seek(max(0, size - w))
        h.update(fh.read(w))                       # tail
    return "s:" + h.hexdigest()[:14]


def _hash_obj(o: Any) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _effective_binning(config: dict, args) -> int:
    toml_binning = int((config.get("multiresolution", {}) or {}).get("extra_projection_binning", 1))
    cli_binning = getattr(args, "extra_binning", None)
    if cli_binning is not None and int(cli_binning) != toml_binning:
        raise ValueError(
            "--extra-binning is a compatibility option and must match "
            f"[multiresolution].extra_projection_binning ({toml_binning}); got {cli_binning}."
        )
    return toml_binning


@dataclass
class StepState:
    path: Path
    data: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path):
        path = Path(path)
        return cls(path, json.loads(path.read_text()) if path.is_file() else {})

    def is_fresh(self, name, input_hash, cmd_hash, outputs):
        rec = self.data.get(name)
        if not rec or rec.get("status") != "ok":
            return False
        if rec.get("input_hash") != input_hash or rec.get("cmd_hash") != cmd_hash:
            return False
        return all(Path(o).exists() for o in outputs)

    def record(self, name, input_hash, cmd_hash, outputs, status="ok", extra=None):
        self.data[name] = {"status": status, "input_hash": input_hash, "cmd_hash": cmd_hash,
                           "outputs": [str(o) for o in outputs],
                           "output_hashes": {str(o): _hash_file(o) for o in outputs}, **(extra or {})}
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")


@dataclass
class OrchestrationResult:
    workspace: Path
    stacks: dict
    steps_run: list
    steps_skipped: list
    manifest_path: Path
    warnings: list = field(default_factory=list)


def _run(cmd, env=None):
    e = dict(os.environ if env is None else env)
    e.setdefault("IMOD_DIR", "/Applications/IMOD")
    return subprocess.run(cmd, env=e, text=True, capture_output=True)


def _newstack_version() -> str:
    if shutil.which("newstack") is None:
        raise RuntimeError(
            "newstack is not on PATH; load IMOD or add the IMOD bin directory before "
            "running projection binning (expected command: newstack -input INPUT -output OUTPUT -shrink FACTOR -float 0)."
        )
    try:
        cp = subprocess.run(["newstack", "-version"], text=True, capture_output=True, timeout=20)
        text = (cp.stdout or cp.stderr).strip()
        return text.splitlines()[0] if text else "newstack version unavailable"
    except Exception as exc:
        return f"newstack version unavailable: {exc}"


def _validate_binned_stack(*, source_measure, output_path: Path, factor: int) -> dict:
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"newstack output missing or empty: {output_path}")
    measured = G.measure_mrc_grid(output_path, role="working_raw")
    if measured.mode not in (0, 1, 2, 6):
        raise RuntimeError(f"newstack output has unsupported MRC mode {measured.mode}: {output_path}")
    if measured.n_sections != source_measure.n_sections:
        raise RuntimeError(
            f"newstack output section count {measured.n_sections} != source {source_measure.n_sections}"
        )
    expected_xy = (source_measure.shape_xy[0] // factor, source_measure.shape_xy[1] // factor)
    if measured.shape_xy != expected_xy:
        raise RuntimeError(f"newstack output dimensions {measured.shape_xy} != expected {expected_xy}")
    expected_pixel = source_measure.pixel_size_xy_A[0] * factor
    if abs(measured.pixel_size_xy_A[0] - expected_pixel) > max(1e-3, expected_pixel * 1e-5):
        raise RuntimeError(
            f"newstack output pixel size {measured.pixel_size_xy_A[0]} != expected {expected_pixel}"
        )
    return measured.to_dict()


def orchestrate(*, config: dict, out_dir: Path, data_dir: Path | None, basename: str,
                inputs: dict, args) -> OrchestrationResult:
    """Execute the preparation pipeline. ``inputs`` carries resolved source paths
    (source_raw, source_aligned, source_xf, tilt_file, defocus_file, ctf_com)."""
    # §6 ONE RunLayout: the workspace IS the run dir. The warp training dir orchestrate
    # produces (ws/warp/warp_<condition>) must be byte-identical to layout.training_dir,
    # which the smoke YAML, cluster jobs (preflight/smoke/full), and finalize all consume.
    # The historical `interoperability/multiresolution` nesting made orchestrate write the
    # Warp output + MissAlignment config where nothing downstream looked (§6 defect).
    ws = Path(out_dir)
    state = StepState.load(ws / "step_state.json")
    stacks: dict[str, DS.Stack] = {}
    steps_run, steps_skipped, warnings = [], [], []
    binning_manifest: dict[str, Any] = {}

    B = _effective_binning(config, args)
    ctf_mode = args.ctf_mode or config.get("ctf", {}).get("mode", "off")
    condition = (args.condition[0] if isinstance(args.condition, list) and args.condition else
                 args.condition) or (config.get("conversion", {}).get("initial_conditions", ["ali_identity"])[0])
    condition_uses_aligned = condition == "ali_identity"
    # --- MEASURED geometry (no (0,0)/1.0 fallback) ---
    geom = config.get("geometry", {})
    with _StageWatchdog("measure_source_geometry"):
        measured = G.measure_source_and_working(
            source_raw=inputs.get("source_raw"), source_aligned=inputs.get("source_aligned"))
    m = measured["measured"]
    if "source_aligned" not in m and "source_raw" not in m:
        raise G.GeometryError(
            "no readable source MRC stack (need [input].aligned_stack or raw_stack); "
            "geometry must be measured from a real header, never assumed (0,0)/1.0).")
    # raw and aligned grids are SEPARATE; the working binning is applied to each.
    src_raw_m = m.get("source_raw")
    src_ali_m = m.get("source_aligned")
    src_dims = (src_raw_m or src_ali_m).shape_xy
    p_src_A = (src_raw_m or src_ali_m).pixel_size_xy_A[0]
    src_ali_dims = (src_ali_m or src_raw_m).shape_xy
    p_src_ali_A = (src_ali_m or src_raw_m).pixel_size_xy_A[0]
    # config geometry, if present, is an assertion only (header is authoritative).
    if geom.get("raw_dimensions_xyz") and src_raw_m:
        disc = G.assert_or_override(src_raw_m, expected_shape_xy=tuple(geom["raw_dimensions_xyz"][:2]),
                                    expected_pixel_A=geom.get("raw_pixel_size_A"),
                                    force=bool(getattr(args, "force_geometry", False)))
        if disc:
            warnings.append("config geometry overridden by force: " + "; ".join(disc))

    only = getattr(args, "only_step", None)
    frm = getattr(args, "from_step", None)
    resume = getattr(args, "resume", False)

    def want(name):
        if only:
            return name == only
        if frm:
            return STEP_ORDER.index(name) >= STEP_ORDER.index(frm)
        return True

    def do(name, input_hash, cmd_hash, outputs, fn):
        if not want(name):
            return False
        if (resume or only or frm) and state.is_fresh(name, input_hash, cmd_hash, outputs):
            print(f"[prepare] skipped: {name}", flush=True)
            steps_skipped.append(name); return False
        with _StageWatchdog(name):
            fn(); state.record(name, input_hash, cmd_hash, outputs)
        steps_run.append(name); return True

    # ---- discover / validate ----
    has_aligned = bool(inputs.get("source_aligned"))
    C.validate_ctf_mode(ctf_mode, condition, has_aligned or ctf_mode in ("off", "final"))
    if B not in (1, 2, 4, 8):
        raise ValueError(f"--extra-binning must be 1,2,4,8; got {B}")
    if B > 1:
        # divisibility gate on the MEASURED dims of every stack that will be binned
        build_plan(B, Grid2D.axis_aligned("source_raw", src_dims, p_src_A))
        build_plan(B, Grid2D.axis_aligned("source_aligned", src_ali_dims, p_src_ali_A))

    # ---- workspace ----
    with _StageWatchdog("workspace"):
        for sub in ("source", "working_raw", "working_aligned", "working_imod/ctf", "working_imod",
                    "missalignment/results", "final_imod", "preview", "restore", "provenance"):
            (ws / sub).mkdir(parents=True, exist_ok=True)

    p_work_A = p_src_A * B
    work_dims = (src_dims[0] // B, src_dims[1] // B)

    # ---- working_raw (real newstack reduction of the source raw stack) ----
    src_raw = inputs.get("source_raw")
    wr_path = ws / "working_raw" / f"{basename}_raw_bin{B}.mrc"
    if src_raw and Path(src_raw).is_file():
        binning_manifest = {
            "factor": B,
            "source_stack": str(src_raw),
            "source_stack_hash": _hash_file(src_raw),
            "source_dimensions_xy": list(src_dims),
            "source_sections": int((src_raw_m or src_ali_m).n_sections),
            "source_pixel_size_A": p_src_A,
            "working_pixel_size_A": p_work_A,
            "source_frame": (src_raw_m or src_ali_m).grid.to_dict(),
            "affine_conversion_method": "homogeneous_grid_transfer:h0_working=inv(G_a)@H0_source@G_r",
        }
        if B > 1:
            cmd = ["newstack", "-input", str(src_raw), "-output", str(wr_path), "-shrink", str(float(B)), "-float", "0"]
            newstack_version = _newstack_version()
            def _mk_wr():
                cp = _run(cmd)
                if cp.returncode != 0:
                    raise RuntimeError(f"working_raw newstack failed: {' '.join(cmd)}\n{cp.stderr[-300:]}")
                measured_wr = _validate_binned_stack(source_measure=src_raw_m or src_ali_m,
                                                     output_path=wr_path, factor=B)
                binning_manifest.update({
                    "derived_stack": str(wr_path),
                    "derived_stack_hash": _hash_file(wr_path),
                    "working_dimensions_xy": measured_wr["shape_xy"],
                    "working_sections": measured_wr["n_sections"],
                    "newstack_command": cmd,
                    "newstack_version": newstack_version,
                    "working_frame": measured_wr["grid"],
                })
            do("working_raw", _hash_file(src_raw), _hash_obj(cmd), [wr_path], _mk_wr)
            if wr_path.is_file():
                measured_wr_now = G.measure_mrc_grid(wr_path, role="working_raw")
                work_dims = measured_wr_now.shape_xy
                p_work_A = measured_wr_now.pixel_size_xy_A[0]
                binning_manifest.update({
                    "derived_stack": str(wr_path),
                    "derived_stack_hash": _hash_file(wr_path),
                    "working_dimensions_xy": list(work_dims),
                    "working_sections": measured_wr_now.n_sections,
                    "newstack_command": cmd,
                    "newstack_version": newstack_version,
                    "working_frame": measured_wr_now.grid.to_dict(),
                })
        else:
            # extra_binning=1: record explicit reuse (no copy), not a fake reduction
            wr_path = Path(src_raw)
            steps_run.append("working_raw") if "working_raw" not in steps_skipped else None
            binning_manifest.update({
                "derived_stack": str(wr_path),
                "derived_stack_hash": _hash_file(wr_path),
                "working_dimensions_xy": list(src_dims),
                "working_sections": int((src_raw_m or src_ali_m).n_sections),
                "newstack_command": None,
                "newstack_version": None,
                "working_frame": (src_raw_m or src_ali_m).grid.to_dict(),
            })
        if wr_path.is_file():
            stacks["working_raw"] = DS.Stack(
                role="working_raw", path=str(wr_path), alignment_state="raw", ctf_state="uncorrected",
                binning_state="working" if B > 1 else "source", intended_use="input",
                source_parent=str(src_raw), grid=binning_manifest.get("working_frame"),
                interpolation_history=(["bin%d" % B] if B > 1 else []),
                created_by_command=("newstack -shrink %d" % B) if B > 1 else "reuse source raw")

    # ---- working_xf (convert source .xf to the working grid, full homogeneous) ----
    src_xf = inputs.get("source_xf")
    wxf_path = ws / "working_aligned" / f"{basename}_raw_to_aligned_bin{B}.xf"
    affine_roundtrip_error = 0.0
    if src_xf and Path(src_xf).is_file() and B > 1:
        sr_grid = Grid2D.axis_aligned("source_raw", src_dims, p_src_A)
        sa_grid = Grid2D.axis_aligned("source_aligned", src_ali_dims, p_src_ali_A)
        from multiresolution import integer_binned_grid
        wr_grid = integer_binned_grid(sr_grid, B, out_shape_xy=work_dims)
        wa_grid = integer_binned_grid(sa_grid, B)
        G_r = wr_grid.mapping_to(sr_grid); G_a = wa_grid.mapping_to(sa_grid)
        def _mk_wxf():
            nonlocal affine_roundtrip_error
            A0, d0 = read_xf(src_xf); Aw, dw = [], []
            errs = []
            for i in range(len(A0)):
                H0 = xf_to_homogeneous(A0[i], d0[i], sr_grid.shape_xy, sa_grid.shape_xy)
                H0w = _T.h0_working(H0, G_r, G_a)   # inv(G_a) @ H0_source @ G_r (NOT translation scaling)
                H0rt = _T.h0_source_from_working(H0w, G_r, G_a)
                errs.append(float(np.max(np.abs(H0rt - H0))))
                a, d = homogeneous_to_xf(H0w, wr_grid.shape_xy, wa_grid.shape_xy)
                Aw.append(a); dw.append(d)
            write_xf(wxf_path, np.asarray(Aw), np.asarray(dw))
            affine_roundtrip_error = max(errs) if errs else 0.0
            binning_manifest["transformed_xf"] = str(wxf_path)
            binning_manifest["source_xf"] = str(src_xf)
            binning_manifest["round_trip_error"] = affine_roundtrip_error
            binning_manifest["maps"] = {
                "G_r_working_raw_to_source_raw": G_r.tolist(),
                "G_a_working_aligned_to_source_aligned": G_a.tolist(),
            }
        do("working_xf", _hash_file(src_xf), _hash_obj(["h0_working", B]), [wxf_path], _mk_wxf)

    # ---- working aligned: create only when the selected path needs it ----
    wa_unc = ws / "working_aligned" / f"{basename}_ali_bin{B}_uncorrected.mrc"
    src_ali = inputs.get("source_aligned")
    need_working_aligned = (condition_uses_aligned or ctf_mode in ("working", "both")
                            or bool(getattr(args, "working_reconstruction", False)))
    if need_working_aligned and src_ali and Path(src_ali).is_file():
        if B > 1:
            cmd = ["newstack", "-input", str(src_ali), "-output", str(wa_unc),
                   "-shrink", str(float(B)), "-float", "0"]
            def _mk_wa():
                cp = _run(cmd)
                if cp.returncode != 0:
                    raise RuntimeError(f"working aligned newstack failed: {cp.stderr[-300:]}")
            do("working_aligned", _hash_file(src_ali), _hash_obj(cmd), [wa_unc], _mk_wa)
            created_by = " ".join(cmd)
            history = ["etomo_align", f"bin{B}"]
            binning_state = "working"
        else:
            # No hidden full-stack copy when no extra binning is requested.
            wa_unc = Path(src_ali)
            if "working_aligned" not in steps_run:
                steps_run.append("working_aligned")
            created_by = "reuse source aligned stack"
            history = ["etomo_align"]
            binning_state = "source"
        if wa_unc.is_file():
            stacks["working_aligned_uncorrected"] = DS.Stack(
                role="working_aligned_uncorrected", path=str(wa_unc), alignment_state="working_aligned",
                ctf_state="uncorrected", binning_state=binning_state,
                intended_use="missalignment_input", source_parent=str(src_ali),
                interpolation_history=history, created_by_command=created_by,
                allowed_for_missalignment=True)
    elif not need_working_aligned:
        steps_skipped.append("working_aligned:not_required_for_condition")

    # ---- CTF parameters: the project ctfcorrection.com is AUTHORITATIVE (defect #9) ----
    ctf_cfg = config.get("ctf", {})
    ctf_params = {"voltage_kv": 300, "cs_mm": 2.7, "amplitude_contrast": 0.07, "axis_angle_deg": 0.0}
    ctf_param_source = {k: "config_default" for k in ctf_params}
    com_path = inputs.get("ctf_com")
    if com_path and Path(com_path).is_file():
        parsed = C.parse_ctf_com_params(Path(com_path))
        for key, pkey in (("voltage_kv", "voltage_kv"), ("cs_mm", "cs_mm"),
                          ("amplitude_contrast", "amplitude_contrast"), ("axis_angle_deg", "axis_angle_deg")):
            if parsed.get(pkey) is not None:
                ctf_params[key] = parsed[pkey]; ctf_param_source[key] = "ctfcorrection.com"
    # config may assert but only overrides with a recorded warning
    for key in ("voltage_kv", "cs_mm", "amplitude_contrast"):
        if key in ctf_cfg and ctf_param_source[key] == "ctfcorrection.com" and \
                float(ctf_cfg[key]) != float(ctf_params[key]):
            warnings.append(f"ctf.{key} config {ctf_cfg[key]} disagrees with ctfcorrection.com "
                            f"{ctf_params[key]}; using the .com (authoritative)")
        elif key in ctf_cfg and ctf_param_source[key] == "config_default":
            ctf_params[key] = type(ctf_params[key])(ctf_cfg[key])

    # ---- working CTF ----
    wa_ctf_stack = None
    if ctf_mode in ("working", "both") and "working_aligned_uncorrected" in stacks:
        wa_ctf = ws / "working_aligned" / f"{basename}_ali_bin{B}_ctf.mrc"
        cmd = None
        if inputs.get("defocus_file") and inputs.get("tilt_file"):
            cmd = C.build_ctfphaseflip_cmd(
                input_stack=wa_unc, output_stack=wa_ctf,
                angle_file=Path(inputs["tilt_file"]), defocus_file=Path(inputs["defocus_file"]),
                pixel_size_A=p_work_A, unbinned_pixel_A=p_src_A,
                voltage_kv=int(ctf_params["voltage_kv"]),
                cs_mm=float(ctf_params["cs_mm"]),
                amp_contrast=float(ctf_params["amplitude_contrast"]))
            def _mk_ctf():
                cp = C.run_ctfphaseflip(cmd)
                if cp.returncode != 0 or not wa_ctf.is_file():
                    raise RuntimeError(f"ctfphaseflip failed: {cp.stdout[-300:]}{cp.stderr[-200:]}")
            do("working_ctf", _hash_file(wa_unc), _hash_obj(cmd), [wa_ctf], _mk_ctf)
            if wa_ctf.is_file():
                wa_ctf_stack = DS.Stack(
                    role="working_aligned_ctf", path=str(wa_ctf), alignment_state="working_aligned",
                    ctf_state="phase_flipped", binning_state="working" if B > 1 else "source",
                    intended_use="missalignment_input", source_parent="working_aligned_uncorrected",
                    interpolation_history=stacks["working_aligned_uncorrected"].interpolation_history,
                    created_by_command=" ".join(cmd), allowed_for_missalignment=True)
                stacks["working_aligned_ctf"] = wa_ctf_stack
        else:
            warnings.append("ctf.mode working/both requested but no .defocus/.tlt discovered; CTF skipped")

    # ---- final CTF is DEFERRED to export/finalization, NOT run in preparation ----
    # final CTF must be applied to the post-MissAlignment refined source aligned
    # stack, which does not exist yet in project preparation. We only RECORD the request here;
    # export_warp_to_imod.py finalize performs it after the refined alignment exists.
    final_ctf_stack = None
    if ctf_mode in ("final", "both"):
        warnings.append("ctf.mode final/both: final CTF is deferred to export/finalization — "
                        "it depends on the post-MissAlignment refined alignment, not the source stack.")

    # ---- working selected ----
    if "working_aligned_uncorrected" in stacks:
        sel = DS.select_for_missalignment(stacks["working_aligned_uncorrected"], wa_ctf_stack, ctf_mode)
        stacks["working_selected"] = sel

    # ---- working reconstruction (REAL local tilt on the selected working stack) ----
    if getattr(args, "working_reconstruction", False) and "working_selected" in stacks:
        if not (inputs.get("tilt_file") and Path(inputs["tilt_file"]).is_file()):
            raise RuntimeError("--working-reconstruction requires a tilt-angle file (.tlt); none discovered")
        from multiresolution.workflow import tilt_working_com
        wsel = stacks["working_selected"]
        work_grid = Grid2D.axis_aligned("working_aligned", work_dims, (p_work_A, p_work_A))
        rec_cfg = config.get("reconstruction", {})
        thk_src = int(rec_cfg.get("thickness_source_px", config.get("multiresolution", {}).get("thickness_source_px", 0)))
        nz = max(1, thk_src // B) if thk_src else max(1, work_dims[0] // 2)
        rec_path = ws / "working_imod" / f"{basename}_rec_bin{B}.mrc"
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        tilt_com = ws / "working_imod" / f"{basename}_tilt_bin{B}.com"
        com_text = tilt_working_com(in_stack=str(wsel.path), out_rec=str(rec_path),
                                    tilt_file=str(inputs["tilt_file"]), working=work_grid, nz=nz)
        def _mk_rec():
            tilt_com.write_text(com_text)
            # Execute tilt via StandardInput (directives only; never edit a source tilt.com).
            directives = "\n".join(ln for ln in com_text.splitlines()
                                   if ln and not ln.startswith("#") and not ln.startswith("$")) + "\n"
            cp = subprocess.run(["tilt", "-StandardInput"], input=directives, text=True,
                                capture_output=True, env={**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD")})
            if cp.returncode != 0 or not rec_path.is_file():
                raise RuntimeError(f"working tilt reconstruction failed: {cp.stdout[-300:]}{cp.stderr[-200:]}")
        do("working_reconstruction", _hash_file(wsel.path), _hash_obj([com_text]), [rec_path], _mk_rec)
        if rec_path.is_file():
            stacks["working_reconstruction"] = DS.Stack(
                role="working_reconstruction", path=str(rec_path), alignment_state="working_aligned",
                ctf_state=wsel.ctf_state, binning_state="working" if B > 1 else "source",
                intended_use="working_qc", source_parent=str(wsel.path),
                interpolation_history=list(wsel.interpolation_history) + ["tilt"],
                created_by_command="tilt -StandardInput")

    # ---- Warp conversion + validation (populate + gate the training dir; §1.2) ----
    refinement_mode = args.refinement_mode or config.get("missalignment", {}).get("refinement_mode", "standard")
    with _StageWatchdog("load_missalignment_config_helpers"):
        _r3 = _load_03run()
    valid_modes = ("smoke", "standard", "translation", "rigid", "similarity", "affine2d")
    if refinement_mode not in valid_modes:
        raise ValueError(f"refinement_mode {refinement_mode!r} not in {valid_modes}")
    res_dir = ws / "missalignment" / "results" / condition / refinement_mode
    res_dir.mkdir(parents=True, exist_ok=True)
    # the warp training directory MissAlignment consumes.
    warp_parent = ws / "warp"
    training_dir = warp_parent / f"warp_{condition}"
    training_dir.mkdir(parents=True, exist_ok=True)

    from . import capabilities as CAP
    from . import project_config as PC
    warp_cap = CAP.can_convert_warp()
    # 2.7/§7: the converter inputs (stack, .xf, alignment mode, axis frame) come from the
    # CONDITION via the single adapter — never from refinement_mode, never an identity .xf
    # for an *_xf condition, never `working_selected` chosen blindly.
    warp_alignment_mode = (config.get("conversion", {}).get("condition_modes", {}) or {}).get(condition)
    if not warp_alignment_mode:
        warp_alignment_mode = PC.warp_alignment_mode_for(condition)
    geomcfg = config.get("geometry", {})
    # Canonical IMOD tilt.com positioning carried by the resolved config. Present -> the
    # ImodPositioning object feeds the converter and its hash keys the cache; absent (old
    # project) -> None keeps the prior no-positioning behaviour.
    from geometry.imod_positioning import (
        IMOD_TO_WARP_TILT_ANGLE_SIGN, from_toml_table as _pos_from_toml,
        validate_tilt_angle_sign)
    _pos_table = geomcfg.get("imod_positioning")
    positioning = _pos_from_toml(_pos_table) if _pos_table else None
    positioning_hash = positioning.positioning_hash() if positioning else "none"
    level_angle_x_sign = int((config.get("geometry", {}).get("imod_positioning", {}) or {}).get(
        "level_angle_x_sign", -1))
    # The ONE canonical IMOD->Warp tilt-angle sign (angles + OFFSET). From the positioning
    # table when present, else the documented default; keyed into the conversion cache.
    imod_tilt_angle_sign = validate_tilt_angle_sign(
        (_pos_table or {}).get("imod_to_warp_tilt_angle_sign", IMOD_TO_WARP_TILT_ANGLE_SIGN))
    eff_geom = PC.Geometry(**{k: geomcfg.get(k) for k in PC.Geometry().__dict__})
    ci = PC.condition_input_from_paths(
        condition, raw_stack=inputs.get("source_raw"), aligned_stack=inputs.get("source_aligned"),
        final_xf_file=inputs.get("source_xf"), final_tilt_file=inputs.get("tilt_file"),
        warp_mode=warp_alignment_mode, geometry=eff_geom, require_files=False)
    axis_frame = ci.axis_frame
    # which working stack feeds the converter: raw-grid conditions consume the working RAW
    # stack + the real source .xf; aligned-grid conditions consume the (CTF-)selected working
    # aligned stack + identity.
    if ci.stack_grid == "raw":
        stage_stack = stacks.get("working_raw")
        xf_staged = (str(wxf_path) if (B > 1 and Path(wxf_path).is_file()) else ci.source_xf)
    else:
        stage_stack = stacks.get("working_selected") or stacks.get("working_aligned_uncorrected")
        xf_staged = None        # already aligned -> identity (ali_identity)
    # 2.11: tilt axis from MEASURED geometry (align.com RotationAngle), never silent 0.0.
    tilt_axis = geomcfg.get("tilt_axis_angle_deg")
    warp_state = {"capability": warp_cap.to_dict(), "training_dir": str(training_dir),
                  "warp_alignment_mode": warp_alignment_mode, "tilt_axis_angle_deg": tilt_axis,
                  "condition_input": ci.to_dict(), "staged_xf": xf_staged,
                  "converted_locally": False, "validated": False, "required": True}
    if not ci.is_identity and not xf_staged:
        raise RuntimeError(
            f"condition {condition!r} requires the source raw->aligned .xf but none was resolved "
            f"(source_xf missing). Refusing to convert with an identity transform (§7).")
    local_warp_convert = bool(getattr(args, "local_warp_convert", False))
    if local_warp_convert and warp_cap.available and stage_stack and inputs.get("tilt_file"):
        if tilt_axis in (None, "", 0, 0.0):
            raise RuntimeError(
                "warp_convert requires a measured tilt_axis_angle_deg (align.com RotationAngle); "
                "got 0/none. Run `init` to resolve it, or set [geometry].tilt_axis_angle_deg "
                "explicitly. Refusing to convert with a 0.0 axis (defect 2.11).")
        out_pix = float(geomcfg.get("target_pixel_size_A") or p_work_A)
        # 2.10: physical volume = TARGET shape (not raw shape) so it does not double.
        target_shape = geomcfg.get("target_volume_shape_xyz") or [work_dims[0], work_dims[1],
                                                                  max(1, work_dims[0] // 2)]
        warp_state["axis_frame"] = axis_frame
        warp_state["target_volume_shape_xyz"] = list(target_shape)
        warp_state["target_volume_frame"] = "imod_reconstruction_mrc_xyz__y_is_thickness"
        warp_state["target_pixel_size_A"] = out_pix
        warp_state["stage_stack"] = stage_stack.path
        # Build the TS_<series>/ staging dir the converter requires (2.5):
        # TS_<...>.st / .rawtlt / .xf / .source.xf, then call with the correct signature.
        ts_name = f"TS_{basename}_{condition}"
        ts_dir = warp_parent / "staging" / ts_name
        def _mk_warp():
            import importlib
            e2w = importlib.import_module("etomo_to_warp")
            ts_dir.mkdir(parents=True, exist_ok=True)
            _link_stack(Path(stage_stack.path), ts_dir / f"{ts_name}.st")
            shutil.copy2(inputs["tilt_file"], ts_dir / f"{ts_name}.rawtlt")
            import numpy as _np
            n_t = sum(1 for ln in Path(inputs["tilt_file"]).read_text().splitlines() if ln.strip())
            if xf_staged and Path(xf_staged).is_file():
                # the REAL source raw->aligned transform (NOT identity) — full-affine condition
                A0, d0 = read_xf(xf_staged)
                if len(A0) != n_t:
                    raise RuntimeError(
                        f"staged .xf {xf_staged} has {len(A0)} rows != {n_t} tilts; refusing to "
                        "convert a mismatched transform (§7).")
                write_xf(ts_dir / f"{ts_name}.xf", A0, d0)
                write_xf(ts_dir / f"{ts_name}.source.xf", A0, d0)
            else:
                write_xf(ts_dir / f"{ts_name}.xf", _np.stack([_np.eye(2)] * n_t), _np.zeros((n_t, 2)))
                write_xf(ts_dir / f"{ts_name}.source.xf", _np.stack([_np.eye(2)] * n_t), _np.zeros((n_t, 2)))
            e2w.process_tilt_series(
                folder_path=ts_dir, output_directory=training_dir,
                tilt_axis_angle=float(tilt_axis), volume_shape=tuple(target_shape),
                output_pixel_size=out_pix, alignment_mode=warp_alignment_mode,
                axis_frame=axis_frame, grid_shape_xy=(5, 5),
                positioning=positioning, level_angle_x_sign=level_angle_x_sign,
                imod_to_warp_tilt_angle_sign=imod_tilt_angle_sign)
        # 2.6: NO swallow. Conversion failure is a hard error (the MissAlignment job
        # must never run against an empty training dir). The positioning hash + tilt-angle
        # sign are part of the command identity so any change (incl. the angle sign)
        # reconverts and a sign +1 XML is treated as stale.
        do("warp_convert", _hash_file(stage_stack.path),
           _hash_obj(["etomo_to_warp", warp_alignment_mode, axis_frame, tilt_axis,
                      _hash_file(xf_staged) if xf_staged else "identity",
                      positioning_hash, level_angle_x_sign, imod_tilt_angle_sign,
                      "warp_axis_convention_v", WARP_AXIS_ANGLE_CONVENTION_VERSION]),
           list(training_dir.glob("*.xml")) or [training_dir / "_converted.marker"], _mk_warp)
        warp_state["converted_locally"] = True
        def _mk_warp_validate():
            _r3.check_warp_dir(training_dir)  # reuse the EXISTING validator (raises if invalid)
            # Validate in the CURRENT Warp XYZ frame recorded by the converter.
            conversion_manifest = training_dir / f"{ts_name}.conversion.json"
            if not conversion_manifest.is_file():
                raise RuntimeError(f"missing conversion manifest: {conversion_manifest}")
            converted = json.loads(conversion_manifest.read_text())
            frame = converted.get("volume_frame") or {}
            version = int(frame.get("contract_version", 0))
            quarter_turn_k = int(frame.get("projection_quarter_turn_k", 0)) % 4
            if version >= 2:
                warp_shape = (
                    frame.get("reconstruction_shape_warp_xyz")
                    or frame.get("current_shape_warp_xyz")
                )
            elif version == 1 and quarter_turn_k % 2 == 0:
                warp_shape = (
                    frame.get("base_shape_warp_xyz")
                    or frame.get("current_shape_warp_xyz")
                )
            else:
                warp_shape = None
            if not warp_shape:
                raise RuntimeError(
                    f"legacy/stale volume-frame contract: {conversion_manifest}")
            expected = [
                float(v) * float(converted["output_pixel_size_A"])
                for v in warp_shape
            ]
            import re as _re
            for xmlp in training_dir.glob("*.xml"):
                m = _re.search(r'VolumeDimensionsAngstrom="([^"]+)"', xmlp.read_text())
                if m:
                    got = [float(x) for x in m.group(1).split(",")]
                    for observed, wanted in zip(got, expected):
                        if wanted and abs(observed - wanted) / wanted > 0.02:
                            raise RuntimeError(
                                f"{xmlp.name} volume {got} != Warp reconstruction XYZ {expected}")
            (training_dir / "_validated.marker").write_text("ok\n")
        do("warp_validate", _hash_obj([str(training_dir)]), _hash_obj(["check_warp_dir", "vol_invariant"]),
           [training_dir / "_validated.marker"], _mk_warp_validate)
        warp_state["validated"] = True
    else:
        # warpylib not available locally -> generate a REAL dedicated conversion job
        # (2.6). The MissAlignment job depends on it; nothing proceeds with an empty dir.
        # We do NOT copy the (multi-GB) stack here; instead we write a staging manifest the
        # cluster conversion job (§9) consumes to stage + convert with the CORRECT inputs:
        # the condition's stack and the REAL source .xf (never identity for *_xf, §7).
        out_pix = float(geomcfg.get("target_pixel_size_A") or p_work_A)
        target_shape = geomcfg.get("target_volume_shape_xyz") or [work_dims[0], work_dims[1],
                                                                  max(1, work_dims[0] // 2)]
        staging_manifest = {
            "series_name": basename,
            "condition": condition, "warp_alignment_mode": warp_alignment_mode,
            "axis_frame": axis_frame, "tilt_axis_angle_deg": tilt_axis,
            "target_volume_shape_xyz": list(target_shape),
            "target_volume_frame": "imod_reconstruction_mrc_xyz__y_is_thickness",
            "target_pixel_size_A": out_pix,
            "input_stack": (stage_stack.path if stage_stack else
                            (inputs.get("source_raw") if ci.stack_grid == "raw"
                             else inputs.get("source_aligned"))),
            "tilt_file": inputs.get("tilt_file"),
            "staged_xf": xf_staged, "is_identity": ci.is_identity,
            "imod_positioning": _pos_table,
            "positioning_hash": positioning_hash,
            "level_angle_x_sign": level_angle_x_sign,
            "imod_to_warp_tilt_angle_sign": imod_tilt_angle_sign,
            "condition_input": ci.to_dict(), "training_dir": str(training_dir)}
        (res_dir / "warp_staging_manifest.json").write_text(json.dumps(staging_manifest, indent=2))
        warp_state["staging_manifest"] = str(res_dir / "warp_staging_manifest.json")
        warp_state["axis_frame"] = axis_frame
        warp_state["target_volume_shape_xyz"] = list(target_shape)
        warp_state["target_pixel_size_A"] = out_pix
        warp_state["conversion_job"] = str(res_dir / "warp_convert.sbatch")
        warnings.append(
            f"warp_convert is cluster_only here (local_warp_convert={local_warp_convert}, "
            f"warpylib state={warp_cap.state}); a dedicated "
            f"conversion job + staging manifest are generated and the MissAlignment job depends on "
            f"it. warp mode={warp_alignment_mode}, axis_frame={axis_frame}, "
            f"xf={'identity' if ci.is_identity else xf_staged}, tilt_axis={tilt_axis}.")
        steps_skipped.append("warp_convert")
        steps_skipped.append("warp_validate")

    # ---- MissAlignment config + run script (REAL, reusing 03_run_missalignment.py) ----
    cfg_yaml = res_dir / "config.yaml"
    run_sh = res_dir / "run_missalignment.sh"
    sel = stacks.get("working_selected")
    sel_path = sel.path if sel else None
    apply_ctf = False  # external IMOD CTF -> MissAlignment apply_ctf is always false

    ma_cfg = config.get("missalignment", {})
    train_dev = str(ma_cfg.get("training_devices", "0"))
    recon_dev = str(ma_cfg.get("reconstruction_devices", "0"))
    dl_per = int(ma_cfg.get("dataloaders_per_trainer", 1))
    executable = str(ma_cfg.get("executable", "miss-alignment"))
    ma_cmd = _r3.missalignment_command(
        config_path=cfg_yaml, training_devices=train_dev, reconstruction_devices=recon_dev,
        dataloaders_per_trainer=dl_per, prepare_stacks=None, start_at_iteration=0, executable=executable)

    def _mk_cfg():
        # REAL MissAlignment config from the same generator 03_run_missalignment.py uses.
        cfg_yaml.write_text(_r3.config_text(training_dir, refinement_mode))
        run_sh.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"# Generated by orchestrate.py (refinement_mode={refinement_mode}, condition={condition}).\n"
            f"# training_directory must hold the Warp XML + tiltstack/*/*.st for this condition.\n"
            f"# selected working stack (CTF-correct input): {sel_path}\n"
            f"{_r3.shell_quote(ma_cmd)}\n")
        run_sh.chmod(0o755)
    do("missalign_config",
       _hash_obj([sel_path, refinement_mode, condition, apply_ctf, str(training_dir)]),
       _hash_obj(["ma_config", refinement_mode, _r3.shell_quote(ma_cmd)]), [cfg_yaml, run_sh], _mk_cfg)

    # ---- internal Slurm disabled ----
    # Canonical prepare emits MissAlignment and export snapshot jobs through
    # pipeline.jobs.generate_jobs. Keep the MissAlignment run script above; do not
    # also generate a per-condition missalign_<mode>.sbatch here.
    sbatch_file = None

    # ---- submission (only with --submit; invokes the REAL sbatch) ----
    submission = {"submitted": False}
    if getattr(args, "submit", False):
        if sbatch_file is None or not sbatch_file.is_file():
            raise RuntimeError("--submit requires --generate-slurm (no .sbatch file to submit)")
        if shutil.which("sbatch") is None:
            warnings.append("--submit requested but 'sbatch' is not on PATH; not on a SLURM host -- "
                            "submission skipped (the .sbatch is ready to submit on the cluster).")
            submission = {"submitted": False, "reason": "sbatch_not_found", "sbatch": str(sbatch_file)}
        else:
            sub_json = res_dir / "submission.json"
            def _mk_submit():
                cp = subprocess.run(["sbatch", str(sbatch_file)], text=True, capture_output=True)
                if cp.returncode != 0:
                    raise RuntimeError(f"sbatch failed (rc={cp.returncode}): {cp.stderr[-300:]}")
                sub_json.write_text(json.dumps(
                    {"sbatch": str(sbatch_file), "stdout": cp.stdout.strip(),
                     "job_id": cp.stdout.strip().split()[-1] if cp.stdout.strip() else None}, indent=2) + "\n")
            do("submission", _hash_file(sbatch_file), _hash_obj(["sbatch", str(sbatch_file)]), [sub_json], _mk_submit)
            submission = {"submitted": True, "sbatch": str(sbatch_file)}

    # ---- manifest ----
    # complete measured geometry: every available stack's Q + the G maps (header-derived).
    geom_manifest = {
        "measured": {role: meas.to_dict() for role, meas in m.items()},
        "Q": measured.get("Q", {}),
        "maps": measured.get("maps", {}),
    }
    if B > 1 and src_raw_m and src_ali_m:
        from multiresolution import integer_binned_grid as _ibg
        sr_g = src_raw_m.grid; sa_g = src_ali_m.grid
        wr_g = _ibg(sr_g, B); wa_g = _ibg(sa_g, B)
        geom_manifest["maps"]["G_r"] = wr_g.mapping_to(sr_g).tolist()
        geom_manifest["maps"]["G_a"] = wa_g.mapping_to(sa_g).tolist()
        geom_manifest["Q"]["working_raw_derived"] = wr_g.Q.tolist()
        geom_manifest["Q"]["working_aligned_derived"] = wa_g.Q.tolist()
    manifest = {
        "schema_version": 2, "basename": basename, "condition": condition,
        "extra_binning": B, "ctf_mode": ctf_mode, "refinement_mode": refinement_mode,
        "source_dims_xy": list(src_dims), "source_aligned_dims_xy": list(src_ali_dims),
        "working_dims_xy": list(work_dims),
        "source_pixel_A": p_src_A, "source_aligned_pixel_A": p_src_ali_A, "working_pixel_A": p_work_A,
        "geometry": geom_manifest,
        "projection_binning": binning_manifest,
        "ctf_params": ctf_params, "ctf_param_source": ctf_param_source,
        "warp": warp_state,
        "final_ctf_deferred_to_export": ctf_mode in ("final", "both"),
        "stacks": {k: v.to_dict() for k, v in stacks.items()},
        "ctf_inputs": {k: inputs.get(k) for k in
                       ("ctf_com", "defocus_file", "tilt_file", "source_aligned", "source_xf")},
        "missalignment": {"refinement_mode": refinement_mode, "apply_ctf": apply_ctf,
                          "config_yaml": str(cfg_yaml), "run_script": str(run_sh),
                          "command": _r3.shell_quote(ma_cmd), "training_directory": str(training_dir),
                          "sbatch": str(sbatch_file) if sbatch_file else None,
                          "submission": submission,
                          "note": "config.yaml/run script/sbatch are REAL; the GPU RUN (MissAlignment/"
                                  "warpylib) executes on the cluster -- not installed locally."},
        "steps_run": steps_run, "steps_skipped": steps_skipped, "warnings": warnings,
    }
    man_path = ws / "binning_ctf_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return OrchestrationResult(ws, stacks, steps_run, steps_skipped, man_path, warnings)
