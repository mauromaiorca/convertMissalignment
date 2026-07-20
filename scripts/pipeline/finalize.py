#!/usr/bin/env python3
"""Phase-3 (FINALIZE) handlers: finalize / verify-final / collect-debug.

``finalize --result auto`` consumes the CANONICAL constrained result
(``constrained_alignment.json`` via ``result_contract.read_constrained_result``);
it NEVER picks the latest XML by mtime (§29). For translation/rigid/similarity the
residual ``.xf`` is exported DIRECTLY from the parameters (no post-hoc affine fit,
§30). Final CTF and final reconstruction are SOURCE-resolution and run only here
(or via the generated CPU jobs), never during import.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import runlog as RL
from .runlayout import RunLayout, dataset_id_from_config


def _run_id(layout: RunLayout) -> str:
    return "r" + hashlib.sha256(str(layout.run_dir).encode()).hexdigest()[:12]


def _layout(cfg, args) -> RunLayout:
    basename = (getattr(args, "basename", None) or cfg.get("project", {}).get("basename")
                or cfg.get("project", {}).get("name") or "series")
    conds = (getattr(args, "condition", None)
             or cfg.get("conversion", {}).get("initial_conditions", ["ali_identity"]))
    condition = conds[0] if isinstance(conds, list) else conds
    mode = (getattr(args, "refinement_mode", None)
            or cfg.get("missalignment", {}).get("refinement_mode", "standard"))
    out_dir = Path(getattr(args, "out_dir", None) or cfg.get("paths", {}).get("output_dir") or ".")
    return RunLayout.from_settings(
        out_dir=out_dir, basename=basename, condition=condition, refinement_mode=mode,
        dataset_id=getattr(args, "dataset", None) or dataset_id_from_config(cfg),
    )


def _find_result_dir(layout: RunLayout, result_arg: str) -> Path:
    """--result auto: use the canonical result location, NOT latest-mtime XML."""
    if result_arg and result_arg != "auto":
        return Path(result_arg)
    # canonical locations searched in order (all deterministic).
    for cand in (layout.results_dir / layout.refinement_mode,
                 layout.results_dir, layout.run_dir / "missalignment" / "results"):
        if (cand / "constrained_alignment.json").is_file():
            return cand
    return layout.results_dir / layout.refinement_mode


def _constrained_source_xf(model_name: str, params, angles, grids: dict, h0_source_rows=None):
    """Export from constrained parameters (§30), COMPOSED with the initial alignment.

    Defect 2.17: the final raw->aligned transform is NOT the bare residual. With
    DeltaH_working the per-tilt residual and H0_source the initial raw->aligned
    transform (from the source .xf), the correct composition is:

        H0_working    = inv(G_a) @ H0_source @ G_r
        Hfinal_working = DeltaH_working @ H0_working
        Hfinal_source  = G_a @ Hfinal_working @ inv(G_r)

    When ``h0_source_rows`` is None (no source .xf, e.g. ali_identity with identity
    H0), H0_source is the identity raw->aligned map ``G_a @ inv(G_r)`` so that
    H0_working = I and the result reduces to the residual route — but for a real
    raw_xf condition the initial alignment is folded in, as required.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from alignment_models.registry import get_model
    from imod_affine import homogeneous_to_xf, xf_to_homogeneous
    import torch

    model = get_model(model_name)
    p = torch.as_tensor(params, dtype=torch.float64)
    n = p.shape[0]
    work_xy = grids["working_aligned_shape_xy"]
    cx = (work_xy[0] - 1) / 2.0
    cy = (work_xy[1] - 1) / 2.0
    centers = torch.tensor([cx, cy], dtype=torch.float64).expand(n, 2)
    DeltaH_working = model.homogeneous_physical(p, centers).detach().cpu().numpy()  # (n,3,3)

    G_r = np.asarray(grids["G_r"], float)
    G_a = np.asarray(grids["G_a"], float)
    Gr_inv = np.linalg.inv(G_r)
    Ga_inv = np.linalg.inv(G_a)
    raw_xy = grids["source_raw_shape_xy"]
    ali_xy = grids["source_aligned_shape_xy"]

    working_residual, source_residual, source_final = [], [], []
    for i in range(n):
        DHw = DeltaH_working[i]
        # working residual .xf (aligned-in, aligned-out) -- still emitted for QC
        a_w, d_w = homogeneous_to_xf(DHw, work_xy, work_xy)
        working_residual.append((a_w, d_w))
        # source residual: G_a @ DeltaH_working @ inv(G_a)
        DHs = G_a @ DHw @ Ga_inv
        a_s, d_s = homogeneous_to_xf(DHs, ali_xy, ali_xy)
        source_residual.append((a_s, d_s))
        # H0_source (raw->aligned). From the source .xf if given, else identity map.
        if h0_source_rows is not None:
            A0, d0 = h0_source_rows[i]
            H0_source = xf_to_homogeneous(A0, d0, raw_xy, ali_xy)
        else:
            H0_source = G_a @ Gr_inv          # identity raw->aligned (H0_working = I)
        H0_working = Ga_inv @ H0_source @ G_r
        Hfinal_working = DHw @ H0_working      # DeltaH composed with the initial alignment
        Hfinal_source = G_a @ Hfinal_working @ Gr_inv
        a_f, d_f = homogeneous_to_xf(Hfinal_source, raw_xy, ali_xy)
        source_final.append((a_f, d_f))
    return working_residual, source_residual, source_final


def _load_h0_source(man: dict, n_tilts: int):
    """Read the initial raw->aligned source .xf rows for H0 composition (2.17).

    Returns a list of (A, d) per tilt, or None when there is no source .xf (the
    condition's initial alignment is identity, e.g. ali_identity). Never globs:
    the path comes from the resolved/orchestrate manifest's recorded inputs.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from imod_affine import read_xf
    src_xf = (man.get("ctf_inputs", {}) or {}).get("source_xf") or man.get("source_xf")
    if not src_xf or not Path(src_xf).is_file():
        return None
    A, d = read_xf(src_xf)
    if len(A) != n_tilts:
        raise ValueError(f"source .xf rows {len(A)} != result tilts {n_tilts}")
    return [(A[i], d[i]) for i in range(n_tilts)]


def _write_xf(path: Path, rows) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from imod_affine import write_xf
    A = np.stack([r[0] for r in rows]); d = np.stack([r[1] for r in rows])
    write_xf(path, A, d)


def _result_backend(cfg, args) -> str:
    if getattr(args, "result_backend", None):
        return args.result_backend
    return (cfg.get("missalignment", {}) or {}).get("result_backend", "warp_xml")


def _synthesize_missalign_params(cfg, condition: str, dest: Path) -> Path:
    """Write the params JSON export_condition_results.py consumes, derived ONLY from the
    resolved config (consume-only, §5). The exporter reads files/geometry/series_name and
    requires the condition key to exist."""
    from . import project_config as PC
    rc = PC.from_dict(cfg)
    g = rc.geometry
    params = {
        "series_name": rc.basename,
        "files": {k: v for k, v in {
            "raw_stack": rc.sources.raw_stack, "aligned_stack": rc.sources.aligned_stack,
            "final_xf": rc.sources.final_xf_file}.items() if v},
        "geometry": {k: v for k, v in {
            "raw_pixel_size_A": g.raw_pixel_size_A, "aligned_pixel_size_A": g.aligned_pixel_size_A,
            "target_volume_shape_xyz": g.target_volume_shape_xyz,
            "target_output_pixel_size_A": g.target_pixel_size_A}.items() if v is not None},
        "conditions": {condition: {"alignment_mode": rc.warp_mode(condition),
                                   "axis_frame": PC.axis_frame_for(condition)}},
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(params, indent=2) + "\n")
    return dest


def _build_export_command(*, exporter, params, warp_dir, condition, out_dir, xml,
                          rms_tol, max_tol) -> list:
    """The CORRECT export_condition_results.py CLI (§14). The exporter requires --params and
    --warp-dir and takes --condition/--out-dir/--xml/--rms-tolerance-px/--max-tolerance-px;
    it has NO positional settings argument (the historical call passed one and always failed)."""
    import sys
    return [sys.executable, str(exporter),
            "--params", str(params), "--warp-dir", str(warp_dir),
            "--condition", str(condition), "--out-dir", str(out_dir),
            "--xml", str(xml), "--rms-tolerance-px", str(rms_tol),
            "--max-tolerance-px", str(max_tol)]


def _finalize_warp_xml(cfg, args, layout, rl) -> int:
    """Stock MissAlignment writes updated Warp XML, not the constrained contract (2.16).

    Use an EXPLICIT final XML path (never latest-mtime) and route through
    export_condition_results.py -> warp_to_imod_affine.py with the correct CLI (§14).
    """
    import shutil
    import subprocess
    rm = layout.manifest("result_manifest.json")
    rmd = json.loads(rm.read_text()) if rm.is_file() else {}
    final_xml = getattr(args, "xml", None) or rmd.get("final_xml")
    if not final_xml:
        print("ERROR: result_backend=warp_xml requires an EXPLICIT final XML. Pass --xml <path> "
              "or record it in manifests/result_manifest.json (final_xml). Refusing to pick the "
              "latest XML by mtime (defect 2.16).")
        return 2
    if not Path(final_xml).is_file():
        print(f"ERROR: final XML not found: {final_xml}")
        return 2
    # condition + warp training dir come from the deterministic result manifest (§13),
    # falling back to the canonical layout (§6: one training dir everywhere).
    condition = rmd.get("condition") or layout.condition
    warp_dir = Path(rmd.get("training_directory") or layout.training_dir)
    ma = cfg.get("missalignment", {}) or {}
    rms_tol = float(ma.get("rms_tolerance_px", 0.10))
    max_tol = float(ma.get("max_tolerance_px", 0.25))
    params = _synthesize_missalign_params(cfg, condition, layout.manifest("missalign_params.json"))
    exporter = Path(__file__).resolve().parents[1] / "export_condition_results.py"
    cmd = _build_export_command(exporter=exporter, params=params, warp_dir=warp_dir,
                                condition=condition, out_dir=layout.final_transforms,
                                xml=final_xml, rms_tol=rms_tol, max_tol=max_tol)
    rl.log_event(step="F-warpxml", event="export_condition_results", status="info",
                 data={"final_xml": str(final_xml), "warp_dir": str(warp_dir),
                       "condition": condition, "cmd": cmd})
    layout.final_transforms.mkdir(parents=True, exist_ok=True)
    cp = subprocess.run(cmd, text=True, capture_output=True)
    if cp.returncode != 0:
        print(f"ERROR: warp_xml export failed (rc={cp.returncode}): "
              f"{(cp.stdout[-400:] + cp.stderr[-600:])}")
        return 1
    _atomic(layout.manifest("finalize_manifest.json"),
            {"run_id": _run_id(layout), "result_backend": "warp_xml", "final_xml": str(final_xml),
             "condition": condition, "warp_dir": str(warp_dir), "params": str(params),
             "transforms_dir": str(layout.final_transforms), "command": cmd,
             "note": "exported via export_condition_results.py -> warp_to_imod_affine.py with an "
                     "EXPLICIT XML (no mtime selection) and the correct --params/--warp-dir CLI."})
    print(f"[finalize] warp_xml backend: exported condition={condition} from explicit XML {final_xml}")
    return 0


def cmd_finalize(cfg, args) -> int:
    layout = _layout(cfg, args)
    if not layout.run_dir.exists():
        print(f"ERROR: no run dir at {layout.run_dir}; run project preparation first.")
        return 2
    rl = RL.RunLogger(layout.run_dir, run_id=_run_id(layout), phase="finalize")
    rl.write_environment(name="finalize")
    backend = _result_backend(cfg, args)
    rl.log_event(step="F00", event="result_backend", status="info", message=backend)
    if backend == "warp_xml":
        try:
            return _finalize_warp_xml(cfg, args, layout, rl)
        except Exception as exc:
            rl.write_postmortem(exc, step="F-warpxml")
            print(f"ERROR: finalize (warp_xml) failed: {exc}")
            return 1
    # constrained_json backend (the constrained fork's canonical result contract)
    try:
        from alignment_models import result_contract as RC
        result_dir = _find_result_dir(layout, getattr(args, "result", "auto"))
        rl.log_event(step="F01", event="locate_result", status="info", message=str(result_dir))
        ref = RC.read_constrained_result(result_dir, expected_model=layout.refinement_mode,
                                         require_completed=True)
        rl.log_event(step="F02", event="read_result", status="ok",
                     data={"model": ref.json["model"], "n_tilts": ref.json["n_tilts"]})

        # grids from the prepare manifest (measured headers)
        prep = json.loads(layout.manifest("prepare_manifest.json").read_text())
        man = json.loads(Path(prep["orchestrate_manifest"]).read_text())
        geom = man["geometry"]["maps"]
        grids = {
            "G_r": geom["G_r"], "G_a": geom["G_a"],
            "source_raw_shape_xy": man["source_dims_xy"],
            "source_aligned_shape_xy": man["source_aligned_dims_xy"],
            "working_aligned_shape_xy": man["working_dims_xy"],
        }
        # H0_source: the initial raw->aligned transform for this condition (2.17).
        # Load the source .xf so DeltaH is composed with the initial alignment, not
        # exported as if it were the complete transform.
        h0_rows = _load_h0_source(man, ref.json["n_tilts"])
        rl.log_event(step="F02b", event="h0_source", status="ok",
                     data={"composed_with_initial_alignment": h0_rows is not None})
        wr, sr, sf = _constrained_source_xf(ref.json["model"], ref.json["free_parameters"],
                                            ref.json["tilt_angles"], grids, h0_source_rows=h0_rows)
        layout.final_transforms.mkdir(parents=True, exist_ok=True)
        _write_xf(layout.final_transforms / "working_residual.xf", wr)
        _write_xf(layout.final_transforms / "source_residual.xf", sr)
        _write_xf(layout.final_transforms / "final_source_raw_to_aligned.xf", sf)
        # working_raw_to_final = same composed transform (raw->final aligned)
        _write_xf(layout.final_transforms / "working_raw_to_final.xf", sf)
        rl.log_event(step="F03", event="export_constrained_xf", status="ok",
                     data={"n_tilts": len(sf), "method": "direct-from-parameters (no fit)"})

        # finalize manifest
        fin = {
            "run_id": _run_id(layout), "model": ref.json["model"], "n_tilts": ref.json["n_tilts"],
            "result_dir": str(result_dir), "result_schema": ref.json["schema_version"],
            "transforms": {
                "working_residual": str(layout.final_transforms / "working_residual.xf"),
                "source_residual": str(layout.final_transforms / "source_residual.xf"),
                "final_source_raw_to_aligned": str(layout.final_transforms / "final_source_raw_to_aligned.xf"),
            },
            "ctf_mode": man.get("ctf_mode"),
            "final_ctf_required": man.get("ctf_mode") in ("final", "both"),
            "next": [],
            "note": "final transforms exported; inspect outputs before any downstream reconstruction.",
        }
        _atomic(layout.manifest("finalize_manifest.json"), fin)
        rl.log_event(step="F04", event="finalize_done", status="ok")
        print(f"[finalize] exported source .xf from the canonical {ref.json['model']} result "
              f"({ref.json['n_tilts']} tilts) -> {layout.final_transforms}")
        print(f"[finalize] manifest: {layout.manifest('finalize_manifest.json')}")
        return 0
    except Exception as exc:
        rl.write_postmortem(exc, step=rl._last_step)
        print(f"ERROR: finalize failed: {exc}")
        print(f"       postmortem: {layout.run_dir}/diagnostics/postmortem/failure.json")
        return 1


def cmd_verify_final(cfg, args) -> int:
    layout = _layout(cfg, args)
    ft = layout.final_transforms
    checks = {}
    expected = ["working_residual.xf", "source_residual.xf", "final_source_raw_to_aligned.xf"]
    for name in expected:
        p = ft / name
        checks[name] = {"exists": p.is_file(),
                        "rows": len(p.read_text().splitlines()) if p.is_file() else 0}
    # final stack / reconstruction (produced by jobs) — report if present
    for sub, label in ((layout.final_aligned, "final_aligned"),
                       (layout.final_reconstruction, "final_reconstruction")):
        files = list(sub.glob("*.mrc")) + list(sub.glob("*.rec")) if sub.exists() else []
        checks[label] = {"present": bool(files), "files": [f.name for f in files]}
    ok = all(checks[n]["exists"] for n in expected)
    out = {"ok": ok, "checks": checks}
    layout.manifest("final_validation.json").parent.mkdir(parents=True, exist_ok=True)
    _atomic(layout.manifest("final_validation.json"), out)
    print(f"[verify-final] transforms present: {ok}")
    for n in expected:
        print(f"    {n:34s} {'OK' if checks[n]['exists'] else 'MISSING'} ({checks[n]['rows']} rows)")
    return 0 if ok else 1


def cmd_collect_debug(cfg, args) -> int:
    layout = _layout(cfg, args)
    if not layout.run_dir.exists():
        print(f"ERROR: no run dir at {layout.run_dir}")
        return 2
    bundle = RL.collect_debug_bundle(layout.run_dir, _run_id(layout),
                                     include_checkpoints=getattr(args, "include_checkpoints", False))
    print(f"[collect-debug] wrote {bundle}")
    return 0


def _atomic(path: Path, obj) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)
