#!/usr/bin/env python3
"""Export a refined MissAlignment result back to IMOD ``.xf`` (exact).

Reverse direction of the interoperability layer. Uses the constrained
refinement model's closed-form per-tilt matrix, so the exported ``.xf`` is
*exact* (not a fitted movement grid). For raw conditions it writes
``raw -> final``; for ``ali_identity`` it writes both the ``ali -> final``
residual and the composed ``raw -> final`` (``Hfinal = DeltaH @ H0``).

The grid-fit exporter ``scripts/warp_to_imod_affine.py`` is retained separately
as an independent verification oracle.

Inputs
------
- A project settings TOML (``[geometry]``, ``[input]``, ``[refinement]``,
  ``[conversion]``).
- A residual-parameter JSON produced by the refinement step
  (``alignment_models.serialization.write_params_json``): ``{model, params,...}``.
  If omitted, the residual defaults to identity (export = initial alignment).

See ``docs/interoperability/WARP_TO_IMOD.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from runtime_env import ensure_scientific_python  # noqa: E402


def _settings_hint(argv: list[str]) -> Path | None:
    if not argv:
        return None
    index = 1 if argv[0] in ("finalize", "verify-final", "collect-debug") else 0
    if index >= len(argv) or argv[index].startswith("-"):
        return None
    return Path(argv[index])


ensure_scientific_python(
    script=Path(__file__), argv=sys.argv[1:], settings=_settings_hint(sys.argv[1:]),
    required=("numpy", "mrcfile"), label="export_warp_to_imod.py")

import numpy as np
import alignment_models as am  # noqa: E402
from alignment_models import composition as comp  # noqa: E402
from alignment_models import interop  # noqa: E402
from alignment_models.refinement_config import from_toml_dict  # noqa: E402
from alignment_models.serialization import (  # noqa: E402
    homogeneous_to_xf_rows,
    params_from_dict,
)
from imod_affine import forward_points_pixels, read_xf, write_xf, xf_to_homogeneous  # noqa: E402


def load_toml(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as fh:
        return tomllib.load(fh)


def _xy(dims, fallback):
    if dims and len(dims) >= 2:
        return (int(dims[0]), int(dims[1]))
    return fallback


def resolve_geometry(cfg: dict[str, Any], n_tilts_hint: int | None):
    g = cfg.get("geometry", {})
    raw = _xy(g.get("raw_shape_xyz") or g.get("raw_dimensions_xyz"), None)
    ali = _xy(g.get("aligned_shape_xyz") or g.get("aligned_dimensions_xyz"), raw)
    tgt = _xy(g.get("target_volume_shape_xyz") or g.get("target_volume_xyz"), ali)
    if raw is None:
        raise SystemExit("ERROR: [geometry].raw_dimensions_xyz is required for export")
    if ali is None:
        ali = raw
    final = tgt or ali
    p_raw = float(g.get("raw_pixel_size_A") or 0) or None
    p_ali = float(g.get("aligned_pixel_size_A") or 0) or (p_raw)
    p_final = float(g.get("target_pixel_size_A") or 0) or (p_ali or p_raw)
    if not p_raw:
        raise SystemExit("ERROR: [geometry].raw_pixel_size_A is required for export")
    p_ali = p_ali or p_raw
    p_final = p_final or p_ali
    return {
        "raw_shape": raw, "ali_shape": ali, "final_shape": final,
        "p_raw": p_raw, "p_ali": p_ali, "p_final": p_final,
    }


def load_residual(path: Path | None, n_tilts: int, model_name: str):
    if path is None:
        model = am.get_model(model_name)
        return model, model.identity_params(n_tilts), "identity (no residual supplied)"
    data = json.loads(Path(path).read_text())
    model, params = params_from_dict(data)
    if model.name != model_name:
        print(f"WARNING: residual model {model.name!r} differs from [refinement].model {model_name!r}; using residual file's model.")
    return model, params, str(path)


def equivalence_check(H0, dH, geom, raw_pts=None):
    """Self-check: composed raw->final == residual(original(raw)). Returns max px err."""
    n = H0.shape[0]
    raw, ali, final = geom["raw_shape"], geom["ali_shape"], geom["final_shape"]
    pr, pa, pf = geom["p_raw"], geom["p_ali"], geom["p_final"]
    Hfinal = comp.compose_final_per_tilt(H0, dH)
    cA, cd = homogeneous_to_xf_rows(Hfinal, raw, final, pr, pf)
    rA, rd = homogeneous_to_xf_rows(dH, ali, final, pa, pf)
    oA, od = homogeneous_to_xf_rows(H0, raw, ali, pr, pa)
    if raw_pts is None:
        raw_pts = np.array([[0, 0], [raw[0], 0], [0, raw[1]], [raw[0], raw[1]],
                            [raw[0] / 2, raw[1] / 2]], float)
    max_err = 0.0
    for i in range(n):
        a = forward_points_pixels(raw_pts, cA[i], cd[i], raw, final)
        ali_pts = forward_points_pixels(raw_pts, oA[i], od[i], raw, ali)
        b = forward_points_pixels(ali_pts, rA[i], rd[i], ali, final)
        max_err = max(max_err, float(np.max(np.abs(a - b))))
    return max_err, (cA, cd), (rA, rd)


def _restore_affine2d_warp(cfg, args) -> int:
    """Restore a real affine2d Warp result (working raw->final .xf) to the source grid.

    Raw and aligned source grids are kept SEPARATE: the aligned grid is measured
    from the real aligned-stack header (authoritative) or taken from an explicit
    ``[geometry].aligned_dimensions_xyz`` -- it is NEVER assumed equal to the raw
    grid (defect #10).
    """
    from multiresolution import Grid2D, build_plan
    from multiresolution import transfer as _T
    from imod_affine import homogeneous_to_xf as _h2x
    g = cfg.get("geometry", {})
    src_dims = _xy(g.get("raw_dimensions_xyz"), None)
    if src_dims is None:
        raise SystemExit("ERROR: affine2d-warp restore requires [geometry].raw_dimensions_xyz")
    p_src = float(g.get("raw_pixel_size_A") or 0) or 1.0
    B = int(cfg.get("multiresolution", {}).get("extra_projection_binning") or 1)
    if B <= 1:
        raise SystemExit("ERROR: affine2d-warp restore requires [multiresolution].extra_projection_binning > 1")
    # --- SEPARATE aligned grid (measured header > config > fail; never == raw) ---
    ali_stack = (cfg.get("input", {}) or {}).get("aligned_stack")
    ali_dims = _xy(g.get("aligned_dimensions_xyz"), None)
    p_ali = float(g.get("aligned_pixel_size_A") or 0) or None
    if ali_stack and Path(ali_stack).is_file():
        try:
            from pipeline.geometry import measure_mrc_grid
            am = measure_mrc_grid(Path(ali_stack), role="source_aligned")
            ali_dims = am.shape_xy
            p_ali = am.pixel_size_xy_A[0]
        except Exception as exc:  # measurement failed: fall through to config/fail
            print(f"WARNING: could not measure aligned stack header ({exc}); using config geometry")
    if ali_dims is None:
        raise SystemExit(
            "ERROR: affine2d-warp restore needs the aligned grid. Provide [input].aligned_stack "
            "(measured) or [geometry].aligned_dimensions_xyz. The aligned grid is NEVER assumed "
            "equal to the raw grid.")
    if p_ali is None:
        p_ali = p_src
    sr = Grid2D.axis_aligned("source_raw", src_dims, p_src)
    sa = Grid2D.axis_aligned("source_aligned", ali_dims, p_ali)
    plan = build_plan(B, sr, sa)  # validates factor + divisibility (rejects unsupported)
    Aw, dw = read_xf(args.working_xf)
    series = cfg.get("project", {}).get("basename") or cfg.get("project", {}).get("name") or "series"
    out_dir = (args.out_dir or interop.export_dir(Path(cfg.get("paths", {}).get("output_dir") or "."),
                                                  "ali_identity", "affine2d")).resolve()
    A_src, d_src = [], []
    for i in range(len(Aw)):
        Hfw = xf_to_homogeneous(Aw[i], dw[i], plan.working_raw.shape_xy, plan.working_aligned.shape_xy)
        Hfs = _T.restore_hfinal_working_to_source(Hfw, plan.G_a, plan.G_r)
        a, d = _h2x(Hfs, sr.shape_xy, sa.shape_xy)
        A_src.append(a); d_src.append(d)
    A_src, d_src = np.asarray(A_src), np.asarray(d_src)
    final_xf = out_dir / f"{series}_ali_identity_affine2d_bin{B}_final_source_raw_to_aligned.xf"
    report = {"mode": "affine2d_warp_restore", "factor": B, "n_tilts": len(Aw),
              "source_raw_dims": list(src_dims), "source_aligned_dims": list(ali_dims),
              "raw_pixel_A": p_src, "aligned_pixel_A": p_ali,
              "G_a": plan.G_a.tolist(), "G_r": plan.G_r.tolist(),
              "output_source_xf": str(final_xf), "software_versions": interop.software_versions()}
    print(f"affine2d-warp restore (bin{B}) -> source raw {src_dims} / aligned {ali_dims}, {len(Aw)} tilts")
    if args.dry_run:
        print(json.dumps(report, indent=2, default=str)); return 0
    if args.validate_only:
        print("validate-only: affine2d-warp restore configuration OK"); return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    write_xf(final_xf, A_src, d_src)
    (out_dir / "affine2d_restore_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"Wrote source-resolution affine2d .xf: {final_xf}")
    return 0


FINALIZE_SUBCOMMANDS = ("finalize", "verify-final", "collect-debug")


def _dispatch_finalize(verb: str, argv: list[str]) -> int:
    from pipeline import finalize as FIN
    ap = argparse.ArgumentParser(prog=f"export_warp_to_imod.py {verb}")
    ap.add_argument("settings", type=Path, help="Project settings TOML")
    ap.add_argument("--result", default="auto",
                    help="Canonical result dir, or 'auto' (default) to use the run's result contract.")
    ap.add_argument("--result-backend", dest="result_backend", default=None,
                    choices=("warp_xml", "constrained_json"),
                    help="Override the result adapter (default from [missalignment].result_backend).")
    ap.add_argument("--xml", default=None, help="(warp_xml) explicit final Warp XML path (never mtime).")
    ap.add_argument("--condition", default=None)
    ap.add_argument("--refinement-mode", default=None)
    ap.add_argument("--basename", default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--include-checkpoints", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_toml(args.settings)
    if verb == "finalize":
        return FIN.cmd_finalize(cfg, args)
    if verb == "verify-final":
        return FIN.cmd_verify_final(cfg, args)
    if verb == "collect-debug":
        return FIN.cmd_collect_debug(cfg, args)
    return 2


def main() -> int:
    # Phase-3 subcommand interface; legacy flat flags still supported below.
    if len(sys.argv) > 1 and sys.argv[1] in FINALIZE_SUBCOMMANDS:
        return _dispatch_finalize(sys.argv[1], sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "revise":
        # Publish the revised IMOD alignment under exported_data/imod/<condition_id>.
        # scripts/ is already on sys.path (module import above), so pipeline.* imports.
        from pipeline.imod_revision_export import main as revise_main
        return revise_main(sys.argv[2:])
    return _legacy_export_main()


def _legacy_export_main() -> int:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "") + (
            "\n\nCANONICAL PHASE-3 SUBCOMMANDS:\n"
            "  export_warp_to_imod.py finalize SETTINGS.toml --result auto   consume the result contract; export source .xf\n"
            "  export_warp_to_imod.py verify-final SETTINGS.toml             validate final outputs\n"
            "  export_warp_to_imod.py collect-debug SETTINGS.toml           build a debug bundle\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("settings", type=Path, help="Project settings TOML")
    ap.add_argument("--result", type=Path, default=None, help="Result dir or residual-params JSON")
    ap.add_argument("--residual-params", type=Path, default=None, help="Residual-params JSON (overrides --result)")
    ap.add_argument("--condition", default=None, help="Override initial condition")
    ap.add_argument("--refinement-model", default=None, help="Override refinement model")
    ap.add_argument("--out-dir", type=Path, default=None, help="Override export output dir")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-reconstruction-files", action="store_true")
    ap.add_argument("--restore-to", choices=("source", "working"), default=None,
                    help="Multiresolution restore target. 'source' restores a working-grid "
                         "residual to the source grid and writes a source-resolution .xf.")
    ap.add_argument("--result-type", choices=("affine2d-warp", "constrained"), default=None,
                    help="affine2d-warp: restore a real MissAlignment affine2d working raw->final .xf.")
    ap.add_argument("--working-xf", type=Path, default=None,
                    help="Working-grid raw->final affine .xf (from a real affine2d Warp result).")
    args = ap.parse_args()

    cfg = load_toml(args.settings)
    refinement = from_toml_dict(cfg.get("refinement", {}),
                               {"model": args.refinement_model} if args.refinement_model else {})
    for w in refinement.warnings:
        print(f"WARNING: {w}")
    model_name = refinement.model

    # --- affine2d Warp result restore (handled first; needs no source H0) -----
    if args.result_type == "affine2d-warp" and args.working_xf:
        return _restore_affine2d_warp(cfg, args)

    conditions = cfg.get("conversion", {}).get("initial_conditions", ["raw_xf_affine_fixed"])
    if args.condition:
        conditions = [args.condition]
    elif len(conditions) > 1:
        raise SystemExit(
            f"ERROR: multiple conditions {conditions}; pass --condition to disambiguate the export."
        )
    condition = conditions[0]
    if condition not in comp.ALL_CONDITIONS:
        raise SystemExit(f"ERROR: unknown condition {condition!r}")

    # Original raw->ali xf (for translation/affine/ali). raw_identity uses identity.
    xf_file = cfg.get("input", {}).get("final_xf_file") or ""
    if condition == "raw_identity":
        A0 = d0 = None
    else:
        if not xf_file or not Path(xf_file).is_file():
            raise SystemExit(f"ERROR: [input].final_xf_file required for condition {condition!r}: {xf_file!r}")
        A0, d0 = read_xf(xf_file)
    n_tilts = len(A0) if A0 is not None else int(cfg.get("geometry", {}).get("tilt_count") or 0)
    geom = resolve_geometry(cfg, n_tilts)

    residual_path = args.residual_params or args.result
    if residual_path and Path(residual_path).is_dir():
        cand = sorted(Path(residual_path).glob("*residual*params*.json")) or sorted(Path(residual_path).glob("*.params.json"))
        residual_path = cand[0] if cand else None
    model, params, residual_src = load_residual(residual_path, n_tilts or 1, model_name)
    n_tilts = model.as_tensor(params).shape[0]

    series = cfg.get("project", {}).get("basename") or cfg.get("project", {}).get("name") or "series"
    out_dir = args.out_dir or interop.export_dir(
        Path(cfg.get("paths", {}).get("output_dir") or "."), condition, model_name)

    mr = cfg.get("multiresolution", {})
    if args.restore_to == "source" and mr.get("enabled"):
        from multiresolution import Grid2D, build_plan
        from multiresolution.restore import restore_residual_to_source
        B = int(mr.get("extra_projection_binning"))
        src_dims = _xy(cfg.get("geometry", {}).get("raw_dimensions_xyz"), geom["raw_shape"])
        p_src = geom["p_raw"]
        sr = Grid2D.axis_aligned("source_raw", src_dims, p_src, role="source_raw")
        sa = Grid2D.axis_aligned("source_aligned", src_dims, p_src, role="source_aligned")
        plan = build_plan(B, sr, sa)  # validates factor + divisibility (rejects unsupported)
        src_h0 = cfg.get("input", {}).get("final_xf_file") or None
        info, A_final, d_final = restore_residual_to_source(
            model=model, params=params, source_raw=sr, source_ali=sa,
            working_raw=plan.working_raw, working_ali=plan.working_aligned,
            source_h0_xf=Path(src_h0) if src_h0 else None)
        final_xf = out_dir / f"{series}_{condition}_{model_name}_bin{B}_final_source_raw_to_aligned.xf"
        report = {
            "mode": "multiresolution_restore", "restore_to": "source",
            "factor": B, "condition": condition, "refinement_model": model_name,
            "n_tilts": n_tilts, "source_dims": list(src_dims),
            "G_a": plan.G_a.tolist(), "G_r": plan.G_r.tolist(),
            "output_source_xf": str(final_xf),
            "software_versions": interop.software_versions(),
        }
        print(f"multiresolution restore (bin{B}) -> source grid {src_dims}")
        if args.dry_run:
            print(json.dumps(report, indent=2, default=str)); return 0
        if args.validate_only:
            print("validate-only: multiresolution restore configuration OK"); return 0
        out_dir.mkdir(parents=True, exist_ok=True)
        write_xf(final_xf, A_final, d_final)
        (out_dir / "multiresolution_restore_report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n")
        print(f"Wrote source-resolution .xf: {final_xf}")
        return 0

    # Build transforms.
    if condition == "raw_identity":
        H0 = np.stack([np.eye(3) for _ in range(n_tilts)])
    else:
        H0 = comp.initial_homogeneous_per_tilt(condition, A0, d0, geom["raw_shape"],
                                               geom["ali_shape"], geom["p_raw"], geom["p_ali"])
    dH = comp.residual_homogeneous_per_tilt(model, params, geom["ali_shape"], geom["p_ali"])
    max_err, (cA, cd), (rA, rd) = equivalence_check(H0, dH, geom)

    print(f"condition={condition}  model={model_name}  tilts={n_tilts}")
    print(f"residual source: {residual_src}")
    print(f"raw/ali composition self-check max error: {max_err:.3e} px")

    plan = {
        "condition": condition,
        "refinement_model": model_name,
        "n_tilts": n_tilts,
        "geometry": geom,
        "residual_source": residual_src,
        "raw_ali_equivalence_max_px": max_err,
        "out_dir": str(out_dir),
        "outputs": {},
    }
    is_raw = condition in comp.RAW_CONDITIONS
    if is_raw:
        plan["outputs"]["raw_to_final_xf"] = str(out_dir / f"{series}_{condition}_{model_name}_raw_to_final.xf")
    else:
        plan["outputs"]["ali_residual_xf"] = str(out_dir / f"{series}_{condition}_{model_name}_ali_residual.xf")
        plan["outputs"]["raw_to_final_xf"] = str(out_dir / f"{series}_{condition}_{model_name}_raw_to_final.xf")

    if args.dry_run:
        print(json.dumps(plan, indent=2, default=str))
        return 0

    tol = float(cfg.get("validation", {}).get("coordinate_max_tolerance_px", 0.01))
    status = "PASS" if max_err <= tol else "FAIL"
    if args.validate_only:
        print(f"validate-only: status={status} (tol={tol} px)")
        return 0 if status == "PASS" else 1

    out_dir.mkdir(parents=True, exist_ok=True)
    if is_raw:
        write_xf(out_dir / f"{series}_{condition}_{model_name}_raw_to_final.xf", cA, cd)
    else:
        write_xf(out_dir / f"{series}_{condition}_{model_name}_ali_residual.xf", rA, rd)
        write_xf(out_dir / f"{series}_{condition}_{model_name}_raw_to_final.xf", cA, cd)

    report = {**plan, "status": status, "tolerance_px": tol,
              "software_versions": interop.software_versions()}
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

    if args.write_reconstruction_files and condition != "raw_identity":
        recon = out_dir / "reconstruction_inputs"
        recon.mkdir(parents=True, exist_ok=True)
        raw_stack = cfg.get("input", {}).get("raw_stack") or "RAW_STACK.st"
        tlt = cfg.get("input", {}).get("final_tilt_file") or "SERIES.tlt"
        xf_for_recon = out_dir / f"{series}_{condition}_{model_name}_raw_to_final.xf"
        script = interop.reconstruction_script(
            Path(raw_stack), xf_for_recon, Path(tlt),
            recon / f"{series}_{condition}_{model_name}_realigned.ali",
            geom["final_shape"], geom["p_final"], n_tilts,
            module_mode=cfg.get("external_tools", {}).get("module_mode", "auto"),
            imod_module=cfg.get("external_tools", {}).get("imod_module", "imod"),
        )
        (recon / "run_imod_reconstruction.sh").write_text(script)
        (recon / "run_imod_reconstruction.sh").chmod(0o755)
        (recon / "README.md").write_text(interop.RECON_README)

    print(f"Wrote exports under: {out_dir}  (status={status})")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
