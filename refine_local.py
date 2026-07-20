#!/usr/bin/env python3
"""Local constrained-residual refinement (coordinate-based; no warpylib/GPU).

Runs the real local refinement forward pass (autograd over per-tilt coordinate
correspondences, staged schedule, scopes, gauge, regularization) and writes a
residual-parameter JSON that ``export_warp_to_imod.py`` turns into an exact
IMOD ``.xf``.

This does NOT use the warpylib image-based forward pass (unavailable locally).
Use ``--self-test`` to run a known-transform recovery end to end, which is a
genuine local demonstration of the optimization + exact export.

    ./refine_local.py settings.toml --self-test rigid --out residual.params.json
    ./export_warp_to_imod.py settings.toml --residual-params residual.params.json --condition ali_identity
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from runtime_env import ensure_scientific_python  # noqa: E402

settings_hint = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
ensure_scientific_python(
    script=Path(__file__), argv=sys.argv[1:], settings=settings_hint,
    required=("numpy", "mrcfile", "torch"), label="refine_local.py")

import numpy as np
import alignment_models as am  # noqa: E402
from alignment_models import coordinate_frames as cf  # noqa: E402
from alignment_models.constraints import GaugeConfig  # noqa: E402
from alignment_models.parameter_scope import ScopeConfig  # noqa: E402
from alignment_models.refine import refine  # noqa: E402
from alignment_models.refinement_config import from_toml_dict  # noqa: E402
from alignment_models.regularization import RegularizationConfig  # noqa: E402
from alignment_models.serialization import write_params_json  # noqa: E402


def _rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def synthetic_truth(kind, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        if kind == "translation":
            A = np.eye(2); t = rng.uniform(-30, 30, 2)
        elif kind == "rigid":
            A = _rot(rng.uniform(-6, 6)); t = rng.uniform(-20, 20, 2)
        elif kind == "similarity":
            A = np.exp(rng.uniform(-0.05, 0.05)) * _rot(rng.uniform(-5, 5)); t = rng.uniform(-15, 15, 2)
        elif kind == "affine":
            A = _rot(rng.uniform(-4, 4)) @ np.array([[np.exp(rng.uniform(-0.05, 0.05)), rng.uniform(-0.06, 0.06)],
                                                     [0.0, np.exp(rng.uniform(-0.05, 0.05))]])
            t = rng.uniform(-12, 12, 2)
        else:
            raise SystemExit(f"unknown truth kind {kind!r}")
        out.append((A, t))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("settings", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="Residual-params JSON to write")
    ap.add_argument("--self-test", choices=["translation", "rigid", "similarity", "affine"], default=None,
                    help="Generate a known truth of this kind and refine it (recovery demo)")
    ap.add_argument("--n-tilts", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gauge", action="store_true", help="Disable gauge fixing (use for coordinate recovery, where global DOF are observable)")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with args.settings.open("rb") as fh:
        cfg_dict = tomllib.load(fh)
    config = from_toml_dict(cfg_dict.get("refinement", {}))
    for w in config.warnings:
        print(f"WARNING: {w}")
    if args.no_gauge:
        config.gauge = GaugeConfig(anchor_tilt="none", zero_mean_rotation=False,
                                   zero_mean_log_scale=False, zero_mean_shear=False)
    print(f"refinement model: {config.model}  schedule: {config.schedule}  "
          f"stages: {[s.model for s in config.resolved_stages()]}")

    if args.validate_only:
        print("validate-only: refinement configuration OK.")
        return 0

    if args.self_test is None:
        raise SystemExit("ERROR: provide --self-test <kind> (real correspondence input files are not yet a CLI option).")

    geom = cfg_dict.get("geometry", {})
    dims = geom.get("aligned_dimensions_xyz") or geom.get("raw_dimensions_xyz") or [256, 192, 1]
    shape = (int(dims[0]), int(dims[1]))
    pix = float(geom.get("aligned_pixel_size_A") or geom.get("raw_pixel_size_A") or 10.0)
    n = args.n_tilts
    center = cf.physical_center_xy(shape, pix)
    angles = list(np.linspace(-60, 60, n))
    xs = np.linspace(0, shape[0] * pix, 6); ys = np.linspace(0, shape[1] * pix, 5)
    base = np.array([(x, y) for y in ys for x in xs])
    src = np.tile(base, (n, 1, 1))
    truths = synthetic_truth(args.self_test, n, args.seed)
    tgt = np.stack([(src[i] - center) @ A.T + t + center for i, (A, t) in enumerate(truths)])

    if args.dry_run:
        print(f"dry-run: would refine {args.self_test} truth, {n} tilts, shape {shape}, pixel {pix} A")
        return 0

    result = refine(config, src, tgt, angles, shape, pix, iters_per_stage=args.iters)
    print(f"final data RMS = {result.final_data_rms_A:.4g} A   converged = {result.converged}")
    print(f"gauge report   = {result.gauge}")

    # Recovery vs truth (relative, since absolute may differ by gauge if enabled).
    model = am.get_model(config.model)
    fit_A = model.matrices_numpy(result.params)
    truth_A = np.stack([A for A, _ in truths])
    rel = float(np.max(np.abs(fit_A - truth_A))) if not args.no_gauge else None
    print(f"matrix max |fit - truth| = {np.max(np.abs(fit_A - truth_A)):.4g}"
          + ("  (gauge enabled: compare relative geometry)" if not args.no_gauge else ""))

    out = args.out or args.settings.with_suffix(".residual.params.json")
    write_params_json(out, model, result.params, tilt_angles=angles,
                      extra={"refinement": config.to_dict(),
                             "self_test_truth": args.self_test,
                             "final_data_rms_A": result.final_data_rms_A,
                             "stage_history": result.stage_history})
    print(f"Wrote residual params: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
