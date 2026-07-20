#!/usr/bin/env python3
"""Validate, prepare and inspect a MissAlignment v8 project.

The canonical user entry point is ``setup_missalign_project.py``. This command
provides lower-level ``init``, ``validate``, ``prepare``, ``status`` and
``collect-debug`` operations. ``prepare`` imports the native Warp dataset
synchronously but never submits Slurm jobs automatically.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from runtime_env import ensure_scientific_python  # noqa: E402

ALL_CONDITIONS = ("raw_identity", "raw_xf", "raw_xf_translation",
                  "raw_xf_affine_fixed", "ali_identity")


def load_toml(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as fh:
        return tomllib.load(fh)


def _ensure_runtime(settings: Path | None, *, label: str) -> None:
    ensure_scientific_python(
        script=Path(__file__), argv=sys.argv[1:], settings=settings,
        required=("numpy", "mrcfile"), label=label)


# --------------------------------------------------------------------------- #
# Canonical project subcommand interface (validate/prepare/status/collect-debug)
# --------------------------------------------------------------------------- #
SUBCOMMANDS = ("init", "validate", "prepare", "status", "collect-debug", "regenerate-jobs")


def _inputs_from_inventory(cfg, inv, r):
    """Build the orchestrate() inputs dict from the discovered source inventory."""
    inp = cfg.get("input", {})
    return {
        "source_aligned": inv.aligned_stack or inp.get("aligned_stack"),
        "source_raw": inv.raw_stack or inp.get("raw_stack"),
        "source_xf": inv.final_xf or inp.get("final_xf_file"),
        "tilt_file": inv.tilt_file or inp.get("final_tilt_file"),
        "defocus_file": inv.defocus_file or cfg.get("ctf", {}).get("defocus_file"),
        "ctf_com": inv.ctf_com or cfg.get("ctf", {}).get("command_file"),
    }


def _dispatch_subcommand(verb: str, argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog=f"prepare_imod_to_warp.py {verb}")
    ap.add_argument("settings", type=Path, help="Project settings TOML")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--condition", action="append", default=None)
    ap.add_argument("--basename", default=None)
    ap.add_argument("--dataset", default=None, help="Warp dataset ID, for example 5.45Apx")
    ap.add_argument("--ctf-mode", choices=("off", "working", "final", "both"), default=None)
    ap.add_argument("--working-reconstruction", action="store_true")
    ap.add_argument("--refinement-mode",
                    choices=("smoke", "standard", "translation", "rigid", "similarity", "affine2d"), default=None)
    ap.add_argument("--cluster-profile", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--from-step", default=None)
    ap.add_argument("--only-step", default=None)
    ap.add_argument("--force-geometry", action="store_true")
    ap.add_argument("--local-warp-convert", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--force-warp-import", action="store_true",
                    help="(prepare) rebuild the imported Warp dataset even if it is already valid")
    ap.add_argument("--include-checkpoints", action="store_true",
                    help="(collect-debug) include large checkpoints in the bundle")
    ap.add_argument("--allow-unavailable-mode", action="store_true",
                    help="(prepare) allow rigid/similarity even when the constrained fork is unavailable")
    ap.add_argument("--allow-unresolved-legacy", action="store_true",
                    help="(prepare) MIGRATION ONLY: allow an unresolved TOML (run init for production)")
    args = ap.parse_args(argv)
    # Internal multiresolution support remains TOML-driven; it is no longer a CLI override.
    args.extra_binning = None
    args.generate_slurm = True
    if verb in ("init", "validate", "prepare"):
        _ensure_runtime(args.settings, label=f"prepare_imod_to_warp.py {verb}")
    from pipeline import project_workflow as TP
    cfg = load_toml(args.settings)
    if verb == "init":
        from pipeline import init_project as IP
        try:
            res = IP.init_project(cfg, out_dir_override=str(args.out_dir) if args.out_dir else None,
                                  data_dir_override=str(args.data_dir) if args.data_dir else None,
                                  basename_override=args.basename)
        except Exception as exc:
            print(f"ERROR: init failed: {exc}")
            return 1
        print(f"[init] resolved TOML: {res['resolved_toml']}")
        print(f"[init] provenance:    {res['manifests_dir']}")
        print(f"[init] tilt axis:     {res['tilt_axis'][0]} ({res['tilt_axis'][1]})")
        print(f"[init] warp modes:    {res['warp_modes']}")
        print(f"[init] next: prepare_imod_to_warp.py prepare {res['resolved_toml']}")
        return 0
    if verb == "validate":
        return TP.cmd_validate(cfg, args)
    if verb == "prepare":
        from pipeline.orchestrate import orchestrate
        return TP.cmd_prepare(cfg, args, orchestrate_fn=orchestrate, inputs_builder=_inputs_from_inventory)
    if verb == "status":
        return TP.cmd_status(cfg, args)
    if verb == "collect-debug":
        return TP.cmd_collect_debug(cfg, args)
    if verb == "regenerate-jobs":
        return TP.cmd_regenerate_jobs(cfg, args)
    return 2


def main() -> int:
    # Canonical subcommand interface takes precedence; legacy flat flags still work.
    if len(sys.argv) > 1 and sys.argv[1] in SUBCOMMANDS:
        return _dispatch_subcommand(sys.argv[1], sys.argv[2:])
    return _legacy_main()


def _legacy_main() -> int:
    epilog = (
        "\nPROJECT SUBCOMMANDS:\n"
        "  prepare_imod_to_warp.py validate SETTINGS.toml       validate config + discovery + capabilities\n"
        "  prepare_imod_to_warp.py prepare SETTINGS.toml        prepare the project and generate Slurm batches\n"
        "  prepare_imod_to_warp.py status SETTINGS.toml         show project and dataset status\n"
        "  prepare_imod_to_warp.py collect-debug SETTINGS.toml  build a debug bundle\n"
        "Generated batches are written below PROJECT/batches/.\n"
        "The flat-flag form below remains supported for backward compatibility.\n")
    ap = argparse.ArgumentParser(description=(__doc__ or "") + epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("settings", nargs="?", type=Path, default=None, help="Project settings TOML")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--condition", action="append", default=None)
    ap.add_argument("--basename", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--ctf-mode", choices=("off", "working", "final", "both"), default=None,
                    help="External IMOD CTF phase flipping mode.")
    ap.add_argument("--working-reconstruction", action="store_true",
                    help="Generate a local working IMOD reconstruction (tilt).")
    ap.add_argument("--refinement-mode",
                    choices=("smoke", "standard", "translation", "rigid", "similarity", "affine2d"), default=None,
                    help="Real MissAlignment execution mode (NOT the constrained local model). "
                         "rigid/similarity require the MissAlignment fork with constrained integration.")
    ap.add_argument("--cluster-profile", default=None, help="config/cluster_profiles.toml profile (e.g. maxwell).")
    ap.add_argument("--generate-slurm", action="store_true")
    ap.add_argument("--local-warp-convert", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--force-warp-import", action="store_true",
                    help="Rebuild the imported Warp dataset even if it is already valid.")
    ap.add_argument("--submit", action="store_true", help="Submit the SLURM job (default false; never auto-submits).")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--from-step", default=None)
    ap.add_argument("--only-step", default=None)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    # Internal multiresolution support remains TOML-driven; it is no longer a CLI override.
    args.extra_binning = None

    if args.settings is None and (args.data_dir is None or args.out_dir is None):
        ap.error("provide either a settings TOML or both --data-dir and --out-dir")

    cfg: dict[str, Any] = load_toml(args.settings) if args.settings else {}
    _ensure_runtime(args.settings, label="prepare_imod_to_warp.py")
    from alignment_models import interop
    from alignment_models.refinement_config import from_toml_dict
    out_dir = args.out_dir or Path(cfg.get("paths", {}).get("output_dir") or ".")

    # 1. Validate configuration (this is where bad enums/combinations fail fast).
    conditions = args.condition or cfg.get("conversion", {}).get("initial_conditions", ["raw_xf_affine_fixed", "ali_identity"])
    for c in conditions:
        if c not in ALL_CONDITIONS:
            raise SystemExit(f"ERROR: unknown initial condition {c!r}; choose from {list(ALL_CONDITIONS)}")
    refinement = from_toml_dict(cfg.get("refinement", {}))
    for w in refinement.warnings:
        print(f"WARNING: {w}")
    print(f"initial conditions : {conditions}")
    print(f"refinement model   : {refinement.model} (schedule={refinement.schedule})")
    print(f"resolved stages    : {[s.model for s in refinement.resolved_stages()]}")

    # --- Multiresolution + CTF + affine2d orchestration (EXECUTES) -----------
    mr_enabled = (args.extra_binning is not None or args.ctf_mode is not None
                  or cfg.get("multiresolution", {}).get("enabled")
                  or args.refinement_mode is not None)
    if mr_enabled:
        from pipeline.orchestrate import orchestrate
        from pipeline import ctf as _C
        basename = args.basename or cfg.get("project", {}).get("basename") or "series"
        data_dir = args.data_dir or (Path(cfg["paths"]["data_root"]) if cfg.get("paths", {}).get("data_root") else None)
        inp = cfg.get("input", {})
        ctf_inputs = _C.discover_ctf_inputs(data_dir, basename, cfg.get("ctf", {})) if (data_dir and Path(data_dir).is_dir()) else _C.CtfInputs()
        inputs = {
            "source_aligned": inp.get("aligned_stack") or ctf_inputs.aligned_stack,
            "source_raw": inp.get("raw_stack") or ctf_inputs.raw_stack,
            "source_xf": inp.get("final_xf_file"),
            "tilt_file": inp.get("final_tilt_file") or ctf_inputs.angle_file,
            "defocus_file": cfg.get("ctf", {}).get("defocus_file") or ctf_inputs.defocus_file,
            "ctf_com": cfg.get("ctf", {}).get("command_file") or ctf_inputs.command_file,
        }
        toml_binning = int(cfg.get("multiresolution", {}).get("extra_projection_binning", 1))
        if args.extra_binning is not None and int(args.extra_binning) != toml_binning:
            raise SystemExit(
                "ERROR: --extra-binning is a compatibility option and must match "
                f"[multiresolution].extra_projection_binning ({toml_binning}); got {args.extra_binning}. "
                "Edit the resolved TOML instead."
            )
        B = toml_binning
        ctf_mode = args.ctf_mode or cfg.get("ctf", {}).get("mode", "off")
        ref_mode = args.refinement_mode or cfg.get("missalignment", {}).get("refinement_mode", "standard")
        print(f"multiresolution: extra_binning={B}  ctf_mode={ctf_mode}  refinement_mode={ref_mode}  condition={conditions[0]}")
        # Fail-fast validation (mode rules, divisibility) regardless of execute/validate-only.
        _C.validate_ctf_mode(ctf_mode, conditions[0], bool(inputs["source_aligned"]) or ctf_mode in ("off", "final"))
        if args.validate_only:
            print("validate-only: multiresolution + CTF configuration OK.")
            return 0
        if args.dry_run:
            print(f"dry-run: would orchestrate steps for {basename} at {out_dir} (no IMOD executed).")
            return 0
        args.condition = conditions
        result = orchestrate(config=cfg, out_dir=out_dir, data_dir=data_dir, basename=basename,
                             inputs=inputs, args=args)
        print(f"steps run: {result.steps_run}")
        print(f"steps skipped (stale-detection): {result.steps_skipped}")
        for w in result.warnings:
            print(f"WARNING: {w}")
        print(f"binning/CTF manifest: {result.manifest_path}")
        print("NOTE: the config.yaml, run script and .sbatch are REAL; the GPU MissAlignment RUN and "
              "Warp XML evaluation execute on the cluster (MissAlignment/warpylib/GPU not installed locally). "
              "Source data was not modified.")
        if args.submit:
            if "submission" in result.steps_run:
                print("submission: sbatch invoked (see submission.json under the results dir).")
            else:
                print("submission: --submit set but sbatch was not invoked (not on a SLURM host or "
                      "--generate-slurm missing); see WARNINGs above.")
        return 0

    if args.validate_only:
        print("validate-only: configuration OK.")
        return 0

    # 2. Build the deterministic workspace + resolved manifest.
    settings_echo = {
        "conversion": {"initial_conditions": conditions},
        "refinement": {"model": refinement.model, "schedule": refinement.schedule},
    }
    resolved = {
        "conditions": conditions,
        "refinement": refinement.to_dict(),
        "data_dir": str(args.data_dir or cfg.get("paths", {}).get("data_root") or ""),
        "out_dir": str(out_dir),
    }

    if args.dry_run:
        print("dry-run: would create workspace at", interop.workspace_root(out_dir))
        print("dry-run: would write manifest and (with a dataset) extract params + generate .ali")
        return 0

    root = interop.create_workspace(out_dir)
    interop.write_manifest(out_dir, settings_echo, resolved)
    print(f"workspace: {root}")
    print(f"manifest:  {root / 'project_manifest.json'}")

    # 3. Delegate extraction + .ali to the audited setup when a dataset is present.
    data_dir = args.data_dir or (Path(cfg["paths"]["data_root"]) if cfg.get("paths", {}).get("data_root") else None)
    if data_dir and Path(data_dir).is_dir():
        setup = Path(__file__).resolve().parent / "setup_missalign_project.py"
        print(f"\nDelegating IMOD parameter extraction + automatic .ali to {setup.name}.")
        print(f"  run: ./setup_missalign_project.py --data-dir {data_dir} --out-dir {out_dir}")
    else:
        print("\nNo dataset directory provided; configuration + workspace prepared.")

    # 4. Report the deferred warpylib step honestly.
    if importlib.util.find_spec("warpylib") is None:
        print("\nNOTE: Warp .st + XML encoding (scripts/etomo_to_warp.py) requires warpylib, "
              "which is not installed here. Run that step in the MissAlignment/Warp environment "
              "(see CLUSTER_VALIDATION_PLAN.md and docs/interoperability/IMOD_TO_WARP.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
