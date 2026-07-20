#!/usr/bin/env python3
"""Create one resolved MissAlignment project configuration from an IMOD dataset.

Normal use:
    ./setup_missalign_project.py --data-dir DATA --basename NAME --out-dir OUT \
        --condition raw_xf_affine_fixed
    ./setup_missalign_project.py show-resolved OUT/project_settings.toml

The setup command performs discovery and geometry measurement once, writes the
single authoritative file ``OUT/project_settings.toml``, imports the native IMOD
geometry into Warp synchronously, and prepares the run and Slurm jobs. It does not
submit jobs. Use ``--no-prepare`` to stop after configuration.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import tomllib
from pathlib import Path

DEFAULT_ENV = ""
DEFAULT_EXTRA_PROJECTION_BINNING = 1
CONDITIONS = (
    "raw_identity",
    "raw_xf",
    "raw_xf_translation",
    "raw_xf_affine_fixed",
    "ali_identity",
)
CONDITION_ALIASES = {
    "identity": "raw_identity",
    "raw": "raw_identity",
    "xf": "raw_xf",
    "translation": "raw_xf_translation",
    "affine": "raw_xf_affine_fixed",
    "affine_fixed": "raw_xf_affine_fixed",
    "ali": "ali_identity",
}
ACCEPTED_CONDITIONS = CONDITIONS + tuple(CONDITION_ALIASES)


def _normalise_conditions(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ("raw_xf_affine_fixed",)
    return tuple(CONDITION_ALIASES.get(value, value) for value in values)


def _load_cluster_profile(name: str) -> dict:
    path = Path(__file__).resolve().parent / "config" / "cluster_profiles.toml"
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    profile = dict(data.get(name) or {})
    if not profile:
        raise ValueError(f"cluster profile not found in {path}: {name}")
    nested = (data.get(name, {}) or {}).get("reconstruction_cluster")
    if nested:
        profile["reconstruction_cluster"] = dict(nested)
    warptools = (data.get(name, {}) or {}).get("warptools_reconstruction_cluster")
    if warptools:
        profile["warptools_reconstruction_cluster"] = dict(warptools)
    return profile


def _imod_executable(bin_dir: str, program: str) -> str:
    if bin_dir:
        return str(Path(bin_dir) / program)
    return program


def xyz_type(text: str) -> tuple[int, int, int]:
    """Parse X,Y,Z without converting it into a character string."""
    parts = [p for p in re.split(r"[xX,;\s]+", text.strip()) if p]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "expected three positive integers, for example 2046,494,2880"
        )
    try:
        values = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("XYZ values must be integers") from exc
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("XYZ values must be positive")
    return values


def _maybe_reexec_under_env(argv: list[str]) -> None:
    """Enter an explicitly configured environment before importing dependencies."""
    required = ("numpy", "mrcfile")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if not missing:
        return

    if os.environ.get("MISSALIGN_TESTDEBUG_REEXEC") == "1":
        print(
            "ERROR: the selected MissAlignment Python still lacks required modules: "
            + ", ".join(missing)
            + f". Interpreter: {sys.executable}."
        )
        raise SystemExit(2)

    env_dir = ""
    for index, value in enumerate(argv):
        if value == "--missalign-env" and index + 1 < len(argv):
            env_dir = argv[index + 1]
        elif value.startswith("--missalign-env="):
            env_dir = value.split("=", 1)[1]
    env_dir = os.environ.get("MISSALIGN_ENV", env_dir).strip()

    if not env_dir:
        print(
            f"ERROR: this command requires {', '.join(required)}. Current interpreter "
            f"{sys.executable} lacks {', '.join(missing)}. Activate a suitable environment, "
            "pass --missalign-env, or set MISSALIGN_ENV."
        )
        raise SystemExit(2)

    env_python = Path(env_dir).expanduser() / "bin" / "python"
    if env_python.is_file() and os.access(env_python, os.X_OK):
        try:
            same_python = env_python.resolve() == Path(sys.executable).resolve()
        except OSError:
            same_python = False
        if not same_python:
            print(
                f"[setup] missing {', '.join(missing)} here; re-exec under {env_python}",
                flush=True,
            )
            os.environ["MISSALIGN_TESTDEBUG_REEXEC"] = "1"
            os.execv(
                str(env_python),
                [str(env_python), str(Path(__file__).resolve()), *argv],
            )
            return

    print(
        f"ERROR: no suitable Python was found at {env_python}. Activate the environment "
        "or correct --missalign-env / MISSALIGN_ENV."
    )
    raise SystemExit(2)


def _add_test_debug_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--test-debug", action="store_true")
    parser.add_argument("--test-debug-collect", action="store_true")
    parser.add_argument("--debug-run", type=Path)
    parser.add_argument("--test-debug-full", action="store_true", default=False)
    parser.add_argument("--debug-max-image-dimension", type=int, default=512)
    parser.add_argument("--debug-smoke-max-image-dimension", type=int, default=256)
    parser.add_argument("--debug-tilt-count", type=int, default=9)
    parser.add_argument("--debug-bundle-max-mb", type=int, default=50)
    parser.add_argument("--debug-all-tilts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-run-imod", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-generate-slurm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--submit-debug", action="store_true", default=False)
    parser.add_argument("--debug-keep-intermediates", action="store_true", default=False)
    parser.add_argument("--debug-max-memory-mb", type=int, default=256)
    parser.add_argument("--debug-command-timeout-seconds", type=int, default=600)
    parser.add_argument("--debug-global-timeout-seconds", type=int, default=1800)
    parser.add_argument("--debug-resume", action="store_true", default=False)
    parser.add_argument("--debug-from-stage")
    parser.add_argument("--debug-only-stage")
    parser.add_argument("--debug-run-id")
    parser.add_argument("--force-debug", action="store_true", default=False)


def _dispatch_test_debug(argv: list[str]) -> int:
    _maybe_reexec_under_env(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
    from pipeline import test_debug as debug

    parser = argparse.ArgumentParser(description="Bounded diagnostic harness")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--basename")
    parser.add_argument(
        "--missalign-env", default=None,
        help="absolute path to the scientific Python environment; defaults to MISSALIGN_ENV, the cluster profile, or the active environment",
    )
    parser.add_argument("--condition", action="append", choices=ACCEPTED_CONDITIONS)
    _add_test_debug_args(parser)
    args = parser.parse_args(argv)

    if args.test_debug_collect:
        run = args.debug_run
        if run is None and args.out_dir:
            latest = args.out_dir / "test_debug" / "LATEST_DEBUG_RUN"
            if latest.is_file():
                run = Path(latest.read_text().strip())
        if run is None:
            parser.error(
                "--test-debug-collect requires --debug-run, or --out-dir with LATEST_DEBUG_RUN"
            )
        return debug.collect_test_debug(str(run))

    if not (args.data_dir and args.out_dir):
        parser.error("--test-debug requires --data-dir and --out-dir")

    resolved_environment = (
        args.missalign_env
        or os.environ.get("MISSALIGN_ENV")
        or sys.prefix
    )

    options = debug.DebugOptions(
        data_dir=str(args.data_dir.expanduser().resolve()),
        out_dir=str(args.out_dir.expanduser().resolve()),
        basename=args.basename,
        max_image_dim=args.debug_max_image_dimension,
        smoke_max_image_dim=args.debug_smoke_max_image_dimension,
        tilt_count=args.debug_tilt_count,
        bundle_max_mb=args.debug_bundle_max_mb,
        quick=not args.test_debug_full,
        all_tilts=args.debug_all_tilts or args.test_debug_full,
        run_imod=args.debug_run_imod or args.test_debug_full,
        generate_slurm=args.debug_generate_slurm,
        submit_debug=args.submit_debug,
        keep_intermediates=args.debug_keep_intermediates,
        max_memory_mb=args.debug_max_memory_mb,
        command_timeout_s=args.debug_command_timeout_seconds,
        global_timeout_s=args.debug_global_timeout_seconds,
        resume=args.debug_resume,
        from_stage=args.debug_from_stage,
        only_stage=args.debug_only_stage,
        run_id=args.debug_run_id,
        force=args.force_debug,
        missalign_env=resolved_environment,
        conditions=_normalise_conditions(args.condition),
    )
    return debug.run_test_debug(options)


def _canonical_init(
    *,
    data_dir: Path,
    out_dir: Path,
    basename: str,
    conditions: tuple[str, ...],
    missalign_env: str,
    target_shape: tuple[int, int, int] | None,
    target_pixel_size: float | None,
    tilt_axis_angle: float | None,
    reconstruction_stack: Path | None,
    extra_binning: int,
    cluster_profile: str,
    cluster_profile_data: dict,
    imod_module: str | None,
    imod_bin_dir: str | None,
    warp_module: str | None,
    reconstruct_snapshots: tuple[str, ...],
    disable_imod_reconstruction: bool,
    imod_cpu_partition: str,
    imod_cpus: int,
    imod_memory: str,
    imod_time: str,
    imod_newst_bin: int,
    imod_halfmaps: bool,
    imod_use_gpu: bool,
    imod_gpu_id: int,
):
    """Discover and measure once, then write the sole authoritative TOML."""
    import hashlib

    sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
    from pipeline import init_project
    from pipeline import project_config

    input_table = {}
    if reconstruction_stack is not None:
        rec = reconstruction_stack.expanduser()
        if not rec.is_absolute():
            rec = data_dir / rec
        rec = rec.resolve()
        if not rec.is_file():
            raise project_config.ConfigError(
                f"--reconstruction-stack is not a readable file: {rec}"
            )
        input_table["source_reconstruction"] = str(rec)

    geometry = {}
    if target_shape is not None:
        geometry["target_volume_shape_xyz"] = list(target_shape)
    if target_pixel_size is not None:
        geometry["target_pixel_size_A"] = float(target_pixel_size)
    if tilt_axis_angle is not None:
        geometry["tilt_axis_angle_deg"] = float(tilt_axis_angle)

    profile = dict(cluster_profile_data)
    resolved_imod_module = imod_module or profile.get("imod_module", "")
    resolved_imod_bin = imod_bin_dir or profile.get("imod_bin_dir", "")
    resolved_warp_module = warp_module or profile.get("warp_module", "")
    reconstruction_profile = dict(profile.get("reconstruction_cluster") or {})
    warptools_reconstruction_cluster = dict(profile.get("warptools_reconstruction_cluster") or {})
    reconstruction_partition = imod_cpu_partition or reconstruction_profile.get("partition") or profile.get("cpu_partition", "")
    reconstruction = {
        "enabled": not disable_imod_reconstruction,
        "backend": "imod",
        "snapshots": list(reconstruct_snapshots),
        "canonical_snapshot": "full",
        "diagnostic_snapshots": [s for s in reconstruct_snapshots if s != "full"],
        "warptools": {
            "enabled": True,
            "executable": profile.get("warp_tools_executable", "WarpTools"),
            "output_angpix_A": 0.0,
            "device_list": "0",
            "perdevice": 1,
            "dose_policy": "preserve_if_valid_else_synthetic_monotonic_epsilon",
        },
        "imod": {
            "imod_module": resolved_imod_module,
            "imod_bin_dir": resolved_imod_bin,
            "newstack_executable": _imod_executable(resolved_imod_bin, "newstack"),
            "tilt_executable": _imod_executable(resolved_imod_bin, "tilt"),
            "submfg_executable": _imod_executable(resolved_imod_bin, "submfg"),
            "ctfphaseflip_executable": _imod_executable(resolved_imod_bin, "ctfphaseflip"),
            "execution_mode": "submfg_command_file",
            "newst_bin": int(imod_newst_bin),
            "use_gpu": bool(imod_use_gpu),
            "gpu_id": int(imod_gpu_id),
            "halfmaps": bool(imod_halfmaps),
        },
    }
    cluster = {
        "profile": cluster_profile,
        "environment": missalign_env,
        "module_init_script": profile.get("module_init_script"),
        "imod_module": resolved_imod_module,
        "imod_bin_dir": resolved_imod_bin,
        "warp_module": resolved_warp_module,
        "warp_tools_executable": profile.get("warp_tools_executable", "WarpTools"),
        "warp_worker_executable": profile.get("warp_worker_executable", "WarpWorker"),
        "partition": profile.get("gpu_partition") or profile.get("partition", "vds"),
        "constraint": profile.get("gpu_constraint") or profile.get("constraint", "V100"),
        "gres": profile.get("gres", ""),
        "cpu_partition": profile.get("cpu_partition"),
        "reconstruction_cluster": {
            "partition": str(reconstruction_partition),
            "nodes": int(reconstruction_profile.get("nodes", 1)),
            "tasks": int(reconstruction_profile.get("tasks", 1)),
            "cpus_per_task": int(imod_cpus),
            "memory": str(imod_memory),
            "time": str(imod_time),
            "partition_source": "cli" if imod_cpu_partition else "cluster_profile",
        },
        "warptools_reconstruction_cluster": {
            "partition": str(warptools_reconstruction_cluster.get("partition") or profile.get("gpu_partition") or "vds"),
            "constraint": str(warptools_reconstruction_cluster.get("constraint") or profile.get("gpu_constraint") or "V100"),
            "gres": str(warptools_reconstruction_cluster.get("gres", profile.get("gres", ""))),
            "nodes": int(warptools_reconstruction_cluster.get("nodes", 1)),
            "tasks": int(warptools_reconstruction_cluster.get("tasks", 1)),
            "cpus_per_task": int(warptools_reconstruction_cluster.get("cpus_per_task", 16)),
            "memory": str(warptools_reconstruction_cluster.get("memory", "128G")),
            "time": str(warptools_reconstruction_cluster.get("time", "24:00:00")),
        },
    }

    config = {
        "project": {"basename": basename},
        "paths": {"data_root": str(data_dir), "output_dir": str(out_dir)},
        "input": input_table,
        "geometry": geometry,
        "conversion": {"initial_conditions": list(conditions)},
        "missalignment": {
            "refinement_mode": "standard",
            "result_backend": "warp_xml",
        },
        "ctf": {"mode": "off"},
        "multiresolution": {"extra_projection_binning": int(extra_binning)},
        "cluster": cluster,
        "reconstruction": reconstruction,
    }
    result = init_project.init_project(
        config,
        out_dir_override=str(out_dir),
        data_dir_override=str(data_dir),
        basename_override=basename,
    )
    canonical = Path(result["resolved_toml"])
    digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
    manifests = Path(result["manifests_dir"])
    (manifests / "config_provenance.json").write_text(
        json.dumps(
            {
                "authoritative_toml": str(canonical),
                "content_sha256": digest,
                "tilt_axis": result["tilt_axis"],
                "warp_modes": result["warp_modes"],
            },
            indent=2,
        )
        + "\n"
    )

    resolved = project_config.load(canonical)
    problems = project_config.validate(
        resolved, require_geometry=True, require_resolved=True
    )
    if problems:
        raise project_config.ConfigError(
            "resolved config failed validation:\n  - " + "\n  - ".join(problems)
        )
    return canonical, resolved


def _print_resolved(config) -> None:
    geometry = config.geometry
    print(f"basename            : {config.basename}")
    print(f"conditions          : {config.conditions}")
    print(f"warp alignment modes: {config.warp_alignment_modes}")
    print(
        f"refinement_mode     : {config.refinement_mode}   "
        f"result_backend: {config.result_backend}"
    )
    print(
        f"raw detector        : {geometry.raw_shape_xyz} @ "
        f"{geometry.raw_pixel_size_A} A/px"
    )
    print(
        f"aligned detector    : {geometry.aligned_shape_xyz} @ "
        f"{geometry.aligned_pixel_size_A} A/px"
    )
    print(
        f"tilt axis           : {geometry.tilt_axis_angle_deg} deg "
        f"({geometry.tilt_axis_source})"
    )
    print(
        f"target volume       : {geometry.target_volume_shape_xyz} @ "
        f"{geometry.target_pixel_size_A} A/px"
    )
    print(
        f"target physical (A) : {geometry.target_volume_physical_A}  "
        f"({geometry.target_volume_source})"
    )
    if geometry.raw_shape_xyz and geometry.target_pixel_size_A:
        rx, ry, rz = geometry.raw_shape_xyz
        wp = float(geometry.target_pixel_size_A)
        print(f"native Warp dataset : {wp:g}Apx (created from {rx} × {ry} × {rz} projections)")
        print("derived datasets     : create with warp_preprocess.py --bin N")


def _show_resolved(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="setup_missalign_project.py show-resolved")
    parser.add_argument("settings", type=Path)
    args = parser.parse_args(argv)
    if not args.settings.is_file():
        parser.error(f"resolved TOML not found: {args.settings}")
    sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
    from pipeline import project_config

    config = project_config.load(args.settings)
    if not config.resolved:
        parser.error("the TOML is not resolved; run setup first")
    _print_resolved(config)
    return 0


def _initialise(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="setup_missalign_project.py",
        description="Discover an IMOD project and write one resolved TOML.",
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--basename")
    parser.add_argument("--condition", action="append", choices=ACCEPTED_CONDITIONS)
    parser.add_argument("--cluster-profile", default="maxwell")
    parser.add_argument(
        "--missalign-env", default=None,
        help="absolute path to the scientific Python environment; defaults to MISSALIGN_ENV, the cluster profile, or the active environment",
    )
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument("--reconstruction-stack", type=Path)
    geometry.add_argument("--xyz", type=xyz_type)
    parser.add_argument("--target-pixel-size", type=float)
    parser.add_argument("--tilt-axis-angle", type=float)
    parser.add_argument(
        "--no-prepare", action="store_true",
        help="write the resolved TOML and provenance only; do not generate the project tree or Slurm batches",
    )
    parser.add_argument("--local-warp-convert", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--force-warp-import", action="store_true",
        help="rebuild the imported Warp dataset even if a valid import already exists",
    )
    parser.add_argument(
        "--reconstruct-snapshots", default="pre_missalign,smoke,full",
        help="comma-separated reconstruction snapshots to generate",
    )
    parser.add_argument("--disable-imod-reconstruction", action="store_true")
    parser.add_argument("--imod-module", default=None)
    parser.add_argument("--imod-bin-dir", default=None)
    parser.add_argument("--warp-module", default=None)
    parser.add_argument("--imod-cpu-partition", default="")
    parser.add_argument("--imod-cpus", type=int, default=16)
    parser.add_argument("--imod-memory", default="64G")
    parser.add_argument("--imod-time", default="08:00:00")
    parser.add_argument("--imod-newst-bin", type=int, default=0)
    parser.add_argument("--imod-halfmaps", action="store_true")
    parser.add_argument("--imod-use-gpu", action="store_true")
    parser.add_argument("--imod-gpu-id", type=int, default=0)
    args = parser.parse_args(argv)
    if args.condition and len(args.condition) != 1:
        parser.error("version 8 requires exactly one --condition per project")

    _maybe_reexec_under_env(argv)
    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not data_dir.is_dir():
        parser.error(f"--data-dir is not a directory: {data_dir}")

    basename = args.basename
    if not basename:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
        from pipeline import discovery

        try:
            basename, _ = discovery.infer_basename(data_dir)
        except discovery.DiscoveryError as exc:
            parser.error(str(exc))
        print(f"[setup] inferred basename: {basename}")
    snapshots = tuple(s.strip() for s in args.reconstruct_snapshots.split(",") if s.strip())
    invalid_snapshots = [s for s in snapshots if s not in {"pre_missalign", "smoke", "full"}]
    if invalid_snapshots:
        parser.error(f"--reconstruct-snapshots contains invalid values: {', '.join(invalid_snapshots)}")
    if not snapshots and not args.disable_imod_reconstruction:
        parser.error("--reconstruct-snapshots must name at least one snapshot unless reconstruction is disabled")
    try:
        cluster_profile_data = _load_cluster_profile(args.cluster_profile)
    except Exception as exc:
        parser.error(str(exc))

    missalign_env = (
        args.missalign_env
        or os.environ.get("MISSALIGN_ENV")
        or str(cluster_profile_data.get("missalign_environment") or "").strip()
        or sys.prefix
    )

    try:
        toml_path, resolved = _canonical_init(
            data_dir=data_dir,
            out_dir=out_dir,
            basename=basename,
            conditions=_normalise_conditions(args.condition),
            missalign_env=str(Path(missalign_env).expanduser().resolve()),
            target_shape=args.xyz,
            target_pixel_size=args.target_pixel_size,
            tilt_axis_angle=args.tilt_axis_angle,
            reconstruction_stack=args.reconstruction_stack,
            extra_binning=DEFAULT_EXTRA_PROJECTION_BINNING,
            cluster_profile=args.cluster_profile,
            cluster_profile_data=cluster_profile_data,
            imod_module=args.imod_module,
            imod_bin_dir=args.imod_bin_dir,
            warp_module=args.warp_module,
            reconstruct_snapshots=snapshots,
            disable_imod_reconstruction=args.disable_imod_reconstruction,
            imod_cpu_partition=args.imod_cpu_partition,
            imod_cpus=args.imod_cpus,
            imod_memory=args.imod_memory,
            imod_time=args.imod_time,
            imod_newst_bin=args.imod_newst_bin,
            imod_halfmaps=args.imod_halfmaps,
            imod_use_gpu=args.imod_use_gpu,
            imod_gpu_id=args.imod_gpu_id,
        )
    except Exception as exc:
        print(f"ERROR: setup failed: {exc}", file=sys.stderr)
        return 1

    print(f"[setup] resolved canonical TOML: {toml_path}", flush=True)
    _print_resolved(resolved)
    sys.stdout.flush()

    if args.no_prepare:
        print("\n[setup] preparation skipped (--no-prepare). To prepare later, run:")
        print(f"  ./prepare_imod_to_warp.py prepare {toml_path}")
        return 0

    import subprocess
    prepare_script = Path(__file__).resolve().parent / "prepare_imod_to_warp.py"
    command = [sys.executable, str(prepare_script), "prepare", str(toml_path)]
    if args.force_warp_import:
        command.append("--force-warp-import")
    print("\n[setup] preparing the run and Slurm jobs...", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(f"ERROR: prepare failed with exit code {completed.returncode}", file=sys.stderr)
        return completed.returncode
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if "--test-debug" in argv or "--test-debug-collect" in argv:
        return _dispatch_test_debug(argv)
    if argv and argv[0] == "show-resolved":
        return _show_resolved(argv[1:])
    if argv and argv[0] == "init":
        argv = argv[1:]
    return _initialise(argv)


if __name__ == "__main__":
    raise SystemExit(main())
