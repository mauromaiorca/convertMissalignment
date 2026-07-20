#!/usr/bin/env python3
"""Backward-compatible command dispatcher for the MissAlignment pipeline.

The public command remains ``convertMissalignment`` so existing shell scripts and
editable installations can be replaced without changing their invocation.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import shutil
import sys
from importlib import metadata
from pathlib import Path

from ._version import __pipeline_version__, __version__

DISTRIBUTION = "convertMissAlignment"
PACKAGE_DIR = Path(__file__).resolve().parent
APPLICATION_ROOT = PACKAGE_DIR.parent
PIPELINE_VERSION_FILE = APPLICATION_ROOT / "PIPELINE_VERSION"

# Historical names accepted by older command lines. Values written into project
# settings are always the canonical v8 condition names.
CONDITION_ALIASES = {
    "identity": "raw_identity",
    "raw": "raw_identity",
    "xf": "raw_xf",
    "translation": "raw_xf_translation",
    "affine": "raw_xf_affine_fixed",
    "affine_fixed": "raw_xf_affine_fixed",
    "ali": "ali_identity",
}

COMMANDS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "setup": (
        "setup_missalign_project",
        "create and prepare a project",
        (),
    ),
    "prepare": (
        "prepare_imod_to_warp",
        "run lower-level project preparation operations",
        (),
    ),
    "input": (
        "prepare_missalignment_input",
        "prepare MissAlignment input snapshots",
        (),
    ),
    "prepare-input": (
        "prepare_missalignment_input",
        "alias for the input command",
        (),
    ),
    "preprocess": (
        "warp_preprocess",
        "create a lower-resolution Warp dataset",
        (),
    ),
    "status": (
        "prepare_imod_to_warp",
        "show the current project and dataset state",
        ("status",),
    ),
    "export": (
        "export_warp_to_imod",
        "export results to IMOD",
        (),
    ),
    "refine": (
        "refine_local",
        "run local refinement utilities",
        (),
    ),
    "imod-recon": (
        "setup_imod_recon",
        "run the IMOD reconstruction compatibility entry point",
        (),
    ),
}


def distribution_version() -> str:
    """Return installed metadata, falling back to the source version."""
    for name in (DISTRIBUTION, "convertmissalignment"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return __version__


def pipeline_version() -> str:
    if PIPELINE_VERSION_FILE.is_file():
        value = PIPELINE_VERSION_FILE.read_text(encoding="utf-8").strip()
        if value:
            return value
    return __pipeline_version__


def normalise_setup_arguments(argv: list[str]) -> list[str]:
    """Translate historical ``--condition`` values into canonical v8 values."""
    result: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--condition" and index + 1 < len(argv):
            result.extend((item, CONDITION_ALIASES.get(argv[index + 1], argv[index + 1])))
            index += 2
            continue
        if item.startswith("--condition="):
            value = item.split("=", 1)[1]
            result.append(f"--condition={CONDITION_ALIASES.get(value, value)}")
            index += 1
            continue
        result.append(item)
        index += 1
    return result


def run_module(module_name: str, argv: list[str], *, prefix: tuple[str, ...] = ()) -> int:
    module = importlib.import_module(module_name)
    module_main = getattr(module, "main", None)
    if module_main is None:
        raise SystemExit(f"ERROR: {module_name} does not expose main()")

    previous = sys.argv
    try:
        sys.argv = [f"{Path(previous[0]).name} {module_name}", *prefix, *argv]
        result = module_main()
    finally:
        sys.argv = previous
    return int(result or 0)


def command_path() -> str | None:
    invoked = Path(sys.argv[0]).expanduser()
    if invoked.exists():
        return str(invoked.resolve())
    return shutil.which(Path(sys.argv[0]).name)


def print_where() -> int:
    print(f"distribution : {DISTRIBUTION}")
    print(f"version      : {distribution_version()}")
    source_version = pipeline_version()
    if source_version:
        print(f"pipeline     : {source_version}")
    print(f"module       : {Path(__file__).resolve()}")
    print(f"package      : {PACKAGE_DIR}")
    print(f"source root  : {APPLICATION_ROOT}")
    print(f"executable   : {command_path() or 'not found on PATH'}")
    print(f"python       : {Path(sys.executable).resolve()}")
    print(f"environment  : {Path(sys.prefix).resolve()}")

    try:
        dist = metadata.distribution(DISTRIBUTION)
        direct_url = Path(dist.locate_file("")) / f"{dist.metadata['Name'].lower()}-{dist.version}.dist-info" / "direct_url.json"
        # The exact dist-info spelling is normalised by installers, so search if
        # the direct construction does not exist.
        if not direct_url.is_file():
            candidates = list(Path(dist.locate_file("")).glob("convertmissalignment-*.dist-info/direct_url.json"))
            direct_url = candidates[0] if candidates else direct_url
        if direct_url.is_file():
            print(f"installation : {direct_url}")
    except (metadata.PackageNotFoundError, KeyError, OSError):
        pass
    return 0


def status_line(label: str, ok: bool, detail: str) -> None:
    marker = "OK" if ok else "MISSING"
    print(f"{marker:7} {label:20} {detail}")


def doctor() -> int:
    print("Python environment")
    python_ok = sys.version_info >= (3, 11)
    status_line("Python >= 3.11", python_ok, sys.version.split()[0])
    status_line("package metadata", distribution_version() == __version__, distribution_version())

    required_modules = ("numpy", "mrcfile")
    optional_modules = ("warpylib", "torch", "matplotlib")
    missing_required: list[str] = []

    print("\nPython modules")
    for name in required_modules + optional_modules:
        found = importlib.util.find_spec(name) is not None
        required = name in required_modules
        if required and not found:
            missing_required.append(name)
        status_line(name, found, "required" if required else "workflow-dependent")

    print("\nExternal commands")
    external = (
        ("sbatch", "Slurm submission"),
        ("srun", "Slurm execution"),
        ("WarpTools", "Warp processing"),
        ("newstack", "IMOD stack conversion"),
        ("tilt", "IMOD reconstruction"),
        ("trimvol", "IMOD volume processing"),
        ("miss-alignment", "MissAlignment executable"),
    )
    for command, purpose in external:
        path = shutil.which(command)
        status_line(command, path is not None, path or purpose)

    print("\nApplication files")
    for relative in (
        "config/cluster_profiles.toml",
        "config/conversion_presets.toml",
        "prepare_missalignment_input.py",
        "setup_missalign_project.py",
    ):
        path = APPLICATION_ROOT / relative
        status_line(relative, path.is_file(), str(path))

    if not python_ok or missing_required:
        print("\nResult: installation is incomplete for local configuration commands.")
        return 2
    print("\nResult: Python installation is usable. Missing cluster commands may be expected off-cluster.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description=(
            "Backward-compatible command interface for the IMOD/eTomo, Warp and "
            "MissAlignment processing pipeline."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {distribution_version()}",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("version", help="show the installed package version")
    subparsers.add_parser("where", help="show where the command and source are installed")
    subparsers.add_parser("doctor", help="check Python, cluster tools and application files")
    for command, (_, help_text, _) in COMMANDS.items():
        subparsers.add_parser(command, help=help_text, add_help=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, remainder = parser.parse_known_args(raw_argv)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "version":
        print(distribution_version())
        return 0
    if args.command == "where":
        return print_where()
    if args.command == "doctor":
        return doctor()

    module_name, _, prefix = COMMANDS[args.command]
    if args.command == "setup":
        remainder = normalise_setup_arguments(remainder)
    return run_module(module_name, remainder, prefix=prefix)


if __name__ == "__main__":
    raise SystemExit(main())
