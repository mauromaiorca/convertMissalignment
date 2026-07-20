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
import subprocess
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


def find_reconstruct_batches(project: Path) -> list[Path]:
    """Every generated Warp reconstruction batch, one per imported dataset."""
    return sorted((project / "batches" / "warp_data").glob("*/reconstruct.sbatch"))


def reconstruct(argv: list[str]) -> int:
    """Reconstruct an imported Warp dataset: the step immediately after ``setup``."""
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} reconstruct",
        description=(
            "Reconstruct the imported Warp dataset. Run this straight after 'setup', "
            "which generates the batch but never submits it."
        ),
    )
    parser.add_argument("directory", nargs="?", default=".",
                        help="project directory (default: current directory)")
    parser.add_argument("--dataset", default=None,
                        help="dataset id, e.g. 1.363Apx (needed only when several exist)")
    parser.add_argument("--print", dest="print_only", action="store_true",
                        help="print the submission command instead of running it")
    parser.add_argument("--local", action="store_true",
                        help="run the batch directly with bash (use inside an interactive allocation)")
    args = parser.parse_args(argv)

    project = Path(args.directory).expanduser()
    if not (project / "project_settings.toml").is_file():
        print(f"ERROR: not a project directory (no project_settings.toml): {project}", file=sys.stderr)
        print("Create it first:  convertMissalignment setup --data-dir DATA --out-dir PROJECT", file=sys.stderr)
        return 2

    batches = find_reconstruct_batches(project)
    if args.dataset:
        batches = [b for b in batches if b.parent.name == args.dataset]
    if not batches:
        target = project / "batches" / "warp_data"
        print(f"ERROR: no reconstruct.sbatch found under {target}", file=sys.stderr)
        if args.dataset:
            print(f"       (no dataset named {args.dataset!r})", file=sys.stderr)
        print("       Re-run setup, or regenerate the jobs with:", file=sys.stderr)
        print(f"       convertMissalignment prepare regenerate-jobs {project / 'project_settings.toml'}",
              file=sys.stderr)
        return 2
    if len(batches) > 1:
        print("Several datasets are available; pick one with --dataset:", file=sys.stderr)
        for batch in batches:
            print(f"  {batch.parent.name}", file=sys.stderr)
        return 2

    batch = batches[0]
    dataset = batch.parent.name
    print(f"[reconstruct] project : {project}")
    print(f"[reconstruct] dataset : {dataset}")
    print(f"[reconstruct] batch   : {batch}")

    if args.print_only:
        print(f"sbatch {batch}")
        return 0

    if args.local:
        command = ["bash", str(batch)]
    else:
        if shutil.which("sbatch") is None:
            print("ERROR: sbatch not found on PATH.", file=sys.stderr)
            print("       Use --local inside an interactive allocation, or --print to see the command.",
                  file=sys.stderr)
            return 2
        command = ["sbatch", str(batch)]

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(f"ERROR: {command[0]} failed with exit code {completed.returncode}", file=sys.stderr)
        return completed.returncode

    print(f"[reconstruct] logs    : {project / 'logs' / 'warp_data' / dataset}")
    print(f"[reconstruct] output  : {project / 'warp_data' / dataset / 'reconstructions'}")
    print(f"[reconstruct] next    : convertMissalignment input --directory {project}")
    return 0


def reconstruct_main() -> int:
    """Console entry point for ``missalign-reconstruct``."""
    return reconstruct(sys.argv[1:])


def _human_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return ""


def _report(label: str, path: Path | None, project: Path, *, hint: str = "") -> None:
    """One inventory line: state, project-relative path, and the resolved target."""
    if path is None or not path.exists():
        print(f"  {'--':4} {label:22} {hint or 'not produced yet'}")
        return
    try:
        shown = path.relative_to(project)
    except ValueError:
        shown = path
    size = _human_size(path) if path.is_file() else ""
    print(f"  {'OK':4} {label:22} {shown}{f'  ({size})' if size else ''}")
    resolved = path.resolve()
    if resolved != path.absolute():
        print(f"  {'':4} {'':22}   -> {resolved}")


def _first(paths) -> Path | None:
    for path in sorted(paths):
        return path
    return None


def inventory(argv: list[str]) -> int:
    """Show what has been produced for a project and where each artefact lives."""
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} inventory",
        description=(
            "Map a project: which steps have run, where the data are (resolving symlinks "
            "into .internal), and the command to produce whatever is missing."
        ),
    )
    parser.add_argument("directory", nargs="?", default=".", help="project directory")
    parser.add_argument("--dataset", default=None, help="restrict to one dataset id")
    args = parser.parse_args(argv)

    project = Path(args.directory).expanduser().resolve()
    settings = project / "project_settings.toml"
    if not settings.is_file():
        print(f"ERROR: not a project directory (no project_settings.toml): {project}", file=sys.stderr)
        return 2

    condition = "?"
    try:
        import tomllib

        with settings.open("rb") as handle:
            conditions = tomllib.load(handle).get("conversion", {}).get("initial_conditions") or []
        condition = ", ".join(conditions) or "?"
    except Exception:
        pass

    datasets = sorted(p.name for p in (project / "warp_data").glob("*") if p.is_dir())
    if args.dataset:
        datasets = [d for d in datasets if d == args.dataset]
    print(f"project   : {project}")
    print(f"condition : {condition}")
    print(f"datasets  : {', '.join(datasets) or 'none'}")

    imod = project / "imported_data" / "imod"
    print("\nINPUT (IMOD source, symlinked)")
    _report("raw stack", _first((imod / "data").glob("*.mrc")), project)
    _report("aligned stack", _first((imod / "data").glob("*_ali.mrc")), project)
    _report("original IMOD rec", _first((imod / "reconstructions" / "native").glob("*.mrc")), project)

    for dataset in datasets:
        warp = project / "warp_data" / dataset
        runs = project / "missalignment" / "runs" / dataset
        batches = project / "batches"
        print(f"\n=== dataset {dataset} ===")

        print("IMPORT -> WARP")
        _report("Warp XML", _first((warp / "metadata").glob("*.xml")), project)
        _report("tilt stack", warp / "data" / "tiltstack", project)

        print("RECONSTRUCTION (before MissAlignment, diagnostic)")
        volumes = [p for p in (warp / "reconstructions").glob("*/*.mrc")]
        _report("volume", _first(volumes), project,
                hint=f"convertMissalignment reconstruct {project}")
        _report("preview PNG", _first((warp / "reconstructions").glob("*/*.png")), project)

        print("MISSALIGNMENT")
        _report("input snapshots", runs / "warp_snapshot_manifest.json", project,
                hint=f"convertMissalignment input --directory {project}")
        _report("smoke verdict", runs / "results" / "smoke_verdict.json", project,
                hint=f"sbatch {batches / 'missalignment' / dataset / 'run_smoke.sbatch'}")
        _report("full run result", runs / "result_manifest.json", project,
                hint=f"sbatch {batches / 'missalignment' / dataset / 'run_full.sbatch'}")

        print("RECONSTRUCTION (after MissAlignment)")
        _report("warp before/final", _first((runs / "results" / "reconstructions" / "warp_comparison").glob("*/final.mrc")),
                project, hint=f"sbatch {batches / 'missalignment' / dataset / 'compare_reconstructions.sbatch'}")
        _report("imod before", _first((runs / "results" / "reconstructions" / "before").glob("*.rec")),
                project, hint=f"sbatch {batches / 'missalignment' / dataset / 'reconstruct_before.sbatch'}")
        _report("imod final export", _first((runs / "export" / "imod").rglob("*.rec")), project,
                hint=f"sbatch {batches / 'export' / dataset / 'export_imod_and_reconstruct.sbatch'}")
    return 0


def inventory_main() -> int:
    """Console entry point for ``missalign-inventory``."""
    return inventory(sys.argv[1:])


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
    subparsers.add_parser(
        "reconstruct",
        help="reconstruct the imported Warp dataset (the step right after setup)",
        add_help=False,
    )
    subparsers.add_parser(
        "inventory",
        help="show what a project has produced and where the data are",
        add_help=False,
    )
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
    if args.command == "reconstruct":
        return reconstruct(remainder)
    if args.command == "inventory":
        return inventory(remainder)

    module_name, _, prefix = COMMANDS[args.command]
    if args.command == "setup":
        remainder = normalise_setup_arguments(remainder)
    return run_module(module_name, remainder, prefix=prefix)


if __name__ == "__main__":
    raise SystemExit(main())
