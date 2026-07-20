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
        "prepare MissAlignment input snapshots (before/smoke/full)",
        (),
    ),
    "preprocess": (
        "warp_preprocess",
        "create a lower-resolution Warp dataset",
        (),
    ),
    "export": (
        "export_warp_to_imod",
        "export the refined alignment back to IMOD",
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


PROJECT_MARKER = "project_settings.toml"

CONDITION_NOTES = {
    "raw_xf_affine_fixed": "full per-tilt affine from the IMOD .xf (quarter-turn geometry)",
    "raw_xf_translation": "per-tilt shifts only; the tilt axis is left to MissAlignment",
    "raw_identity": "raw stack, no alignment applied",
    "ali_identity": "pre-aligned stack, no further alignment",
    "raw_xf": "per-tilt shifts from the IMOD .xf (legacy name)",
}


def _human_size(path: Path) -> str:
    try:
        size = float(path.stat().st_size)
    except OSError:
        return ""
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return ""


def _import_alignment(warp_dir: Path) -> str:
    """The alignment the import itself applied (from the conversion manifest)."""
    manifest = _first((warp_dir / "metadata").glob("*.conversion.json"))
    if manifest is None:
        return ""
    try:
        import json

        data = json.loads(manifest.read_text())
        return str(data.get("alignment_mode") or "")
    except Exception:
        return ""


def _declared_purpose(volume: Path | None) -> str:
    """The purpose the engine itself recorded next to a published reconstruction."""
    if volume is None:
        return ""
    manifest = volume.parent / "manifest.json"
    if not manifest.is_file():
        return ""
    try:
        import json

        return str(json.loads(manifest.read_text()).get("purpose") or "")
    except Exception:
        return ""


def _origin(path: Path) -> str:
    """Which attempt and output stage produced a published artefact."""
    parts = path.resolve().parts
    attempt = next((p for p in parts if p.startswith("attempt_")), "")
    stage = next((p for p in parts if p.startswith("output_")), "")
    if not attempt:
        return ""
    return f"{attempt}" + (f"  ({stage})" if stage else "")


def _short(path: Path) -> str:
    """Path relative to the working directory when that is shorter."""
    try:
        relative = path.resolve().relative_to(Path.cwd().resolve())
        return str(relative) or "."
    except ValueError:
        return str(path)


def _first(paths) -> Path | None:
    """The first match of a glob, or None."""
    for path in sorted(paths):
        return path
    return None


def find_projects(directory) -> list[Path]:
    """The project in ``directory``, or the projects directly inside it."""
    directory = Path(directory).expanduser()
    if (directory / PROJECT_MARKER).is_file():
        return [directory.resolve()]
    return sorted({p.parent.resolve() for p in directory.glob("*/" + PROJECT_MARKER)})


def project_condition(project: Path) -> str:
    try:
        import tomllib

        with (project / PROJECT_MARKER).open("rb") as handle:
            data = tomllib.load(handle)
        return ", ".join(data.get("conversion", {}).get("initial_conditions") or []) or "?"
    except Exception:
        return "?"


def project_datasets(project: Path) -> list[str]:
    return sorted(p.name for p in (project / "warp_data").glob("*") if p.is_dir())


def choose_project(directory, command: str) -> Path | None:
    """Resolve one project, or explain what to type when the choice is not obvious."""
    projects = find_projects(directory)
    if len(projects) == 1:
        return projects[0]
    where = Path(directory).expanduser().resolve()
    if not projects:
        print(f"No project found in {where}", file=sys.stderr)
        print("A project is a directory containing project_settings.toml.", file=sys.stderr)
        print("Create one with:  convertMissalignment setup --data-dir DATA --out-dir PROJECT",
              file=sys.stderr)
        return None
    print(f"{len(projects)} projects found in {where}. Pick one:\n")
    for item in projects:
        datasets = ", ".join(project_datasets(item)) or "no dataset yet"
        print(f"  {item.name}")
        print(f"      condition {project_condition(item)}   datasets {datasets}")
        print(f"      convertMissalignment {command} {item.name}\n")
    return None


def _stage(number: int, title: str, done: bool) -> None:
    state = "done" if done else "MISSING"
    print(f"\n{number}  {title:<52}{state:>8}")


def _detail(text: str) -> None:
    print(f"   {text}")


def inventory(argv: list[str]) -> int:
    """Show, stage by stage, what a project has produced and where it is."""
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} inventory",
        description="Show what a project has produced, where the data are, and what to run next.",
    )
    parser.add_argument("directory", nargs="?", default=".", help="project directory (default: .)")
    parser.add_argument("--dataset", default=None, help="restrict to one dataset id")
    parser.add_argument("--full-paths", action="store_true",
                        help="also print the real target of each symlink")
    args = parser.parse_args(argv)

    project = choose_project(args.directory, "inventory")
    if project is None:
        return 2

    condition = project_condition(project)
    note = CONDITION_NOTES.get(condition, "")
    datasets = project_datasets(project)
    if args.dataset:
        datasets = [d for d in datasets if d == args.dataset]

    print(f"Project  {project.name}")
    print(f"Path     {project}")
    print(f"Model    {condition}" + (f"\n         {note}" if note else ""))
    print(f"Dataset  {', '.join(datasets) or 'none yet'}")

    def show(path: Path | None, *, origin: bool = False) -> None:
        if path is None or not path.exists():
            return
        try:
            shown = path.relative_to(project)
        except ValueError:
            shown = path
        size = _human_size(path) if path.is_file() else ""
        _detail(f"{shown}" + (f"   ({size})" if size else ""))
        if origin:
            source = _origin(path)
            if source:
                _detail(f"from:  {source}")
        if args.full_paths and path.resolve() != path.absolute():
            _detail(f"real:  {path.resolve()}")

    for dataset in datasets:
        warp = project / "warp_data" / dataset
        runs = project / "missalignment" / "runs" / dataset
        if len(datasets) > 1:
            print(f"\n--- dataset {dataset} ---")

        xml = _first((warp / "metadata").glob("*.xml"))
        _stage(1, "IMPORT", xml is not None)
        _detail("IMOD geometry converted into a Warp dataset.")
        show(xml)

        volume = _first((warp / "reconstructions").glob("*/*.mrc"))
        _stage(2, "RECONSTRUCTION AT IMPORT (pre-MissAlignment)", volume is not None)
        applied = _import_alignment(warp) or "the imported alignment"
        _detail(f"ALREADY ALIGNED at import by the IMOD .xf ({applied}).")
        _detail("\"Before\" means before MissAlignment REFINES that alignment,")
        _detail("not an unaligned raw volume.")
        purpose = _declared_purpose(volume)
        if purpose:
            _detail(f"engine: {purpose}")
        show(volume, origin=True)
        show(_first((warp / "reconstructions").glob("*/*.png")))
        if volume is not None:
            _detail(f"view:  3dmod warp_data/{dataset}/reconstructions/*/*.mrc")
        if volume is None:
            _detail("run:   convertMissalignment reconstruct")

        snapshots = runs / "warp_snapshot_manifest.json"
        _stage(3, "MISSALIGNMENT INPUT", snapshots.exists())
        _detail("Isolated before/smoke/full copies of the Warp project.")
        show(snapshots if snapshots.exists() else None)
        snapdir = project / ".internal" / "workspaces" / "missalignment" / dataset
        if snapdir.is_dir():
            kept = sorted(d.name for d in snapdir.glob("*") if d.is_dir())
            _detail(f"copies: .internal/workspaces/missalignment/{dataset}/"
                    + (f"{{{','.join(kept)}}}" if kept else ""))
        if not snapshots.exists():
            _detail("run:   convertMissalignment input --directory .")

        smoke = runs / "results" / "smoke_verdict.json"
        _stage(4, "SMOKE RUN", smoke.exists())
        _detail("Short safety check; optional but recommended.")
        if not smoke.exists():
            _detail(f"run:   sbatch batches/missalignment/{dataset}/run_smoke.sbatch")

        full = runs / "result_manifest.json"
        _stage(5, "FULL RUN", full.exists())
        _detail("The MissAlignment refinement itself.")
        show(full if full.exists() else None)
        if not full.exists():
            _detail(f"run:   sbatch batches/missalignment/{dataset}/run_full.sbatch")

        pair = _first((runs / "results" / "reconstructions" / "warp_comparison").glob("*/final.mrc"))
        _stage(6, "RECONSTRUCTION AFTER MISSALIGNMENT", pair is not None)
        _detail("before.mrc and final.mrc built together, same Warp parameters.")
        show(pair, origin=True)
        if pair is not None:
            _detail(f"view:  3dmod missalignment/runs/{dataset}/results/"
                    "reconstructions/warp_comparison/*/*.mrc")
        if pair is None:
            _detail(f"run:   sbatch batches/missalignment/{dataset}/compare_reconstructions.sbatch")

        exported = _first((runs / "export" / "imod").rglob("*.rec"))
        _stage(7, "EXPORT TO IMOD", exported is not None)
        _detail("Refined alignment written back as IMOD .xf plus a reconstruction.")
        show(exported, origin=True)
        if exported is not None:
            _detail(f"view:  3dmod missalignment/runs/{dataset}/export/imod/**/*.rec")
        if exported is None:
            _detail(f"run:   sbatch batches/export/{dataset}/export_imod_and_reconstruct.sbatch")

    print("\nTOMOGRAMS  (which volume is which)")
    for dataset in datasets:
        warp = project / "warp_data" / dataset
        runs = project / "missalignment" / "runs" / dataset
        before = _first((warp / "reconstructions").glob("*/*.mrc"))
        after = _first((runs / "results" / "reconstructions" / "warp_comparison").glob("*/final.mrc"))
        applied = _import_alignment(warp) or "imported alignment"
        _detail(f"BEFORE MissAlignment = imported data already aligned by the")
        _detail(f"  IMOD .xf ({applied}); MissAlignment has not refined it yet")
        if before is not None:
            _detail(f"  {before.relative_to(project)}")
        else:
            _detail("  not produced yet -> convertMissalignment reconstruct")
        _detail("AFTER MissAlignment  = the same data once MissAlignment has")
        _detail("  refined that alignment")
        if after is not None:
            _detail(f"  {after.relative_to(project)}")
        else:
            _detail("  NOT PRODUCED. The full run refines the alignment but does not")
            _detail("  reconstruct a volume. To get before+final with identical")
            _detail("  parameters, run:")
            _detail(f"  sbatch batches/missalignment/{dataset}/compare_reconstructions.sbatch")

    print("\nPROJECT RECORD")
    records = [
        ("prepare manifest", project / "provenance" / "project_prepare_manifest.json"),
        ("code provenance", project / "provenance" / "code_provenance.json"),
    ]
    for dataset in datasets:
        runs = project / "missalignment" / "runs" / dataset
        records += [
            ("missalignment run", runs / "missalignment_run_manifest.json"),
            ("result manifest", runs / "result_manifest.json"),
            ("finalize manifest", runs / "finalize_manifest.json"),
        ]
    for label, path in records:
        _detail(f"{label:22}{'present' if path.exists() else '-'}")
    events = project / "logs" / "events.jsonl"
    if events.is_file():
        lines = [ln for ln in events.read_text().splitlines() if ln.strip()]
        last = ""
        if lines:
            try:
                import json

                last = json.loads(lines[-1]).get("event", "")
            except Exception:
                last = ""
        _detail(f"{'events':22}{len(lines)}" + (f"  (last: {last})" if last else ""))

    print("\nUse --full-paths to resolve symlinks into .internal/")
    return 0


def inventory_main() -> int:
    """Console entry point for ``missalign-inventory``."""
    return inventory(sys.argv[1:])


def export_guide(argv: list[str]) -> int:
    """Explain exactly what to export and how, then hand over to the engine."""
    if argv:                                   # explicit arguments: unchanged behaviour
        return run_module("export_warp_to_imod", argv)
    project = choose_project(".", "export")
    if project is None:
        return 2
    settings = project / PROJECT_MARKER
    datasets = project_datasets(project)
    print(f"Project  {project.name}")
    print(f"Path     {project}")
    print(f"Model    {project_condition(project)}")
    print("\nExport writes the refined alignment back to IMOD (.xf plus a reconstruction).")
    print("It needs a completed full MissAlignment run.\n")
    print("Run one of:\n")
    print(f"  convertMissalignment export finalize {_short(settings)}")
    for dataset in datasets or ["<dataset>"]:
        batch = project / "batches" / "export" / dataset / "export_imod_and_reconstruct.sbatch"
        print(f"  sbatch {_short(batch)}")
    print("\nNot sure what is ready? Run:  convertMissalignment inventory")
    return 0


def info() -> int:
    """Installation locations and environment check, in one place."""
    print("INSTALLATION")
    print(f"  distribution {DISTRIBUTION}  {distribution_version()}")
    print(f"  pipeline     {pipeline_version()}")
    print(f"  source root  {APPLICATION_ROOT}")
    print(f"  executable   {command_path() or 'not found on PATH'}")
    print(f"  python       {Path(sys.executable).resolve()}")
    print(f"  environment  {Path(sys.prefix).resolve()}")
    print()
    return doctor()


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
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.add_parser("version", help="show the installed package version")
    subparsers.add_parser(
        "info", help="installation locations plus an environment check")
    subparsers.add_parser("inventory", help="show what a project produced and what to run next",
                          add_help=False)
    # accepted for backward compatibility; omitting help keeps them out of the listing
    subparsers.add_parser("where")
    subparsers.add_parser("doctor")
    subparsers.add_parser("prepare-input", add_help=False)
    subparsers.add_parser("status", add_help=False)
    subparsers.add_parser(
        "reconstruct",
        help="reconstruct the imported Warp dataset (the step right after setup)",
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
    if args.command in ("info", "where", "doctor"):
        return info()
    if args.command == "reconstruct":
        return reconstruct(remainder)
    if args.command == "inventory":
        return inventory(remainder)

    if args.command == "export":
        return export_guide(remainder)
    if args.command == "status":                 # folded into inventory
        print("note: 'status' is now part of 'inventory'.\n", file=sys.stderr)
        return inventory([a for a in remainder if not a.startswith("-")][:1])
    if args.command == "prepare-input":          # historical alias
        args.command = "input"

    module_name, _, prefix = COMMANDS[args.command]
    if args.command == "setup":
        remainder = normalise_setup_arguments(remainder)
    return run_module(module_name, remainder, prefix=prefix)


if __name__ == "__main__":
    raise SystemExit(main())
