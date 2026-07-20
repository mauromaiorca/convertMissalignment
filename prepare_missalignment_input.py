#!/usr/bin/env python3
"""Prepare MissAlignment input snapshots synchronously, without Slurm or CUDA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.dataset_selection import (  # noqa: E402
    DatasetSelectionError,
    discover_datasets,
    format_dataset_table,
    resolve_project_settings,
    select_dataset,
)
from pipeline.missalignment_input import prepare, layout_for  # noqa: E402
from pipeline.reconstruction_validation import (  # noqa: E402
    ReconstructionValidationError,
    record_reconstruction_validation,
)


def _project_argument(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    value = args.directory or args.project_settings
    if value is None:
        parser.error("--directory is required")
    return Path(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the selected Warp dataset for MissAlignment. The project directory "
            "is sufficient; the dataset is selected automatically when unambiguous."
        )
    )
    parser.add_argument(
        "--directory",
        type=Path,
        help="project directory containing project_settings.toml",
    )
    parser.add_argument(
        "--project-settings",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "optional dataset ID or path, for example 5.452Apx or "
            "PROJECT/warp_data/5.452Apx"
        ),
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="show all datasets, reconstruction state and acceptance state, then exit",
    )
    parser.add_argument("--force", action="store_true", help="replace existing snapshots")
    parser.add_argument(
        "--allow-without-reconstruction-acceptance",
        action="store_true",
        help="diagnostic override; not recommended for raw_xf_affine_fixed",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the complete machine-readable result instead of the concise summary",
    )
    args = parser.parse_args()
    project_arg = _project_argument(args, parser)

    try:
        settings, cfg, project_root = resolve_project_settings(project_arg)
        records, recorded_selected = discover_datasets(settings, cfg, project_root)
        if args.list_datasets:
            print(format_dataset_table(records, recorded_selected))
            return 0
        selected = select_dataset(
            records,
            project_root=project_root,
            requested=args.dataset,
            recorded_selected=recorded_selected,
        )
    except (OSError, ValueError, DatasetSelectionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        try:
            if 'records' in locals():
                print(format_dataset_table(records, recorded_selected), file=sys.stderr)
        except Exception:
            pass
        return 2

    # Alpha4 and earlier projects may contain a successful reconstruction but no
    # separate acceptance record. Backfill the new technical validation here so
    # users do not need a second command after a completed reconstruction.
    if selected.reconstruction_complete and not selected.accepted:
        try:
            validation = record_reconstruction_validation(
                layout_for(cfg, selected.dataset_id),
                level="technical",
                note=(
                    "Automatically backfilled from an existing completed Warp "
                    "reconstruction during MissAlignment input preparation."
                ),
            )
            print(
                f"[missalignment] reconstruction validation: "
                f"{validation['validation_level']} (automatic)"
            )
            records, recorded_selected = discover_datasets(settings, cfg, project_root)
            selected = select_dataset(
                records,
                project_root=project_root,
                requested=selected.dataset_id,
                recorded_selected=recorded_selected,
            )
        except ReconstructionValidationError as exc:
            print(f"ERROR: completed reconstruction could not be validated: {exc}", file=sys.stderr)
            print(format_dataset_table(records, selected.dataset_id), file=sys.stderr)
            return 2

    print(
        f"[missalignment] selected dataset: {selected.dataset_id} "
        f"({selected.origin}, status={selected.status}, "
        f"validation={selected.validation_level or 'none'})"
    )

    if not selected.source_valid:
        print(f"ERROR: selected dataset is incomplete: {selected.source_problem}", file=sys.stderr)
        print(format_dataset_table(records, selected.dataset_id), file=sys.stderr)
        return 2

    conditions = (cfg.get("conversion", {}) or {}).get("initial_conditions") or ["ali_identity"]
    condition = conditions[0] if isinstance(conditions, list) else conditions
    if (
        not selected.accepted
        and not args.allow_without_reconstruction_acceptance
        and str(condition) == "raw_xf_affine_fixed"
    ):
        print(
            "ERROR: the selected affine dataset has no successful validated Warp reconstruction.",
            file=sys.stderr,
        )
        print("Run the reconstruction batch:", file=sys.stderr)
        print(
            f"  sbatch {project_root / 'batches' / 'warp_data' / selected.dataset_id / 'reconstruct.sbatch'}",
            file=sys.stderr,
        )
        print(
            "A successful reconstruction is technically validated automatically; "
            "no separate acceptance command is required.",
            file=sys.stderr,
        )
        print(format_dataset_table(records, selected.dataset_id), file=sys.stderr)
        return 2

    try:
        result = prepare(
            settings,
            dataset_id=selected.dataset_id,
            force=args.force,
            allow_without_acceptance=args.allow_without_reconstruction_acceptance,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(format_dataset_table(records, selected.dataset_id), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        execution = result.get("execution", "prepared")
        print(f"[missalignment] input snapshots: {execution}")
        print(f"[missalignment] project: {project_root}")
        print(f"[missalignment] dataset: {selected.dataset_id}")
        batch_dir = project_root / "batches" / "missalignment" / selected.dataset_id
        print("[missalignment] next options:")
        print("  recommended safety check (optional):")
        print(f"    sbatch {batch_dir / 'run_smoke.sbatch'}")
        print("  direct full run:")
        print(f"    sbatch {batch_dir / 'run_full.sbatch'}")
        print("[missalignment] smoke testing is recommended but is not required by run_full.sbatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
