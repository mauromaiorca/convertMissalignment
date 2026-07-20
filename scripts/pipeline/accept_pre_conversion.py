#!/usr/bin/env python3
"""Optionally record visual review of a v8 Warp reconstruction.

Successful Warp reconstructions are technically validated automatically. This
command is retained only when a scientist wants provenance of an explicit visual
inspection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .dataset_selection import (
        DatasetSelectionError,
        discover_datasets,
        format_dataset_table,
        resolve_project_settings,
        select_dataset,
    )
    from .reconstruction_validation import (
        ReconstructionValidationError,
        record_reconstruction_validation,
    )
    from .warptools_reconstruction import layout_for
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.dataset_selection import (
        DatasetSelectionError,
        discover_datasets,
        format_dataset_table,
        resolve_project_settings,
        select_dataset,
    )
    from pipeline.reconstruction_validation import (
        ReconstructionValidationError,
        record_reconstruction_validation,
    )
    from pipeline.warptools_reconstruction import layout_for


def _project_argument(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    value = args.directory or args.project_settings
    if value is None:
        parser.error("--directory is required")
    return Path(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", type=Path, help="project directory containing project_settings.toml")
    ap.add_argument("--project-settings", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--note", default="manual visual review")
    ap.add_argument("--dataset", default=None, help="optional dataset ID or warp_data directory")
    ap.add_argument("--list-datasets", action="store_true")
    args = ap.parse_args()

    try:
        settings, cfg, project_root = resolve_project_settings(_project_argument(args, ap))
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
        layout = layout_for(cfg, selected.dataset_id)
        validation = record_reconstruction_validation(
            layout,
            level="visual",
            note=args.note,
        )
    except (OSError, ValueError, DatasetSelectionError, ReconstructionValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if "records" in locals():
            print(format_dataset_table(records, recorded_selected), file=sys.stderr)
        return 2

    print(f"Visual review recorded for dataset: {layout.dataset_id}")
    print(f"Reconstruction: {validation['reconstruction']}")
    print(f"Validation record: {layout.acceptance_path}")
    print("This step is optional; technical validation was already recorded by the reconstruction job.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
