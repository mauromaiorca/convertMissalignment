#!/usr/bin/env python3
"""Version 8 project layout.

The public project tree is organised by scientific meaning rather than execution
phases. Tool-specific mutable workspaces are isolated below ``.internal``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def format_angpix(value: float) -> str:
    """Return a stable, human-readable Warp dataset identifier."""
    value = float(value)
    if value <= 0:
        raise ValueError(f"pixel size must be positive, got {value}")
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return f"{text}Apx"


def parse_angpix_id(value: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)Apx", str(value))
    if not match:
        raise ValueError(f"invalid dataset identifier {value!r}; expected e.g. 5.45Apx")
    result = float(match.group(1))
    if result <= 0:
        raise ValueError(f"invalid non-positive pixel size in {value!r}")
    return result


def dataset_id_from_config(cfg: dict[str, Any]) -> str:
    datasets = cfg.get("datasets", {}) or {}
    explicit = datasets.get("native_id") or datasets.get("selected_id")
    if explicit:
        parse_angpix_id(str(explicit))
        return str(explicit)
    geometry = cfg.get("geometry", {}) or {}
    pixel = geometry.get("raw_pixel_size_A") or geometry.get("aligned_pixel_size_A")
    if pixel in (None, 0, 0.0):
        # Legacy/unit-test fallback only. Resolved v8 projects always record native_id.
        return "native"
    return format_angpix(float(pixel))


MANIFEST_ALIASES = {
    "source_inventory.json": "source_inventory.json",
    "source_hashes.json": "source_hashes.json",
    "prepare_manifest.json": "project_prepare_manifest.json",
    "job_graph.json": "job_graph.json",
    "missalignment_run_manifest.json": "missalignment_run_manifest.json",
    "result_manifest.json": "result_manifest.json",
    "finalize_manifest.json": "export_manifest.json",
    "final_validation.json": "validation_manifest.json",
    "warp_staging_manifest.json": "warp_staging_manifest.json",
    "warp_snapshot_manifest.json": "warp_snapshot_manifest.json",
    "missalign_params.json": "missalign_params.json",
    "code_provenance.json": "code_provenance.json",
}


@dataclass(frozen=True)
class RunLayout:
    """Authoritative v8 paths for one project and one Warp pixel-size dataset."""

    run_dir: Path
    basename: str
    condition: str
    refinement_mode: str
    dataset_id: str = "native"

    @classmethod
    def from_settings(
        cls,
        *,
        out_dir: Path,
        basename: str,
        condition: str,
        refinement_mode: str,
        dataset_id: str | None = None,
        pixel_size_A: float | None = None,
    ) -> "RunLayout":
        selected = dataset_id
        if not selected and pixel_size_A not in (None, 0, 0.0):
            selected = format_angpix(float(pixel_size_A))
        selected = selected or "native"
        if selected != "native":
            parse_angpix_id(selected)
        return cls(
            run_dir=Path(out_dir),
            basename=basename,
            condition=condition,
            refinement_mode=refinement_mode,
            dataset_id=selected,
        )

    # ------------------------------------------------------------------
    # Public tree
    @property
    def provenance_dir(self) -> Path:
        return self.run_dir / "provenance"

    @property
    def imported_imod_dir(self) -> Path:
        return self.run_dir / "imported_data" / "imod"

    @property
    def warp_dataset_dir(self) -> Path:
        return self.run_dir / "warp_data" / self.dataset_id

    @property
    def warp_data_dir(self) -> Path:
        return self.warp_dataset_dir / "data"

    @property
    def warp_metadata_dir(self) -> Path:
        return self.warp_dataset_dir / "metadata"

    @property
    def warp_reconstructions_dir(self) -> Path:
        return self.warp_dataset_dir / "reconstructions"

    @property
    def missalignment_run_dir(self) -> Path:
        return self.run_dir / "missalignment" / "runs" / self.dataset_id

    @property
    def config_dir(self) -> Path:
        return self.missalignment_run_dir / "configuration"

    @property
    def config_yaml(self) -> Path:
        return self.config_dir / "config.yaml"

    @property
    def results_dir(self) -> Path:
        return self.missalignment_run_dir / "results"

    @property
    def export_dir(self) -> Path:
        return self.missalignment_run_dir / "export"

    @property
    def jobs_dir(self) -> Path:
        return self.run_dir / "batches"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    # ------------------------------------------------------------------
    # Internal tool workspaces
    @property
    def internal_dir(self) -> Path:
        return self.run_dir / ".internal"

    @property
    def internal_runtime_dir(self) -> Path:
        return self.internal_dir / "workspaces" / "runtime" / self.dataset_id

    @property
    def internal_warp_project(self) -> Path:
        return self.internal_runtime_dir / "warp" / f"warp_{self.condition}"

    @property
    def training_dir(self) -> Path:
        # Hidden compatibility entry within the dataset. It points to the mutable
        # Warp project below .internal; data/metadata are published separately.
        return self.warp_dataset_dir / ".warp_project"

    @property
    def snapshot_root(self) -> Path:
        return self.internal_dir / "workspaces" / "missalignment" / self.dataset_id

    @property
    def pre_missalign_dir(self) -> Path:
        return self.snapshot_root / "before"

    @property
    def smoke_warp_dir(self) -> Path:
        return self.snapshot_root / "smoke"

    @property
    def full_warp_dir(self) -> Path:
        return self.snapshot_root / "full"

    @property
    def diagnostics_dir(self) -> Path:
        return self.internal_dir / "diagnostics"

    @property
    def attempts_dir(self) -> Path:
        return self.internal_dir / "attempts"

    @property
    def helpers_dir(self) -> Path:
        return self.internal_dir / "helpers"

    @property
    def state_dir(self) -> Path:
        return self.internal_dir / "state"

    # ------------------------------------------------------------------
    def create(self) -> "RunLayout":
        directories = (
            self.provenance_dir,
            self.imported_imod_dir / "data",
            self.imported_imod_dir / "configuration",
            self.imported_imod_dir / "reconstructions",
            self.warp_data_dir,
            self.warp_metadata_dir,
            self.warp_reconstructions_dir,
            self.config_dir,
            self.missalignment_run_dir / "input",
            self.missalignment_run_dir / "checkpoints",
            self.results_dir / "parameters",
            self.results_dir / "transforms",
            self.results_dir / "reconstructions",
            self.results_dir / "validation",
            self.export_dir / "imod",
            self.export_dir / "warp",
            self.export_dir / "relion",
            self.export_dir / "m",
            self.jobs_dir / "import",
            self.jobs_dir / "warp_data" / self.dataset_id,
            self.jobs_dir / "missalignment" / self.dataset_id,
            self.jobs_dir / "export" / self.dataset_id,
            self.logs_dir / "import",
            self.logs_dir / "warp_data" / self.dataset_id,
            self.logs_dir / "missalignment" / self.dataset_id,
            self.logs_dir / "export" / self.dataset_id,
            self.logs_dir / "commands",
            self.logs_dir / "environment",
            self.logs_dir / "resources",
            self.internal_runtime_dir,
            self.internal_warp_project,
            self.snapshot_root,
            self.diagnostics_dir / "preflight",
            self.diagnostics_dir / "geometry",
            self.diagnostics_dir / "gradients",
            self.diagnostics_dir / "checkpoints",
            self.diagnostics_dir / "postmortem",
            self.diagnostics_dir / "ctf",
            self.diagnostics_dir / "warp",
            self.attempts_dir / "reconstruction" / self.dataset_id,
            self.helpers_dir,
            self.state_dir,
            self.internal_dir / "debug_bundles",
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        if self.training_dir.exists() and not self.training_dir.is_symlink():
            # An existing real directory is valid for migrated projects.
            pass
        elif not self.training_dir.exists():
            target = self.internal_warp_project
            relative = Path("..", "..", ".internal", "workspaces", "runtime", self.dataset_id,
                            "warp", f"warp_{self.condition}")
            self.training_dir.symlink_to(relative, target_is_directory=True)
        return self

    def manifest(self, name: str) -> Path:
        """Return the authoritative path for a project- or dataset-level manifest.

        Project discovery/import manifests live in ``provenance``. Manifests that
        describe one MissAlignment dataset/run live beside that run so multiple
        pixel-size datasets cannot overwrite one another.
        """
        try:
            public_name = MANIFEST_ALIASES[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown manifest {name!r}; known: {tuple(MANIFEST_ALIASES)}"
            ) from exc
        run_level = {
            "missalignment_run_manifest.json",
            "result_manifest.json",
            "finalize_manifest.json",
            "final_validation.json",
            "warp_snapshot_manifest.json",
            "missalign_params.json",
        }
        if name in run_level:
            return self.missalignment_run_dir / public_name
        return self.provenance_dir / public_name

    def job(self, name: str) -> Path:
        return self.jobs_dir / name

    def batch_path(self, category: str, name: str) -> Path:
        if category == "import":
            return self.jobs_dir / "import" / name
        if category in {"warp_data", "missalignment", "export"}:
            return self.jobs_dir / category / self.dataset_id / name
        raise ValueError(f"unknown batch category: {category}")

    def log_dir(self, category: str) -> Path:
        if category == "import":
            return self.logs_dir / "import"
        if category in {"warp_data", "missalignment", "export"}:
            return self.logs_dir / category / self.dataset_id
        raise ValueError(f"unknown log category: {category}")

    @property
    def acceptance_path(self) -> Path:
        return self.warp_reconstructions_dir / "acceptance.json"

    @property
    def dataset_manifest(self) -> Path:
        return self.warp_dataset_dir / "manifest.json"

    @property
    def dataset_config(self) -> Path:
        return self.warp_dataset_dir / "dataset.toml"

    @property
    def final_transforms(self) -> Path:
        return self.results_dir / "transforms"

    @property
    def final_aligned(self) -> Path:
        return self.results_dir / "parameters" / "aligned"

    @property
    def final_ctf(self) -> Path:
        return self.results_dir / "parameters" / "ctf"

    @property
    def final_reconstruction(self) -> Path:
        return self.results_dir / "reconstructions" / "final"

    def to_dict(self) -> dict[str, str]:
        return {
            "layout_version": "8",
            "project_root": str(self.run_dir),
            "basename": self.basename,
            "condition": self.condition,
            "refinement_mode": self.refinement_mode,
            "dataset_id": self.dataset_id,
            "imported_imod": str(self.imported_imod_dir),
            "warp_dataset": str(self.warp_dataset_dir),
            "training_dir": str(self.training_dir),
            "missalignment_run": str(self.missalignment_run_dir),
            "batches": str(self.jobs_dir),
            "logs": str(self.logs_dir),
            "provenance": str(self.provenance_dir),
            "internal": str(self.internal_dir),
        }
