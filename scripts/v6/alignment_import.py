from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .commands import CommandPlan


@dataclass
class AlignmentImportPlan:
    backend: str
    source_xf: str
    target_snapshot: str
    coordinate_method: str
    round_trip_required: bool
    commands: list[CommandPlan]

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "source_xf": self.source_xf,
            "target_snapshot": self.target_snapshot,
            "coordinate_method": self.coordinate_method,
            "round_trip_required": self.round_trip_required,
            "commands": [c.to_dict() for c in self.commands],
        }


class AlignmentImporter:
    backend = "base"

    def plan(self, *, toml: Path, project_dir: Path, source_xf: str) -> AlignmentImportPlan:
        raise NotImplementedError


class LegacyAffineAlignmentImporter(AlignmentImporter):
    backend = "legacy_affine"

    def plan(self, *, toml: Path, project_dir: Path, source_xf: str) -> AlignmentImportPlan:
        return AlignmentImportPlan(
            backend=self.backend,
            source_xf=source_xf,
            target_snapshot="alignment_initial",
            coordinate_method="v5 homogeneous Grid2D transfer with IMOD centre convention",
            round_trip_required=True,
            commands=[
                CommandPlan(
                    stage_id="20_initial_alignment_and_qc",
                    executable="python3",
                    arguments=[
                        str(Path(__file__).resolve().parent / "execute_stage.py"),
                        "--stage", "20_initial_alignment_and_qc",
                        "--settings", str(toml),
                        "--run-dir", str(project_dir),
                        "--expected-toml-hash", "<resolved-at-job-generation>",
                    ],
                    working_directory=str(project_dir),
                    description="compatibility backend reusing verified v5 affine/binning modules",
                )
            ],
        )


class WarpToolsNativeAlignmentImporter(AlignmentImporter):
    backend = "warptools_native"

    def plan(self, *, toml: Path, project_dir: Path, source_xf: str) -> AlignmentImportPlan:
        return AlignmentImportPlan(
            backend=self.backend,
            source_xf=source_xf,
            target_snapshot="alignment_initial",
            coordinate_method="WarpTools native alignment import; execution gated by syntax probe",
            round_trip_required=True,
            commands=[
                CommandPlan(
                    stage_id="20_initial_alignment_and_qc",
                    executable="WarpTools",
                    arguments=[],
                    working_directory=str(project_dir),
                    description="native WarpTools alignment import plan; exact argv requires cached WarpTools help",
                    requires_syntax_probe=True,
                )
            ],
        )


def importer_for(name: str) -> AlignmentImporter:
    if name == "legacy_affine":
        return LegacyAffineAlignmentImporter()
    if name == "warptools_native":
        return WarpToolsNativeAlignmentImporter()
    raise ValueError(f"unknown alignment backend {name!r}")
