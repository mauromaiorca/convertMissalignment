from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .commands import CommandPlan
from .config import CapabilitySet, ProjectConfig


class StagePlanningError(ValueError):
    pass


@dataclass
class StageResources:
    partition_kind: str = "gpu"
    cpus: int = 16
    time: str = "7-00:00:00"


@dataclass
class StageSpec:
    stage_id: str
    scientific_purpose: str
    input_snapshot: str
    output_snapshot: str
    required_capabilities: list[str]
    active_parameters: dict[str, Any]
    frozen_parameters: dict[str, Any]
    external_executable: str
    resources: StageResources
    expected_outputs: list[str]
    validation_function: str
    resume_policy: str
    commands: list[CommandPlan] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["commands"] = [c.to_dict() for c in self.commands]
        return data


def assert_capabilities(stage: StageSpec, capabilities: CapabilitySet) -> None:
    missing = [name for name in stage.required_capabilities if not bool(getattr(capabilities, name, False))]
    if missing:
        raise StagePlanningError(f"{stage.stage_id} requires unavailable capabilities: {missing}")


def plan_stages(config: ProjectConfig) -> list[StageSpec]:
    stages: list[StageSpec] = []
    ts = config.tilt_series[0]
    source_mode = ts.source.mode
    if source_mode == "movies":
        raise StagePlanningError("movie ingest is not implemented in this stack-only vertical pass")
    else:
        ingest_caps = ["tilt_ctf_available"]
        ingest_purpose = "WarpTools stack-only ingest planning for pre-averaged tilt images"
    stages.append(StageSpec(
        stage_id="10_warp_ingest",
        scientific_purpose=ingest_purpose,
        input_snapshot="external_source",
        output_snapshot="base",
        required_capabilities=ingest_caps,
        active_parameters={"source_mode": source_mode},
        frozen_parameters={"extra_projection_binning": ts.binning.extra_projection_binning},
        external_executable=config.software.warptools_executable,
        resources=StageResources("gpu"),
        expected_outputs=["warp/base/snapshot_manifest.json"],
        validation_function="validate_warp_project_ref",
        resume_policy="stage-result manifest and expected outputs",
    ))
    alignment_caps = ["imod_alignment_available"] if ts.warp.alignment_backend == "legacy_affine" else []
    stages.append(StageSpec(
        stage_id="20_initial_alignment_and_qc",
        scientific_purpose="IMOD alignment import and pre-MissAlignment WarpTools QC",
        input_snapshot="base",
        output_snapshot="pre_missalign",
        required_capabilities=alignment_caps,
        active_parameters={"alignment_backend": ts.warp.alignment_backend,
                           "initial_ctf": ts.preprocessing.initial_ctf},
        frozen_parameters={"condition": config.project.get("condition", "raw_xf_affine_fixed")},
        external_executable=config.software.warptools_executable,
        resources=StageResources("gpu"),
        expected_outputs=["warp/pre_missalign/snapshot_manifest.json"],
        validation_function="validate_pre_missalign_snapshot",
        resume_policy="snapshot manifest",
    ))
    for stage in stages:
        assert_capabilities(stage, ts.capabilities)
    if source_mode == "tilt_stack" and config.m.motion_refinement:
        raise StagePlanningError("M motion refinement requested for stack-only source")
    return stages


def missalignment_stage(config: ProjectConfig) -> StageSpec:
    ts = config.tilt_series[0]
    return StageSpec(
        stage_id="30_missalignment",
        scientific_purpose="Isolated MissAlignment smoke and full geometry refinement",
        input_snapshot="pre_missalign",
        output_snapshot="missalign_full",
        required_capabilities=[],
        active_parameters={"smoke_mode": ts.missalignment.smoke_mode,
                           "full_mode": ts.missalignment.full_mode},
        frozen_parameters={"full_starts_from": "pre_missalign"},
        external_executable="miss-alignment",
        resources=StageResources("gpu"),
        expected_outputs=["warp/missalign_smoke/snapshot_manifest.json",
                          "warp/missalign_full/snapshot_manifest.json",
                          "manifests/result_manifest.json"],
        validation_function="validate_deterministic_final_xml_and_xml_diff",
        resume_policy="separate smoke/full snapshot manifests",
    )
