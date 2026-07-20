from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RelionExportContract:
    particle_star: str = ""
    source_snapshot: str = "selection"
    quantitative_observations: bool = True
    branch: str = "quantitative"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MInputContract:
    warp_processing_settings: str = ""
    particle_star: str = ""
    half_map_1: str = ""
    half_map_2: str = ""
    mask: str = ""
    symmetry: str = "C1"
    particle_diameter_A: float | None = None
    pixel_size_A: float | None = None
    half_set_policy: str = "preserve_input_halfsets"
    available_refinement_capabilities: dict[str, bool] = field(default_factory=dict)
    branch: str = "quantitative"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def m_contract_for_source(source_mode: str) -> MInputContract:
    motion = source_mode == "movies"
    return MInputContract(
        available_refinement_capabilities={
            "import_only": True,
            "pose_refinement": True,
            "image_warp_refinement": motion,
            "motion_refinement": motion,
            "defocus_refinement": True,
            "stage_angle_refinement": True,
            "magnification_refinement": True,
            "cs_zernike_refinement": True,
            "weight_estimation": True,
        }
    )

