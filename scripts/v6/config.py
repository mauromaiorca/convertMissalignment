from __future__ import annotations

import hashlib
import os
try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # local macOS system Python 3.9
    tomllib = None
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION, SOFTWARE_VERSION


class V6ConfigError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_cluster_profile(profile: str, *, path: Path | None = None) -> tuple["ClusterConfig", "SoftwareConfig", list[str]]:
    path = path or Path(__file__).resolve().parents[2] / "config" / "cluster_profiles.toml"
    notes: list[str] = []
    cluster = ClusterConfig(profile=profile)
    software = SoftwareConfig()
    if not path.is_file():
        notes.append(f"cluster profile file missing: {path}")
        return cluster, software, notes
    data = _load_toml(path)
    table = data.get(profile)
    if not isinstance(table, dict):
        notes.append(f"cluster profile {profile!r} not found in {path}")
        return cluster, software, notes
    cluster.profile = profile
    cluster.gpu_partition = table.get("gpu_partition") or table.get("partition") or cluster.gpu_partition
    cluster.gpu_constraint = table.get("gpu_constraint") or table.get("constraint") or cluster.gpu_constraint
    cluster.gres = table.get("gres") or ""
    cluster.cpu_partition = table.get("cpu_partition") or ""
    cluster.module_init_script = table.get("module_init_script") or ""
    cluster.imod_module = table.get("imod_module") or ""
    cluster.environment = table.get("missalign_environment") or table.get("environment") or ""
    software.missalignment_environment = cluster.environment
    software.missalignment_python = (str(Path(cluster.environment) / "bin" / "python")
                                     if cluster.environment else "")
    software.warptools_executable = table.get("warptools_executable") or software.warptools_executable
    software.warptools_environment = table.get("warptools_environment") or ""
    software.imod_bin_dir = table.get("imod_bin_dir") or ""
    return cluster, software, notes


@dataclass
class ClusterConfig:
    profile: str = "maxwell"
    gpu_partition: str = "vds"
    cpu_partition: str = ""
    gpu_constraint: str = "V100"
    gres: str = ""
    time_gpu: str = "7-00:00:00"
    time_cpu: str = "0-04:00:00"
    cpus: int = 16
    memory: str = ""
    account: str = ""
    qos: str = ""
    environment: str = ""
    module_init_script: str = ""
    imod_module: str = ""


@dataclass
class SoftwareConfig:
    warptools_executable: str = "WarpTools"
    mtools_executable: str = "MTools"
    mcore_executable: str = "MCore"
    missalignment_python: str = ""
    missalignment_environment: str = ""
    warptools_environment: str = ""
    imod_bin_dir: str = ""


@dataclass
class MovieSourceConfig:
    directory: str = ""
    pattern: str = ""
    mdoc: str = ""
    gain: str = ""


@dataclass
class TiltStackSourceConfig:
    path: str = ""
    tilt_file: str = ""
    mdoc: str = ""
    section_to_angle_known: bool = False
    acquisition_order_known: bool = False


@dataclass
class SourceConfig:
    mode: str = "tilt_stack"
    selected_reason: str = ""
    movies: MovieSourceConfig = field(default_factory=MovieSourceConfig)
    stack: TiltStackSourceConfig = field(default_factory=TiltStackSourceConfig)


@dataclass
class MicroscopeConfig:
    pixel_size_A: float | None = None
    voltage_kV: float | None = None
    cs_mm: float | None = None
    amplitude_contrast: float | None = None


@dataclass
class ImodConfig:
    raw_stack: str = ""
    aligned_stack: str = ""
    xf: str = ""
    tlt: str = ""
    align_com: str = ""
    raw_dimensions_xyz: list[int] = field(default_factory=list)
    aligned_dimensions_xyz: list[int] = field(default_factory=list)
    raw_pixel_size_A: float | None = None
    aligned_pixel_size_A: float | None = None
    tilt_count: int = 0
    tilt_axis_angle_deg: float | None = None
    tilt_axis_source: str = ""
    source_reconstruction: str = ""
    target_volume_dimensions_xyz: list[int] = field(default_factory=list)
    target_voxel_size_A: float | None = None
    target_physical_dimensions_A: list[float] = field(default_factory=list)
    target_geometry_source: str = ""


@dataclass
class BinningConfig:
    extra_projection_binning: int = 1


@dataclass
class WarpConfig:
    alignment_backend: str = "legacy_affine"


@dataclass
class WarpPreprocessingConfig:
    initial_ctf: str = "estimate_for_qc"
    defocus_handedness_check: str = "estimate_for_qc"
    initial_reconstruction: str = "estimate_for_qc"
    selection_qc: str = "estimate_for_qc"


@dataclass
class MissAlignmentConfig:
    enabled: bool = True
    smoke_mode: str = "smoke"
    full_mode: str = "standard"


@dataclass
class WarpPostprocessingConfig:
    ctf_after_missalignment: bool = True
    reconstruction_pixel_sizes_A: list[float] = field(default_factory=list)
    qc: str = "required"
    selection: str = "required"


@dataclass
class RelionConfig:
    enabled: bool = False
    particle_star: str = ""


@dataclass
class MConfig:
    enabled: bool = False
    import_only: bool = True
    motion_refinement: bool = False


@dataclass
class CapabilitySet:
    movies_available: bool = False
    motion_correction_available: bool = False
    frame_trajectories_available: bool = False
    gain_correction_available: bool = False
    frame_ctf_available: bool = False
    tilt_ctf_available: bool = False
    average_halves_available: bool = False
    dose_metadata_complete: bool = False
    acquisition_order_known: bool = False
    imod_alignment_available: bool = False
    motion_refinement_in_m_available: bool = False


@dataclass
class TiltSeriesConfig:
    id: str
    basename: str
    source: SourceConfig = field(default_factory=SourceConfig)
    microscope: MicroscopeConfig = field(default_factory=MicroscopeConfig)
    imod: ImodConfig = field(default_factory=ImodConfig)
    binning: BinningConfig = field(default_factory=BinningConfig)
    warp: WarpConfig = field(default_factory=WarpConfig)
    preprocessing: WarpPreprocessingConfig = field(default_factory=WarpPreprocessingConfig)
    missalignment: MissAlignmentConfig = field(default_factory=MissAlignmentConfig)
    postprocessing: WarpPostprocessingConfig = field(default_factory=WarpPostprocessingConfig)
    capabilities: CapabilitySet = field(default_factory=CapabilitySet)


@dataclass
class ProjectConfig:
    schema_version: int
    project: dict[str, Any]
    cluster: ClusterConfig
    software: SoftwareConfig
    tilt_series: list[TiltSeriesConfig]
    relion: RelionConfig = field(default_factory=RelionConfig)
    m: MConfig = field(default_factory=MConfig)
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            problems.append(f"schema_version must be {SCHEMA_VERSION}; got {self.schema_version!r}")
        if not self.tilt_series:
            problems.append("at least one [[tilt_series]] is required")
        for ts in self.tilt_series:
            if ts.source.mode not in ("movies", "tilt_stack"):
                problems.append(f"{ts.id}: source.mode must be resolved to movies or tilt_stack")
            if ts.warp.alignment_backend not in ("legacy_affine", "warptools_native"):
                problems.append(f"{ts.id}: unsupported alignment backend {ts.warp.alignment_backend!r}")
            if ts.binning.extra_projection_binning not in (1, 2, 4, 8):
                problems.append(f"{ts.id}: extra_projection_binning must be one of 1,2,4,8")
            if ts.source.mode == "tilt_stack" and self.m.motion_refinement:
                problems.append(f"{ts.id}: stack-only source cannot enable M motion refinement")
        return problems


def _table(cls, data: dict[str, Any] | None):
    data = data or {}
    names = cls().__dict__.keys()
    return cls(**{k: data.get(k, getattr(cls(), k)) for k in names})


def _source(data: dict[str, Any] | None) -> SourceConfig:
    data = data or {}
    return SourceConfig(
        mode=data.get("mode", "tilt_stack"),
        selected_reason=data.get("selected_reason", ""),
        movies=_table(MovieSourceConfig, data.get("movies")),
        stack=_table(TiltStackSourceConfig, data.get("stack")),
    )


def _tilt_series(data: dict[str, Any]) -> TiltSeriesConfig:
    return TiltSeriesConfig(
        id=data.get("id") or data.get("basename") or "TS_001",
        basename=data.get("basename") or data.get("id") or "series",
        source=_source(data.get("source")),
        microscope=_table(MicroscopeConfig, data.get("microscope")),
        imod=_table(ImodConfig, data.get("imod")),
        binning=_table(BinningConfig, data.get("binning")),
        warp=_table(WarpConfig, data.get("warp")),
        preprocessing=_table(WarpPreprocessingConfig, data.get("preprocessing")),
        missalignment=_table(MissAlignmentConfig, data.get("missalignment")),
        postprocessing=_table(WarpPostprocessingConfig, data.get("postprocessing")),
        capabilities=_table(CapabilitySet, data.get("capabilities")),
    )


def from_dict(data: dict[str, Any], *, require_v6: bool = True) -> ProjectConfig:
    if require_v6 and data.get("schema_version") != SCHEMA_VERSION:
        raise V6ConfigError(
            f"not a v6 TOML: schema_version={data.get('schema_version')!r}; "
            "use load_v5_compatibility_config() for explicit migration."
        )
    series = data.get("tilt_series") or []
    if isinstance(series, dict):
        series = [series]
    cfg = ProjectConfig(
        schema_version=int(data.get("schema_version", 0)),
        project=dict(data.get("project") or {}),
        cluster=_table(ClusterConfig, data.get("cluster")),
        software=_table(SoftwareConfig, data.get("software")),
        tilt_series=[_tilt_series(x) for x in series],
        relion=_table(RelionConfig, data.get("relion")),
        m=_table(MConfig, data.get("m")),
        provenance=dict(data.get("provenance") or {}),
    )
    problems = cfg.validate()
    if problems:
        raise V6ConfigError("invalid v6 config:\n  - " + "\n  - ".join(problems))
    return cfg


def load(path: Path) -> ProjectConfig:
    return from_dict(_load_toml(path), require_v6=True)


def load_v5_compatibility_config(path: Path) -> tuple[ProjectConfig, list[str]]:
    raw = _load_toml(path)
    inferred: list[str] = []
    if raw.get("schema_version") == SCHEMA_VERSION:
        return from_dict(raw), inferred
    project = raw.get("project", {})
    paths = raw.get("paths", {})
    inp = raw.get("input", {})
    geom = raw.get("geometry", {})
    conversion = raw.get("conversion", {})
    cluster = raw.get("cluster", {})
    ts = {
        "id": project.get("basename") or project.get("name") or "TS_001",
        "basename": project.get("basename") or project.get("name") or "series",
        "source": {
            "mode": "tilt_stack",
            "selected_reason": "explicit v5 compatibility loader inferred stack-only source",
            "stack": {"path": inp.get("raw_stack", ""), "tilt_file": inp.get("final_tilt_file", "")},
        },
        "microscope": {
            "pixel_size_A": geom.get("raw_pixel_size_A"),
        },
        "imod": {
            "raw_stack": inp.get("raw_stack", ""),
            "aligned_stack": inp.get("aligned_stack", ""),
            "xf": inp.get("final_xf_file", ""),
            "tlt": inp.get("final_tilt_file", ""),
            "align_com": inp.get("align_com", ""),
        },
        "binning": {
            "extra_projection_binning": (raw.get("multiresolution") or {}).get("extra_projection_binning", 1),
        },
        "warp": {
            "alignment_backend": "legacy_affine",
        },
        "missalignment": {
            "enabled": True,
            "full_mode": (raw.get("missalignment") or {}).get("refinement_mode", "standard"),
        },
    }
    inferred.extend([
        "source.mode=tilt_stack",
        "tilt_series.warp.alignment_backend=legacy_affine",
        "missalignment.enabled=true",
    ])
    compat = {
        "schema_version": SCHEMA_VERSION,
        "project": {
            "name": project.get("basename") or project.get("name") or "series",
            "output_dir": paths.get("output_dir", "."),
        },
        "cluster": {
            "profile": cluster.get("profile", "maxwell"),
            "gpu_partition": cluster.get("partition", "vds"),
            "gpu_constraint": cluster.get("constraint", "V100"),
            "gres": cluster.get("gres") or "",
            "environment": cluster.get("environment", ""),
            "cpu_partition": cluster.get("cpu_partition", ""),
        },
        "software": {},
        "tilt_series": [ts],
        "provenance": {"compatibility_loader": "v5", "source_toml": str(Path(path).resolve())},
    }
    return from_dict(compat), inferred


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if text in ("true", "false"):
        return text == "true"
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _fallback_load_toml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current: dict[str, Any] = root
    current_array_item: dict[str, Any] | None = None
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[[") and line.endswith("]]"):
            name = line[2:-2].strip()
            if name != "tilt_series":
                raise V6ConfigError(f"fallback TOML parser only supports [[tilt_series]], got {name}")
            item: dict[str, Any] = {}
            root.setdefault(name, []).append(item)
            current_array_item = item
            current = item
            continue
        if line.startswith("[") and line.endswith("]"):
            parts = line[1:-1].strip().split(".")
            if parts[0] == "tilt_series":
                if current_array_item is None:
                    root.setdefault("tilt_series", []).append({})
                    current_array_item = root["tilt_series"][-1]
                current = current_array_item
                for part in parts[1:]:
                    current = current.setdefault(part, {})
            else:
                current = root
                for part in parts:
                    current = current.setdefault(part, {})
            continue
        if "=" in line:
            key, value = [x.strip() for x in line.split("=", 1)]
            current[key] = _parse_scalar(value)
    return root


def _load_toml(path: Path) -> dict[str, Any]:
    if tomllib is not None:
        with Path(path).open("rb") as fh:
            return tomllib.load(fh)
    return _fallback_load_toml(path)


def to_plain(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    return obj


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(x) for x in value) + "]"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{text}"'


def write_toml(path: Path, data: dict[str, Any]) -> None:
    lines: list[str] = []
    scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
    for k, v in scalars.items():
        lines.append(f"{k} = {_toml_value(v)}")
    if scalars:
        lines.append("")

    def emit_table(prefix: str, table: dict[str, Any]) -> None:
        lines.append(f"[{prefix}]")
        nested: list[tuple[str, Any]] = []
        for key, value in table.items():
            if isinstance(value, dict):
                nested.append((key, value))
            elif value is not None:
                lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
        for key, value in nested:
            emit_table(f"{prefix}.{key}", value)

    for key in ("project", "cluster", "software", "relion", "m", "provenance"):
        if isinstance(data.get(key), dict):
            emit_table(key, data[key])
    for item in data.get("tilt_series", []):
        lines.append("[[tilt_series]]")
        nested = []
        for key, value in item.items():
            if isinstance(value, dict):
                nested.append((key, value))
            elif value is not None:
                lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
        for key, value in nested:
            emit_table(f"tilt_series.{key}", value)
    tmp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)
