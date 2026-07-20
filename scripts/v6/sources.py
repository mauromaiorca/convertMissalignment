from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import CapabilitySet
from .mrc import MrcHeader, validate_stack


class SourceDiscoveryError(ValueError):
    pass


MOVIE_SUFFIXES = (".eer", ".tif", ".tiff", ".mrc", ".mrcs")


@dataclass
class TiltIdentity:
    tilt_id: str
    acquisition_index: int | None
    angle_sorted_index: int | None
    tilt_angle_deg: float | None
    movie_path: str | None = None
    mdoc_section: str | None = None
    dose: float | None = None
    corresponding_imod_stack_section: int | None = None
    corresponding_warp_average: str | None = None


@dataclass
class SourceDiscoveryResult:
    mode: str
    selected_reason: str
    movies: list[str] = field(default_factory=list)
    stack_path: str = ""
    tilt_file: str = ""
    mdoc: str = ""
    gain: str = ""
    raw_stack: str = ""
    aligned_stack: str = ""
    final_xf: str = ""
    align_com: str = ""
    source_reconstruction: str = ""
    raw_header: dict[str, Any] = field(default_factory=dict)
    aligned_header: dict[str, Any] = field(default_factory=dict)
    target_geometry: dict[str, Any] = field(default_factory=dict)
    tilt_axis_angle_deg: float | None = None
    tilt_axis_source: str = ""
    identity_table: list[TiltIdentity] = field(default_factory=list)
    capabilities: CapabilitySet = field(default_factory=CapabilitySet)
    observations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = asdict(self.capabilities)
        return data


def _read_tilts(path: Path | None) -> list[float]:
    if not path or not path.is_file():
        return []
    values = []
    for line in path.read_text(errors="ignore").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            values.append(float(text.split()[0]))
        except ValueError:
            continue
    return values


def _mdoc_sections(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.is_file():
        return []
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in path.read_text(errors="ignore").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("[") and text.endswith("]"):
            if current:
                sections.append(current)
            current = {"section": text}
            continue
        if "=" in text and current is not None:
            key, value = [x.strip() for x in text.split("=", 1)]
            current[key] = value
    if current:
        sections.append(current)
    return sections


def _float_field(section: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in section:
            try:
                return float(str(section[name]).split()[0])
            except ValueError:
                return None
    return None


def _movie_name(section: dict[str, Any]) -> str:
    for key in ("SubFramePath", "MoviePath", "FramePath"):
        if section.get(key):
            return Path(str(section[key]).replace("\\", "/")).name
    return ""


class MovieSourceAdapter:
    def discover(self, data_dir: Path, basename: str) -> SourceDiscoveryResult:
        data_dir = Path(data_dir)
        movies = sorted(
            p for p in data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in MOVIE_SUFFIXES and "gain" not in p.name.lower()
        )
        mdoc = self._find(data_dir, basename, (".mdoc",))
        gain = self._find_gain(data_dir)
        sections = _mdoc_sections(mdoc)
        if not movies or not mdoc or not sections:
            raise SourceDiscoveryError("movie source is incomplete: need movie files and MDOC sections")
        by_name = {p.name: p for p in movies}
        table: list[TiltIdentity] = []
        missing: list[str] = []
        for index, section in enumerate(sections):
            name = _movie_name(section)
            movie = by_name.get(name) if name else None
            if not movie:
                missing.append(name or f"section-{index}")
                continue
            angle = _float_field(section, "TiltAngle", "TiltAngleDeg")
            dose = _float_field(section, "ExposureDose", "DoseRate", "FrameDosesAndNumber")
            table.append(TiltIdentity(
                tilt_id=f"tilt_{index:04d}",
                acquisition_index=index,
                angle_sorted_index=None,
                tilt_angle_deg=angle,
                movie_path=str(movie.resolve()),
                mdoc_section=section.get("section"),
                dose=dose,
                corresponding_imod_stack_section=index,
            ))
        if missing:
            raise SourceDiscoveryError(f"MDOC references movies that are absent or ambiguous: {missing[:5]}")
        angles = [x.tilt_angle_deg for x in table]
        if any(x is None for x in angles):
            raise SourceDiscoveryError("MDOC movie source lacks complete tilt angles")
        order = sorted(range(len(table)), key=lambda i: float(table[i].tilt_angle_deg))
        for sorted_index, original_index in enumerate(order):
            table[original_index].angle_sorted_index = sorted_index
        caps = CapabilitySet(
            movies_available=True,
            motion_correction_available=True,
            frame_trajectories_available=True,
            gain_correction_available=bool(gain),
            frame_ctf_available=True,
            tilt_ctf_available=True,
            average_halves_available=True,
            dose_metadata_complete=all(x.dose is not None for x in table),
            acquisition_order_known=True,
            imod_alignment_available=False,
            motion_refinement_in_m_available=True,
        )
        return SourceDiscoveryResult(
            mode="movies",
            selected_reason="complete and internally consistent movie set selected as quantitative source",
            movies=[str(p.resolve()) for p in movies],
            mdoc=str(mdoc.resolve()),
            gain=str(gain.resolve()) if gain else "",
            identity_table=table,
            capabilities=caps,
            observations={"movie_count": len(movies), "mdoc_sections": len(sections)},
        )

    @staticmethod
    def _find(data_dir: Path, basename: str, suffixes: tuple[str, ...]) -> Path | None:
        exact = [data_dir / f"{basename}{suffix}" for suffix in suffixes]
        for path in exact:
            if path.is_file():
                return path
        matches = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _find_gain(data_dir: Path) -> Path | None:
        matches = [p for p in data_dir.rglob("*") if p.is_file() and "gain" in p.name.lower()]
        return matches[0] if len(matches) == 1 else None


class TiltStackSourceAdapter:
    def discover(self, data_dir: Path, basename: str, *, condition: str = "raw_xf_affine_fixed") -> SourceDiscoveryResult:
        data_dir = Path(data_dir)
        inv = self._discover_v5(data_dir, basename)
        stack = Path(inv.raw_stack) if inv.raw_stack else self._find_stack(data_dir, basename)
        tilt = Path(inv.tilt_file) if inv.tilt_file else self._find_tilt(data_dir, basename)
        if not stack:
            raise SourceDiscoveryError("no raw tilt stack found")
        if not tilt:
            raise SourceDiscoveryError("no tilt-angle file found")
        angles = _read_tilts(tilt)
        if not angles:
            raise SourceDiscoveryError(f"no tilt angles parsed from {tilt}")
        try:
            raw_header = validate_stack(stack, expected_tilts=len(angles))
        except Exception as exc:
            raise SourceDiscoveryError(str(exc)) from exc
        aligned_header = None
        if inv.aligned_stack:
            try:
                aligned_header = validate_stack(Path(inv.aligned_stack), expected_tilts=len(angles))
            except Exception as exc:
                raise SourceDiscoveryError(str(exc)) from exc
        if condition == "raw_xf_affine_fixed":
            if not inv.final_xf:
                raise SourceDiscoveryError("raw_xf_affine_fixed requires an unambiguous final .xf")
            xf_rows = _count_nonempty(Path(inv.final_xf))
            if xf_rows != len(angles):
                raise SourceDiscoveryError(f"final .xf rows {xf_rows} != tilt-angle count {len(angles)}")
        table = [
            TiltIdentity(
                tilt_id=f"tilt_{i:04d}",
                acquisition_index=i,
                angle_sorted_index=j,
                tilt_angle_deg=angles[i],
                corresponding_imod_stack_section=i,
            )
            for j, i in enumerate(sorted(range(len(angles)), key=lambda idx: angles[idx]))
        ] if angles else []
        caps = CapabilitySet(
            movies_available=False,
            motion_correction_available=False,
            frame_trajectories_available=False,
            gain_correction_available=False,
            frame_ctf_available=False,
            tilt_ctf_available=True,
            average_halves_available=False,
            dose_metadata_complete=False,
            acquisition_order_known=bool(inv.mdoc_file),
            imod_alignment_available=bool(inv.final_xf and tilt and stack),
            motion_refinement_in_m_available=False,
        )
        target = _target_geometry(data_dir, basename, inv, raw_header, aligned_header)
        tilt_axis, tilt_axis_source = _tilt_axis(data_dir, basename, inv)
        return SourceDiscoveryResult(
            mode="tilt_stack",
            selected_reason="stack-only quantitative source selected; movie motion capabilities unavailable",
            stack_path=str(stack.resolve()),
            tilt_file=str(tilt.resolve()) if tilt else "",
            mdoc=str(Path(inv.mdoc_file).resolve()) if inv.mdoc_file else "",
            raw_stack=str(stack.resolve()),
            aligned_stack=str(Path(inv.aligned_stack).resolve()) if inv.aligned_stack else "",
            final_xf=str(Path(inv.final_xf).resolve()) if inv.final_xf else "",
            align_com=str(Path(inv.tilt_com).resolve()) if inv.tilt_com else "",
            source_reconstruction=str(Path(inv.source_reconstruction).resolve()) if inv.source_reconstruction else "",
            raw_header=raw_header.to_dict(),
            aligned_header=aligned_header.to_dict() if aligned_header else {},
            target_geometry=target,
            tilt_axis_angle_deg=tilt_axis,
            tilt_axis_source=tilt_axis_source,
            identity_table=table,
            capabilities=caps,
            observations={"source_was_preaveraged": True, "frame_count_per_tilt": 1,
                          "motion_model_available": False, "tilt_count": len(angles) or None,
                          "section_to_angle_mapping_known": True,
                          "acquisition_order_known": bool(inv.mdoc_file)},
        )

    @staticmethod
    def _discover_v5(data_dir: Path, basename: str):
        import sys
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from pipeline import discovery as DISC
        try:
            return DISC.discover_sources(data_dir, basename)
        except Exception as exc:
            raise SourceDiscoveryError(str(exc)) from exc

    @staticmethod
    def _find_stack(data_dir: Path, basename: str) -> Path | None:
        names = [f"{basename}.st", f"{basename}.mrc", f"{basename}_raw.mrc"]
        for name in names:
            p = data_dir / name
            if p.is_file():
                return p
        matches = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".st", ".mrc")]
        matches = [p for p in matches if not re.search(r"(_ali|_rec|gain)", p.name, re.I)]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _find_tilt(data_dir: Path, basename: str) -> Path | None:
        for name in (f"{basename}.tlt", f"{basename}.rawtlt"):
            p = data_dir / name
            if p.is_file():
                return p
        matches = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".tlt", ".rawtlt")]
        return matches[0] if len(matches) == 1 else None


def resolve_source(data_dir: Path, basename: str, source_mode: str, *, condition: str = "raw_xf_affine_fixed") -> SourceDiscoveryResult:
    if source_mode not in ("auto", "movies", "tilt_stack"):
        raise SourceDiscoveryError(f"unsupported source mode {source_mode!r}")
    movie_adapter = MovieSourceAdapter()
    stack_adapter = TiltStackSourceAdapter()
    if source_mode == "movies":
        return movie_adapter.discover(data_dir, basename)
    if source_mode == "tilt_stack":
        return stack_adapter.discover(data_dir, basename, condition=condition)
    try:
        return movie_adapter.discover(data_dir, basename)
    except SourceDiscoveryError as movie_error:
        stack = stack_adapter.discover(data_dir, basename, condition=condition)
        stack.selected_reason = (
            "source-mode auto fell back to tilt_stack because movie source was incomplete: "
            f"{movie_error}"
        )
        return stack


def _count_nonempty(path: Path) -> int:
    return sum(1 for line in Path(path).read_text().splitlines() if line.strip())


def _tilt_axis(data_dir: Path, basename: str, inv) -> tuple[float | None, str]:
    if not inv.tilt_com:
        return None, ""
    try:
        import importlib.util
        import sys
        p = Path(__file__).resolve().parents[1] / "01_extract_etomo_params.py"
        spec = importlib.util.spec_from_file_location("extract01_for_v6", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        value, source = mod.parse_tilt_axis_angle(
            Path(inv.tilt_com).parent,
            Path(data_dir),
            basename,
            Path(inv.mdoc_file) if inv.mdoc_file else None,
        )
        return (float(value), source) if value is not None else (None, "")
    except BaseException as exc:
        return None, f"unresolved: {exc}"


def _target_geometry(data_dir: Path, basename: str, inv, raw_header: MrcHeader, aligned_header: MrcHeader | None) -> dict:
    try:
        import sys
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from pipeline import imod_geometry as IG
        target = IG.resolve_target_geometry(
            reconstruction_path=inv.source_reconstruction,
            tilt_com_path=inv.tilt_com,
            newst_com_path=inv.newst_com,
            imod_dir=str(Path(inv.tilt_com).parent) if inv.tilt_com else str(data_dir),
            mdoc_path=inv.mdoc_file,
            aligned_shape_xyz=(aligned_header.shape_xyz if aligned_header else raw_header.shape_xyz),
            aligned_pixel_A=(aligned_header.pixel_size_A if aligned_header else raw_header.pixel_size_A),
            raw_pixel_A=raw_header.pixel_size_A,
        )
        return target
    except BaseException as exc:
        return {
            "shape_xyz": raw_header.shape_xyz,
            "pixel_size_A": raw_header.pixel_size_A,
            "physical_size_A": [raw_header.nx * raw_header.pixel_size_A,
                                raw_header.ny * raw_header.pixel_size_A,
                                raw_header.nz * raw_header.pixel_size_A],
            "source": f"fallback raw stack geometry; target unresolved: {exc}",
        }
