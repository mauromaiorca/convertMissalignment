#!/usr/bin/env python3
"""WarpTools diagnostic reconstruction of pre/full MissAlignment snapshots.

This executor is called by a generated Warp reconstruction batch.  It reconstructs
copies of the Warp ``pre_missalign`` and ``missalign_full`` snapshots from the
same quantitative tilt stack.  It never modifies the source snapshots.

The current path is explicitly diagnostic: legacy converted XMLs may not carry
valid movie paths or experimental cumulative dose.  The executor materializes
one image per tilt and, only when source dose is unusable, writes a tiny
monotonic epsilon dose to the diagnostic copies to avoid Warp's division by a
zero dose range.  That fallback is recorded in the preparation manifest and is
not valid experimental dose metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from . import project_config as PC
    from .runlayout import RunLayout, dataset_id_from_config, format_angpix
except ImportError:  # direct execution from a generated Slurm job
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline import project_config as PC
    from pipeline.runlayout import RunLayout, dataset_id_from_config, format_angpix


class WarpToolsReconstructionError(RuntimeError):
    pass


def load_conversion_volume_contract(
    source_directory: Path,
    series: str,
    *,
    xml_volume_dimensions_A: Sequence[float] | None = None,
    relative_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Load and validate the explicit IMOD-MRC -> Warp volume-frame contract.

    Contract v2 separates the 2-D detector quarter turn from the 3-D
    reconstruction-volume frame. Contract-v1 conversions are accepted only
    when no odd detector quarter turn was applied; v1 affine conversions must
    be regenerated because they encoded a spurious Warp X/Y volume swap.
    """

    path = Path(source_directory) / f"{series}.conversion.json"
    if not path.is_file():
        raise WarpToolsReconstructionError(
            f"missing conversion manifest {path}; reconvert this Warp project with "
            "the current volume-frame contract"
        )
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise WarpToolsReconstructionError(
            f"invalid conversion manifest {path}: {exc}"
        ) from exc

    frame = data.get("volume_frame") or {}
    version = int(frame.get("contract_version", 0))
    quarter_turn_k = int(frame.get("projection_quarter_turn_k", 0)) % 4

    if version >= 2:
        shape = (
            frame.get("reconstruction_shape_warp_xyz")
            or frame.get("current_shape_warp_xyz")
            or data.get("warp_volume_shape_xyz")
        )
    elif version == 1 and quarter_turn_k % 2 == 0:
        # Translation/identity conversions from alpha3 already used the correct
        # IMOD-MRC (X,Z,Y) -> Warp volume mapping and are safe to reuse.
        shape = (
            frame.get("base_shape_warp_xyz")
            or frame.get("current_shape_warp_xyz")
            or data.get("warp_volume_shape_xyz")
        )
    elif version == 1:
        raise WarpToolsReconstructionError(
            f"stale contract-v1 affine conversion in {path}: an odd detector "
            "quarter turn was incorrectly applied to the Warp volume X/Y extents; "
            "force reconversion with contract v2"
        )
    else:
        raise WarpToolsReconstructionError(
            f"legacy or invalid volume-frame contract in {path}; force reconversion"
        )

    try:
        shape_xyz = tuple(int(value) for value in shape)
    except Exception as exc:
        raise WarpToolsReconstructionError(
            f"conversion manifest lacks Warp reconstruction XYZ shape: {path}"
        ) from exc
    if len(shape_xyz) != 3 or any(value <= 0 for value in shape_xyz):
        raise WarpToolsReconstructionError(
            f"invalid Warp reconstruction XYZ shape in {path}: {shape_xyz}"
        )

    pixel = float(data.get("output_pixel_size_A") or 0.0)
    if not math.isfinite(pixel) or pixel <= 0:
        raise WarpToolsReconstructionError(
            f"invalid conversion output pixel size in {path}: {pixel}"
        )
    expected_A = [value * pixel for value in shape_xyz]
    if xml_volume_dimensions_A is not None:
        got = [float(value) for value in xml_volume_dimensions_A]
        if len(got) != 3:
            raise WarpToolsReconstructionError(
                f"invalid XML VolumeDimensionsAngstrom: {got}"
            )
        for axis, (observed, expected) in enumerate(zip(got, expected_A, strict=True)):
            if expected <= 0 or abs(observed - expected) / expected > relative_tolerance:
                raise WarpToolsReconstructionError(
                    "XML volume dimensions disagree with conversion frame contract: "
                    f"axis={axis}, observed_A={observed}, expected_A={expected}, "
                    f"manifest={path}"
                )
    return {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": sha256_file(path),
        "shape_warp_xyz": list(shape_xyz),
        "pixel_size_A": pixel,
        "physical_dimensions_A_xyz": expected_A,
        "volume_frame": frame,
        "volume_frame_contract_version": version,
        "legacy_v1_translation_accepted": bool(version == 1),
    }


def volume_shape_xyz_from_angstrom(
    volume_dimensions_A: Sequence[float],
    pixel_size_A: float,
    *,
    integer_tolerance_px: float = 0.1,
) -> tuple[int, int, int]:
    """Convert Warp's declared physical volume extent to an XYZ voxel shape.

    ``VolumeDimensionsAngstrom`` is a volume-space contract and is independent
    of the projection-stack width and height.  In particular, a lossless
    quarter-turn of the projection images swaps the stack's X/Y dimensions but
    must not silently swap the requested reconstruction volume dimensions.

    The returned order is exactly the XML order: X, Y, Z.
    """

    values = [float(value) for value in volume_dimensions_A]
    if len(values) != 3 or not all(math.isfinite(value) and value > 0 for value in values):
        raise WarpToolsReconstructionError(
            f"invalid VolumeDimensionsAngstrom: {volume_dimensions_A!r}"
        )
    pixel = float(pixel_size_A)
    if not math.isfinite(pixel) or pixel <= 0:
        raise WarpToolsReconstructionError(f"invalid volume pixel size: {pixel_size_A!r}")

    floating_shape = [value / pixel for value in values]
    shape = tuple(max(1, int(round(value))) for value in floating_shape)
    errors = [abs(value - rounded) for value, rounded in zip(floating_shape, shape, strict=True)]
    if max(errors) > float(integer_tolerance_px):
        raise WarpToolsReconstructionError(
            "VolumeDimensionsAngstrom is not compatible with the input pixel size: "
            f"dimensions_A={values}, pixel_size_A={pixel}, "
            f"shape_float={floating_shape}, max_rounding_error_px={max(errors):.6g}"
        )
    return shape


@dataclass(frozen=True)
class WarpToolsPlan:
    settings_path: Path
    run_dir: Path
    attempt_dir: Path
    work_dir: Path
    pre_source: Path
    full_source: Path
    pre_source_xml: Path
    full_source_xml: Path
    quantitative_stack: Path
    raw_data_dir: Path
    pre_input: Path
    full_input: Path
    pre_output: Path
    full_output: Path
    tomostar: Path
    warptools_settings: Path
    preparation_manifest: Path
    result_manifest: Path
    warptools_executable: str
    output_angpix: float | None
    device_list: str
    perdevice: int
    dataset_id: str
    public_results_dir: Path


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=str, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_settings(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise WarpToolsReconstructionError(
            f"project settings not found: {path}"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise WarpToolsReconstructionError(
            f"invalid project settings TOML {path}: {exc}"
        ) from exc


def layout_for(cfg: dict[str, Any], dataset_id: str | None = None) -> RunLayout:
    resolved = PC.from_dict(cfg).require_resolved()
    if len(resolved.conditions) != 1:
        raise WarpToolsReconstructionError(
            "WarpTools pre/full reconstruction requires exactly one condition; "
            f"found {resolved.conditions}"
        )
    return RunLayout.from_settings(
        out_dir=Path(resolved.output_dir),
        basename=resolved.basename,
        condition=resolved.conditions[0],
        refinement_mode=resolved.refinement_mode,
        dataset_id=dataset_id or dataset_id_from_config(cfg),
    )


def exactly_one_root_xml(directory: Path, label: str) -> Path:
    candidates = sorted(
        path
        for path in directory.glob("*.xml")
        if path.is_file() and path.stat().st_size > 0
    )
    if len(candidates) != 1:
        raise WarpToolsReconstructionError(
            f"{label}: expected exactly one non-empty root XML in {directory}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def resolve_source_snapshots(layout: RunLayout) -> tuple[Path, Path]:
    result_path = layout.manifest("result_manifest.json")
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    pre = Path(
        result.get("pre_missalign_directory")
        or layout.pre_missalign_dir
    )
    full = Path(
        result.get("training_directory")
        or layout.full_warp_dir
    )
    for label, directory in (("pre_missalign", pre), ("missalign_full", full)):
        if not directory.is_dir():
            raise WarpToolsReconstructionError(
                f"missing {label} Warp snapshot: {directory}; the MissAlignment run must complete "
                "or its immutable results must be imported before this job"
            )
    return pre.resolve(), full.resolve()


def next_attempt_dir(layout: RunLayout) -> Path:
    root = layout.attempts_dir / "reconstruction" / layout.dataset_id / "pre_vs_full"
    root.mkdir(parents=True, exist_ok=True)
    slurm_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if slurm_id:
        candidate = root / f"attempt_{slurm_id}"
        if candidate.exists():
            raise WarpToolsReconstructionError(
                f"attempt directory already exists: {candidate}; refusing overwrite"
            )
        return candidate
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for index in range(1, 1000):
        candidate = root / f"attempt_{timestamp}_{index:03d}"
        if not candidate.exists():
            return candidate
    raise WarpToolsReconstructionError("could not allocate an attempt directory")


def build_plan(
    settings_path: Path,
    *,
    output_angpix: float | None = None,
    device_list: str = "0",
    perdevice: int = 1,
    dataset_id: str | None = None,
) -> WarpToolsPlan:
    cfg = load_settings(settings_path)
    layout = layout_for(cfg, dataset_id)
    pre_source, full_source = resolve_source_snapshots(layout)
    pre_xml = exactly_one_root_xml(pre_source, "pre_missalign")
    full_xml = exactly_one_root_xml(full_source, "missalign_full")
    if pre_xml.stem != full_xml.stem:
        raise WarpToolsReconstructionError(
            f"pre/full series mismatch: {pre_xml.stem!r} versus {full_xml.stem!r}"
        )
    series = pre_xml.stem
    pre_stack = pre_source / "tiltstack" / series / f"{series}.st"
    full_stack = full_source / "tiltstack" / series / f"{series}.st"
    for label, path in (("pre stack", pre_stack), ("full stack", full_stack)):
        if not path.is_file() or path.stat().st_size <= 0:
            raise WarpToolsReconstructionError(f"{label} missing or empty: {path}")
    if pre_stack.resolve() != full_stack.resolve():
        raise WarpToolsReconstructionError(
            "pre/full snapshots do not reference the same quantitative tilt stack:\n"
            f"  pre:  {pre_stack.resolve()}\n"
            f"  full: {full_stack.resolve()}"
        )

    attempt = next_attempt_dir(layout)
    work = attempt / "work"
    raw_data = work / "raw_data"
    pre_input = work / "input_pre_missalign"
    full_input = work / "input_missalign_full"
    pre_output = attempt / "output_pre_missalign"
    full_output = attempt / "output_missalign_full"
    cluster = cfg.get("cluster", {}) or {}
    rec = cfg.get("reconstruction", {}) or {}
    wt = rec.get("warptools", {}) or {}
    executable = str(
        wt.get("executable")
        or cluster.get("warp_tools_executable")
        or "WarpTools"
    )
    configured_angpix = wt.get("output_angpix_A")
    selected_angpix = output_angpix
    if selected_angpix in (None, 0, 0.0) and configured_angpix not in (None, 0, 0.0):
        selected_angpix = float(configured_angpix)

    return WarpToolsPlan(
        settings_path=Path(settings_path).resolve(),
        run_dir=layout.run_dir,
        attempt_dir=attempt,
        work_dir=work,
        pre_source=pre_source,
        full_source=full_source,
        pre_source_xml=pre_xml.resolve(),
        full_source_xml=full_xml.resolve(),
        quantitative_stack=pre_stack.resolve(),
        raw_data_dir=raw_data,
        pre_input=pre_input,
        full_input=full_input,
        pre_output=pre_output,
        full_output=full_output,
        tomostar=raw_data / f"{series}.tomostar",
        warptools_settings=work / "warp_tiltseries.settings",
        preparation_manifest=attempt / "preparation_manifest.json",
        result_manifest=attempt / "result_manifest.json",
        warptools_executable=executable,
        output_angpix=float(selected_angpix) if selected_angpix not in (None, 0, 0.0) else None,
        device_list=str(device_list),
        perdevice=int(perdevice),
        dataset_id=layout.dataset_id,
        public_results_dir=layout.results_dir / "reconstructions" / "warp_comparison",
    )


def _as_array(value: Any, *, dtype=float):
    import numpy as np

    if value is None:
        return np.asarray([], dtype=dtype)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype).reshape(-1)


def choose_dose_values(
    pre_dose: Sequence[float],
    full_dose: Sequence[float],
    n_tilts: int,
):
    """Return comparable finite non-constant dose values and an explicit policy."""
    import numpy as np

    pre = np.asarray(pre_dose, dtype=float).reshape(-1)
    full = np.asarray(full_dose, dtype=float).reshape(-1)

    def usable(values) -> bool:
        return (
            len(values) == n_tilts
            and bool(np.all(np.isfinite(values)))
            and float(np.ptp(values)) > 1e-8
        )

    pre_ok = usable(pre)
    full_ok = usable(full)
    if pre_ok and full_ok:
        if not np.allclose(pre, full, rtol=1e-6, atol=1e-8):
            raise WarpToolsReconstructionError(
                "pre/full source dose arrays are both usable but differ"
            )
        values = pre.copy()
        policy = "preserved_identical_source_dose"
    elif pre_ok:
        values = pre.copy()
        policy = "preserved_pre_source_dose_and_applied_to_full"
    elif full_ok:
        values = full.copy()
        policy = "preserved_full_source_dose_and_applied_to_pre"
    else:
        # Warp normalizes Dose using (value-min)/(max-min).  A constant source
        # array creates non-finite temporal coordinates in CTF interpolation.
        values = np.arange(n_tilts, dtype=float) * 1e-6
        policy = "synthetic_monotonic_epsilon_for_warp_coordinate_only"

    span = float(np.ptp(values))
    if len(values) != n_tilts or not bool(np.all(np.isfinite(values))) or span <= 0:
        raise WarpToolsReconstructionError(
            "WarpTools diagnostic dose must be finite and non-constant"
        )
    normalized = (values - float(np.min(values))) / span
    if not bool(np.all(np.isfinite(normalized))):
        raise WarpToolsReconstructionError("normalized dose contains non-finite values")
    return values, policy, pre_ok, full_ok


def _xml_root(path: Path) -> ET.Element:
    root = ET.parse(path).getroot()
    if root.tag.rsplit("}", 1)[-1] != "TiltSeries":
        raise WarpToolsReconstructionError(
            f"not a Warp TiltSeries XML: {path}; root={root.tag!r}"
        )
    return root


def _vector_attribute(root: ET.Element, name: str) -> list[float]:
    raw = root.attrib.get(name)
    if raw is None:
        raise WarpToolsReconstructionError(f"missing XML attribute {name}")
    values = [float(item.strip()) for item in raw.split(",")]
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise WarpToolsReconstructionError(f"invalid XML attribute {name}={raw!r}")
    return values


def _set_vector_node(root: ET.Element, local_name: str, values: Sequence[str]) -> None:
    nodes = [
        child for child in list(root)
        if child.tag.rsplit("}", 1)[-1] == local_name
    ]
    if nodes:
        node = nodes[0]
        for extra in nodes[1:]:
            root.remove(extra)
    else:
        node = ET.SubElement(root, local_name)
    node.text = "\n".join(values)


def _patch_xml(
    source: Path,
    destination: Path,
    *,
    raw_data_dir: Path,
    movie_names: Sequence[str],
    dose_values: Sequence[float],
) -> None:
    shutil.copy2(source, destination)
    tree = ET.parse(destination)
    root = tree.getroot()
    if root.tag.rsplit("}", 1)[-1] != "TiltSeries":
        raise WarpToolsReconstructionError(f"not a TiltSeries XML: {source}")
    root.set("DataDirectory", str(raw_data_dir))
    root.set("UnselectManual", "")
    _set_vector_node(root, "MoviePath", list(movie_names))
    _set_vector_node(root, "Dose", [f"{float(value):.12g}" for value in dose_values])
    tree.write(destination, encoding="UTF-8", xml_declaration=True)


def prepare_workspace(plan: WarpToolsPlan) -> dict[str, Any]:
    try:
        import mrcfile
        import numpy as np
        from warpylib import TiltSeries
    except ImportError as exc:
        raise WarpToolsReconstructionError(
            "the Phase-3 Python environment must provide mrcfile, numpy, and warpylib"
        ) from exc

    for directory in (
        plan.raw_data_dir,
        plan.work_dir / "default_processing",
        plan.pre_input,
        plan.full_input,
        plan.pre_output,
        plan.full_output,
    ):
        directory.mkdir(parents=True, exist_ok=False)
    average_dir = plan.raw_data_dir / "average"
    average_dir.mkdir()

    pre_ts = TiltSeries(str(plan.pre_source_xml))
    full_ts = TiltSeries(str(plan.full_source_xml))
    pre_angles = _as_array(pre_ts.angles)
    full_angles = _as_array(full_ts.angles)
    axis_angles = _as_array(pre_ts.tilt_axis_angles)
    offsets_x = _as_array(pre_ts.tilt_axis_offset_x)
    offsets_y = _as_array(pre_ts.tilt_axis_offset_y)
    pre_dose = _as_array(getattr(pre_ts, "dose", None))
    full_dose = _as_array(getattr(full_ts, "dose", None))

    pre_root = _xml_root(plan.pre_source_xml)
    full_root = _xml_root(plan.full_source_xml)
    pre_volume = _vector_attribute(pre_root, "VolumeDimensionsAngstrom")
    full_volume = _vector_attribute(full_root, "VolumeDimensionsAngstrom")
    if not np.allclose(pre_volume, full_volume, rtol=1e-6, atol=1e-2):
        raise WarpToolsReconstructionError(
            "pre/full XMLs describe different physical volume dimensions"
        )

    with mrcfile.mmap(plan.quantitative_stack, mode="r", permissive=True) as source:
        if source.data.ndim != 3:
            raise WarpToolsReconstructionError(
                f"expected a 3-D tilt stack, got {source.data.shape}"
            )
        n_tilts, ny, nx = map(int, source.data.shape)
        pixel_x = float(source.voxel_size.x)
        pixel_y = float(source.voxel_size.y)
        if pixel_x <= 0 or pixel_y <= 0:
            raise WarpToolsReconstructionError(
                f"invalid stack pixel size {pixel_x}, {pixel_y}"
            )
        if abs(pixel_x - pixel_y) > max(1e-4, pixel_x * 1e-5):
            raise WarpToolsReconstructionError(
                f"anisotropic stack pixels are unsupported: {pixel_x}, {pixel_y}"
            )
        for label, values in (
            ("pre angles", pre_angles),
            ("full angles", full_angles),
            ("axis angles", axis_angles),
            ("offset X", offsets_x),
            ("offset Y", offsets_y),
        ):
            if len(values) != n_tilts:
                raise WarpToolsReconstructionError(
                    f"{label}: {len(values)} values for {n_tilts} stack sections"
                )
        if not np.allclose(pre_angles, full_angles, rtol=0, atol=1e-5):
            raise WarpToolsReconstructionError("pre/full tilt-angle arrays differ")

        dose_values, dose_policy, pre_dose_ok, full_dose_ok = choose_dose_values(
            pre_dose, full_dose, n_tilts
        )
        series = plan.pre_source_xml.stem
        movie_names: list[str] = []
        for index in range(n_tilts):
            name = f"{series}_tilt_{index:04d}.mrc"
            output = plan.raw_data_dir / name
            temporary = plan.raw_data_dir / f".{name}.tmp"
            image = np.asarray(source.data[index], dtype=np.float32)
            with mrcfile.new(temporary, overwrite=True) as destination:
                destination.set_data(image)
                destination.voxel_size = (pixel_x, pixel_y, pixel_x)
                destination.update_header_stats()
            os.replace(temporary, output)
            (average_dir / name).symlink_to(Path("..") / name)
            movie_names.append(name)

    averages_alias = plan.raw_data_dir / "averages"
    averages_alias.symlink_to("average", target_is_directory=True)

    pre_contract = load_conversion_volume_contract(
        plan.pre_source,
        series,
        xml_volume_dimensions_A=pre_volume,
    )
    full_contract = load_conversion_volume_contract(
        plan.full_source,
        series,
        xml_volume_dimensions_A=full_volume,
    )
    if pre_contract["shape_warp_xyz"] != full_contract["shape_warp_xyz"]:
        raise WarpToolsReconstructionError(
            "pre/full conversion volume-frame shapes differ: "
            f"{pre_contract['shape_warp_xyz']} versus {full_contract['shape_warp_xyz']}"
        )
    volume_shape_xyz = tuple(pre_contract["shape_warp_xyz"])
    dimensions = "x".join(str(value) for value in volume_shape_xyz)

    pre_xml = plan.pre_input / plan.pre_source_xml.name
    full_xml = plan.full_input / plan.full_source_xml.name
    _patch_xml(
        plan.pre_source_xml, pre_xml,
        raw_data_dir=plan.raw_data_dir,
        movie_names=movie_names,
        dose_values=dose_values,
    )
    _patch_xml(
        plan.full_source_xml, full_xml,
        raw_data_dir=plan.raw_data_dir,
        movie_names=movie_names,
        dose_values=dose_values,
    )

    with plan.tomostar.open("w", encoding="utf-8") as handle:
        handle.write("data_\n\nloop_\n")
        handle.write("_wrpMovieName #1\n")
        handle.write("_wrpAngleTilt #2\n")
        handle.write("_wrpAxisAngle #3\n")
        handle.write("_wrpAxisOffsetX #4\n")
        handle.write("_wrpAxisOffsetY #5\n")
        handle.write("_wrpDose #6\n")
        for index, name in enumerate(movie_names):
            handle.write(
                f"{name} {pre_angles[index]:.8f} {axis_angles[index]:.8f} "
                f"{offsets_x[index]:.8f} {offsets_y[index]:.8f} "
                f"{dose_values[index]:.12g}\n"
            )

    for name in movie_names:
        if not (plan.raw_data_dir / name).is_file():
            raise WarpToolsReconstructionError(f"missing raw tilt: {name}")
        if not (average_dir / name).is_file():
            raise WarpToolsReconstructionError(f"missing average tilt: {name}")

    manifest = {
        "schema_version": 4,
        "purpose": "WarpTools-only diagnostic comparison of pre/full MissAlignment geometry",
        "quantitative_branch": True,
        "source_snapshots_modified": False,
        "series": plan.pre_source_xml.stem,
        "quantitative_stack": str(plan.quantitative_stack),
        "quantitative_stack_sha256": sha256_file(plan.quantitative_stack),
        "source_pre_xml": str(plan.pre_source_xml),
        "source_full_xml": str(plan.full_source_xml),
        "source_pre_xml_sha256": sha256_file(plan.pre_source_xml),
        "source_full_xml_sha256": sha256_file(plan.full_source_xml),
        "prepared_pre_xml": str(pre_xml),
        "prepared_full_xml": str(full_xml),
        "raw_data_directory": str(plan.raw_data_dir),
        "tomostar": str(plan.tomostar),
        "n_tilts": n_tilts,
        "stack_shape_zyx": [n_tilts, ny, nx],
        "input_pixel_size_A": pixel_x,
        "projection_dimensions_xy_A": [nx * pixel_x, ny * pixel_y],
        "volume_dimensions_A_xyz": [float(value) for value in pre_volume],
        "conversion_volume_contract_pre": pre_contract,
        "conversion_volume_contract_full": full_contract,
        "tomo_dimensions_xyz": list(volume_shape_xyz),
        "tomo_dimensions_argument": dimensions,
        "tomo_dimensions_source": (
            "explicit current Warp XYZ shape from the conversion manifest; "
            "legacy axis inference is forbidden"
        ),
        "dose_policy": dose_policy,
        "source_pre_dose_usable": pre_dose_ok,
        "source_full_dose_usable": full_dose_ok,
        "diagnostic_dose_min": float(np.min(dose_values)),
        "diagnostic_dose_max": float(np.max(dose_values)),
        "diagnostic_dose_span": float(np.ptp(dose_values)),
        "dose_warning": (
            "Synthetic epsilon dose, when used, is a Warp numerical coordinate only; "
            "it is not experimental cumulative dose and must not be used for quantitative dose weighting."
        ),
        "allowed_uses": ["visualization", "diagnostic geometry comparison"],
        "forbidden_uses": ["FSC resolution estimation", "quantitative dose validation"],
    }
    atomic_json(plan.preparation_manifest, manifest)
    return manifest


def _run(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(completed.stdout or "")
        sys.stdout.write(completed.stdout or "")
        sys.stdout.flush()
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout or "").splitlines()[-100:])
        raise WarpToolsReconstructionError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{tail}"
        )


def find_reconstruction(output_dir: Path) -> Path:
    candidates = sorted(
        path for path in (output_dir / "reconstruction").glob("*.mrc")
        if path.is_file() and path.stat().st_size > 0
    )
    if not candidates:
        raise WarpToolsReconstructionError(
            f"WarpTools produced no non-empty reconstruction MRC under {output_dir / 'reconstruction'}"
        )
    return candidates[0].resolve()


def run_reconstruction(plan: WarpToolsPlan) -> dict[str, Any]:
    preparation = prepare_workspace(plan)
    input_angpix = float(preparation["input_pixel_size_A"])
    output_angpix = float(plan.output_angpix or input_angpix)
    if not math.isfinite(output_angpix) or output_angpix <= 0:
        raise WarpToolsReconstructionError(f"invalid output pixel size: {output_angpix}")
    if output_angpix < input_angpix - 1e-6:
        raise WarpToolsReconstructionError(
            f"output pixel size {output_angpix} would upsample input {input_angpix}"
        )
    executable = shutil.which(plan.warptools_executable) or plan.warptools_executable
    if not Path(executable).is_file() and shutil.which(executable) is None:
        raise WarpToolsReconstructionError(
            f"WarpTools executable not found: {plan.warptools_executable}"
        )

    create_command = [
        executable, "create_settings",
        "--output", plan.warptools_settings.name,
        "--folder_processing", "default_processing",
        "--folder_data", "raw_data",
        "--extension", "*.tomostar",
        "--angpix", f"{input_angpix:.12g}",
        "--tomo_dimensions", str(preparation["tomo_dimensions_argument"]),
    ]
    _run(
        create_command,
        cwd=plan.work_dir,
        log_path=plan.attempt_dir / "create_settings.log",
    )
    if not plan.warptools_settings.is_file():
        raise WarpToolsReconstructionError(
            f"WarpTools settings were not created: {plan.warptools_settings}"
        )

    common = [
        executable, "ts_reconstruct",
        "--settings", str(plan.warptools_settings),
        "--input_data", str(plan.tomostar),
        "--angpix", f"{output_angpix:.12g}",
        "--device_list", plan.device_list,
        "--perdevice", str(plan.perdevice),
        "--dont_invert",
        "--dont_normalize",
        "--dont_mask",
    ]
    _run(
        common + [
            "--input_processing", str(plan.pre_input),
            "--output_processing", str(plan.pre_output),
        ],
        cwd=plan.work_dir,
        log_path=plan.attempt_dir / "pre_missalign_reconstruct.log",
    )
    pre_volume = find_reconstruction(plan.pre_output)

    _run(
        common + [
            "--input_processing", str(plan.full_input),
            "--output_processing", str(plan.full_output),
        ],
        cwd=plan.work_dir,
        log_path=plan.attempt_dir / "missalign_full_reconstruct.log",
    )
    full_volume = find_reconstruction(plan.full_output)

    result = {
        "schema_version": 2,
        "status": "completed",
        "purpose": "diagnostic pre/full MissAlignment geometry comparison",
        "preparation_manifest": str(plan.preparation_manifest),
        "output_pixel_size_A": output_angpix,
        "pre_missalign_reconstruction": str(pre_volume),
        "pre_missalign_size": pre_volume.stat().st_size,
        "missalign_full_reconstruction": str(full_volume),
        "missalign_full_size": full_volume.stat().st_size,
        "quantitative_warning": (
            "Do not use these diagnostic volumes for FSC or quantitative dose/CTF validation "
            "unless the source XML contains experimentally valid dose and CTF metadata."
        ),
    }
    atomic_json(plan.result_manifest, result)
    latest = plan.attempt_dir.parent / "latest_success"
    temporary_link = latest.with_name(latest.name + ".tmp")
    if temporary_link.exists() or temporary_link.is_symlink():
        temporary_link.unlink()
    temporary_link.symlink_to(plan.attempt_dir.name, target_is_directory=True)
    os.replace(temporary_link, latest)
    public = plan.public_results_dir / format_angpix(output_angpix)
    public.mkdir(parents=True, exist_ok=True)
    for label, source in (("before.mrc", pre_volume), ("final.mrc", full_volume)):
        destination = public / label
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(os.path.relpath(source, start=public))
    public_result = dict(result)
    public_result.update({
        "dataset_id": plan.dataset_id,
        "pre_missalign_reconstruction": str(public / "before.mrc"),
        "missalign_full_reconstruction": str(public / "final.mrc"),
        "internal_attempt": str(plan.attempt_dir),
    })
    atomic_json(public / "manifest.json", public_result)
    result["public_manifest"] = str(public / "manifest.json")
    atomic_json(plan.result_manifest, result)
    return result


def verify_runtime_hashes(
    plan: WarpToolsPlan,
    *,
    expected_executor_sha: str,
    expected_settings_sha: str,
    allow_mismatch: bool,
) -> None:
    current_executor = sha256_file(Path(__file__).resolve())
    current_settings = sha256_file(plan.settings_path)
    mismatches = []
    if expected_executor_sha and expected_executor_sha != current_executor:
        mismatches.append(
            f"executor changed: expected {expected_executor_sha}, got {current_executor}"
        )
    if expected_settings_sha and expected_settings_sha != current_settings:
        mismatches.append(
            f"project settings changed: expected {expected_settings_sha}, got {current_settings}"
        )
    if mismatches and not allow_mismatch:
        raise WarpToolsReconstructionError(
            "stale generated job detected; regenerate jobs from the authoritative TOML:\n  - "
            + "\n  - ".join(mismatches)
        )
    if mismatches:
        print("WARNING: provenance mismatch explicitly allowed:")
        for mismatch in mismatches:
            print(f"  - {mismatch}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--project-settings", required=True, type=Path)
    run.add_argument("--output-angpix", type=float, default=0.0)
    run.add_argument("--dataset", default=None)
    run.add_argument("--device-list", default="0")
    run.add_argument("--perdevice", type=int, default=1)
    run.add_argument("--expected-executor-sha", default="")
    run.add_argument("--expected-settings-sha", default="")
    run.add_argument("--allow-provenance-mismatch", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        plan = build_plan(
            args.project_settings,
            output_angpix=args.output_angpix,
            device_list=args.device_list,
            perdevice=args.perdevice,
            dataset_id=args.dataset,
        )
        verify_runtime_hashes(
            plan,
            expected_executor_sha=args.expected_executor_sha,
            expected_settings_sha=args.expected_settings_sha,
            allow_mismatch=args.allow_provenance_mismatch,
        )
        print(f"[warptools] attempt: {plan.attempt_dir}")
        result = run_reconstruction(plan)
        print(f"[warptools] pre:  {result['pre_missalign_reconstruction']}")
        print(f"[warptools] full: {result['missalign_full_reconstruction']}")
        print(f"[warptools] manifest: {plan.result_manifest}")
        return 0
    except WarpToolsReconstructionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
