#!/usr/bin/env python3
"""Orchestrate the revised-IMOD export (CLI: ``convertMissalignment export revise``).

Ties the tested geometry core (``imod_revision``) and writer (``imod_revision_writer``)
to a real project: it reads the imported ORIGINAL IMOD alignment, loads the
MissAlignment refinement's per-tilt aligned-frame correction (``DeltaH``) produced by
``pipeline.finalize`` for the selected backend, converges them into the canonical
:class:`ImodAlignmentRevision` (composing ``H_final = DeltaH @ H_original``) and publishes
ONE physical export under ``exported_data/imod/<condition_id>`` with a single
compatibility symlink at ``missalignment/runs/<condition_id>/export/imod``.

``pipeline.finalize`` remains the orchestrator that COMPUTES the transforms; this module
is the publication step. It never recomputes the refinement and never touches
``imported_data``.

The heavy backends (warpylib for warp_xml, torch for constrained) are NOT imported here:
this reads the ``.xf`` files finalize already wrote, so the publication step runs and is
tested off-cluster. Live end-to-end still needs finalize to have produced those files.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from imod_affine import read_xf  # noqa: E402
from pipeline import project_config as PC  # noqa: E402
from pipeline.imod_revision import (  # noqa: E402
    Affine2D, OriginalImodGeometry, RevisionError, RevisionPolicy, converge_revision,
)
from pipeline.imod_revision_writer import (  # noqa: E402
    ExportPaths, export_cache_key, write_revision_export,
)
from pipeline.runlayout import RunLayout  # noqa: E402


class ExportInputsMissing(RevisionError):
    """Raised when finalize has not produced the transforms this step needs."""


# --------------------------------------------------------------------------- #
# reading the imported original + the refinement residual
# --------------------------------------------------------------------------- #
def _read_xf_as_affines(path: Path) -> list[Affine2D]:
    matrices, shifts = read_xf(path)
    return [Affine2D(m, s) for m, s in zip(matrices, shifts)]


def _read_angles(path: Path) -> list[float]:
    return [float(x) for x in Path(path).read_text().split()]


def _read_optional_column(path: Optional[Path]) -> Optional[list[float]]:
    if path and Path(path).is_file():
        vals = [float(x) for x in Path(path).read_text().split()]
        return vals or None
    return None


@dataclass(frozen=True)
class RevisionSources:
    """Resolved on-disk inputs for the publication step."""

    series: str
    condition_id: str
    original_xf: Path              # imported raw->aligned .xf (H_original)
    original_tlt: Path
    residual_xf: Path              # finalize DeltaH (aligned-frame) .xf
    raw_stack: Path
    imported_imod_dir: Path
    tilt_com: Optional[Path] = None
    newst_com: Optional[Path] = None
    xtilt: Optional[Path] = None
    stats_json: Optional[Path] = None       # optional per-tilt representability stats
    final_xf: Optional[Path] = None         # finalize's own final .xf (consistency check)
    conversion_manifest: Optional[Path] = None   # conversion_validation.json (sign + view order)


def _tlt_row_count(path: Path) -> int:
    return sum(1 for ln in Path(path).read_text().splitlines() if ln.strip())


def _orientation_manifest(tilt_angle_sign: int) -> dict:
    """The signed IMOD-MRC -> Warp orientation recorded in the export manifest."""
    import numpy as np
    from geometry.volume_frames import BASE_AXIS_PERMUTATION, imod_mrc_to_warp_orientation
    m = imod_mrc_to_warp_orientation(tilt_angle_sign)
    det = int(round(float(np.linalg.det(m))))
    return {
        "orientation_matrix_imod_mrc_to_warp": m.tolist(),
        "orientation_determinant": det,
        "handedness_effect": "preserved" if det == 1 else "flipped",
        "shape_permutation": list(BASE_AXIS_PERMUTATION),
        "tilt_angle_sign": int(tilt_angle_sign),
    }


def _read_conversion_contract(sources: "RevisionSources") -> dict:
    """The recorded conversion contract (tilt-angle sign + view order), read — not assumed.

    Prefers the conversion manifest written at conversion time; returns {} when absent so
    the caller falls back to the resolved config's positioning table.
    """
    p = getattr(sources, "conversion_manifest", None)
    if p and Path(p).is_file():
        try:
            data = json.loads(Path(p).read_text())
        except Exception:
            return {}
        return {
            "imod_to_warp_tilt_angle_sign": data.get("imod_to_warp_tilt_angle_sign"),
            "tilt_view_order": data.get("tilt_view_order"),
            "tilt_angle_convention": data.get("tilt_angle_convention"),
            "tilt_axis_angles_hash": data.get("tilt_axis_angles_hash"),
        }
    return {}


def _load_representability_stats(path: Optional[Path], n: int) -> Optional[list[dict]]:
    """Per-tilt {rms_residual_px, max_residual_px} from a backend validation report.

    Accepts either a list of per-tilt dicts or the warp_to_imod_affine report shape
    ({"per_tilt": [{"stats_px": {"rms","max"}}, ...]}). Absent -> None (exact-affine).
    """
    if not path or not Path(path).is_file():
        return None
    data = json.loads(Path(path).read_text())
    rows = data if isinstance(data, list) else data.get("per_tilt", [])
    out: list[dict] = []
    for row in rows:
        stats = row.get("stats_px", row)
        out.append({"rms_residual_px": float(stats.get("rms", stats.get("rms_residual_px", 0.0))),
                    "max_residual_px": float(stats.get("max", stats.get("max_residual_px", 0.0)))})
    if not out:
        return None
    if len(out) != n:
        raise RevisionError(f"representability stats have {len(out)} rows != {n} tilts")
    return out


def build_revision_from_sources(sources: RevisionSources, *, config: dict,
                                policy: RevisionPolicy, backend: str,
                                provenance: Optional[dict] = None):
    """Load the imported original + finalize DeltaH and converge the canonical object."""
    for label, p in (("original .xf", sources.original_xf),
                     ("residual .xf", sources.residual_xf),
                     ("original .tlt", sources.original_tlt)):
        if not Path(p).is_file():
            raise ExportInputsMissing(
                f"missing {label}: {p}. Run `convertMissalignment export finalize` first "
                "so the refinement transforms exist.")

    geom = config.get("geometry", {})
    raw_xy = tuple(int(v) for v in geom["raw_shape_xyz"][:2])
    ali_xy = tuple(int(v) for v in geom["aligned_shape_xyz"][:2])
    raw_px = float(geom["raw_pixel_size_A"])
    ali_px = float(geom["aligned_pixel_size_A"])

    originals = _read_xf_as_affines(sources.original_xf)
    deltas = _read_xf_as_affines(sources.residual_xf)
    if len(deltas) != len(originals):
        raise RevisionError(
            f"residual .xf has {len(deltas)} rows != {len(originals)} original rows")
    angles = _read_angles(sources.original_tlt)
    xtilt = _read_optional_column(sources.xtilt)

    og = OriginalImodGeometry(
        series=sources.series, raw_shape_xy=raw_xy, aligned_shape_xy=ali_xy,
        raw_pixel_size_A=raw_px, aligned_pixel_size_A=ali_px,
        original_transforms=originals, original_tilt_angles_deg=angles,
        x_axis_tilt_per_view_deg=xtilt)

    stats = _load_representability_stats(sources.stats_json, len(originals))
    positioning = geom.get("imod_positioning") or {}
    revision = converge_revision(
        og, deltas, policy=policy, backend=backend,
        representability_stats=stats,
        original_positioning=positioning,
        revised_positioning={},                      # preserve_unless_refined
        provenance=provenance or {})
    return revision


# --------------------------------------------------------------------------- #
# top-level publication
# --------------------------------------------------------------------------- #
def export_revised_imod(sources: RevisionSources, *, config: dict, layout: RunLayout,
                        policy: Optional[RevisionPolicy] = None, backend: str = "warp_xml",
                        measured_pixel_size_A: Optional[float] = None,
                        source_hashes: Optional[dict] = None,
                        software_versions: Optional[dict] = None,
                        scipion_report: Optional[dict] = None,
                        creation_command: str = "") -> dict:
    """Build the canonical revision and publish the single export tree; return the manifest."""
    policy = policy or RevisionPolicy.from_config(
        (config.get("export", {}) or {}).get("imod_revision"))
    if not policy.enabled:
        raise RevisionError("[export.imod_revision].enabled is false")

    positioning = (config.get("geometry", {}) or {}).get("imod_positioning") or {}
    # Read the IMOD->Warp tilt-angle sign and the view mapping from the conversion manifest
    # (fall back to the resolved config), NOT independently assumed. The sign is +-1 (its own
    # inverse); the Warp->IMOD angle inverse uses it exactly once.
    from geometry.imod_positioning import (
        IMOD_TO_WARP_TILT_ANGLE_SIGN, tilt_angle_convention_manifest,
        tilt_view_order_identity, validate_tilt_angle_sign)
    conv = _read_conversion_contract(sources)
    tilt_angle_sign = validate_tilt_angle_sign(
        conv.get("imod_to_warp_tilt_angle_sign")
        if conv.get("imod_to_warp_tilt_angle_sign") is not None
        else positioning.get("imod_to_warp_tilt_angle_sign", IMOD_TO_WARP_TILT_ANGLE_SIGN))
    n_views = _tlt_row_count(sources.original_tlt)
    view_order = conv.get("tilt_view_order") or tilt_view_order_identity(n_views)

    cache_key = export_cache_key(
        source_geometry_hash=(source_hashes or {}).get("final_xf", {}).get("sha256", ""),
        refined_geometry_hash=(source_hashes or {}).get("residual_xf", {}).get("sha256", ""),
        positioning_hash=positioning.get("positioning_hash", "") if isinstance(positioning, dict) else "",
        volume_frame_contract_version=2, policy=policy,
        imod_version=(software_versions or {}).get("imod", ""),
        tilt_angle_sign=tilt_angle_sign, view_mapping=view_order.get("mapping", "identity"),
        tilt_axis_angles_hash=conv.get("tilt_axis_angles_hash"))

    revision = build_revision_from_sources(
        sources, config=config, policy=policy, backend=backend,
        provenance={
            "positioning_hash": positioning.get("positioning_hash") if isinstance(positioning, dict) else None,
            "volume_frame_contract_version": 2,
            "export_cache_key": cache_key,
            "creation_command": creation_command,
            "software_versions": software_versions or {},
            "original_geometry_hash": (source_hashes or {}).get("final_xf", {}).get("sha256"),
            "refined_geometry_hash": (source_hashes or {}).get("residual_xf", {}).get("sha256"),
            "backend": backend,
            "imod_to_warp_tilt_angle_sign": tilt_angle_sign,
            "tilt_view_order": view_order,
            "tilt_angle_convention": tilt_angle_convention_manifest(tilt_angle_sign),
            "volume_frame_orientation": _orientation_manifest(tilt_angle_sign),
        })

    paths = ExportPaths.resolve(layout.exported_imod_dir, layout.export_imod_link)
    tilt_com_text = sources.tilt_com.read_text() if (sources.tilt_com and sources.tilt_com.is_file()) else ""
    newst_com_text = sources.newst_com.read_text() if (sources.newst_com and sources.newst_com.is_file()) else ""
    xtilt_text = sources.xtilt.read_text() if (sources.xtilt and sources.xtilt.is_file()) else None

    return write_revision_export(
        revision, paths, policy=policy, imported_imod_dir=sources.imported_imod_dir,
        raw_stack_source=sources.raw_stack, original_tilt_com=tilt_com_text,
        original_newst_com=newst_com_text, original_xtilt_text=xtilt_text,
        source_hashes=source_hashes, measured_pixel_size_A=measured_pixel_size_A,
        condition_id=sources.condition_id, scipion_report=scipion_report)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _resolve_sources_from_layout(config: dict, layout: RunLayout, args) -> RevisionSources:
    inp = config.get("input", {})
    series = config.get("project", {}).get("basename") or layout.basename
    transforms = layout.results_dir / "transforms"
    residual = Path(args.residual_xf) if args.residual_xf else (transforms / "source_residual.xf")
    final = transforms / "final_source_raw_to_aligned.xf"
    imported = layout.imported_imod_dir
    return RevisionSources(
        series=series, condition_id=layout.dataset_id,
        original_xf=Path(args.original_xf or inp.get("final_xf_file")),
        original_tlt=Path(args.original_tlt or inp.get("final_tilt_file")),
        residual_xf=residual, raw_stack=Path(args.raw_stack or inp.get("raw_stack")),
        imported_imod_dir=imported,
        tilt_com=Path(inp["tilt_com"]) if inp.get("tilt_com") else None,
        newst_com=Path(inp["newst_com"]) if inp.get("newst_com") else None,
        xtilt=Path(inp["xtilt_file"]) if inp.get("xtilt_file") else None,
        stats_json=Path(args.stats_json) if args.stats_json else None,
        final_xf=final if final.is_file() else None,
        conversion_manifest=(layout.training_dir / "conversion_validation.json"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Publish the revised IMOD alignment under exported_data/imod/<condition_id>.")
    ap.add_argument("settings", type=Path, help="resolved project_settings.toml")
    ap.add_argument("--result-backend", choices=("warp_xml", "constrained_json"), default=None)
    ap.add_argument("--dataset", default=None, help="dataset/condition id (default: config selected)")
    ap.add_argument("--residual-xf", default=None, help="override DeltaH .xf path")
    ap.add_argument("--original-xf", default=None)
    ap.add_argument("--original-tlt", default=None)
    ap.add_argument("--raw-stack", default=None)
    ap.add_argument("--stats-json", default=None, help="per-tilt representability stats JSON")
    ap.add_argument("--measured-pixel-size", type=float, default=None)
    args = ap.parse_args(argv)

    rc = PC.load(args.settings)
    config = rc.raw
    dataset_id = args.dataset or (config.get("datasets", {}) or {}).get("selected_id") \
        or (config.get("datasets", {}) or {}).get("native_id") or "native"
    layout = RunLayout.from_settings(
        out_dir=Path(config.get("paths", {}).get("output_dir", ".")),
        basename=rc.basename, condition=(rc.conditions or ["raw_xf_affine_fixed"])[0],
        refinement_mode=rc.refinement_mode, dataset_id=dataset_id).create()

    backend = args.result_backend or config.get("missalignment", {}).get("result_backend", "warp_xml")
    sources = _resolve_sources_from_layout(config, layout, args)

    # source hashes for the two geometry-defining files
    from pipeline.imod_revision_writer import _sha256_file
    source_hashes = {}
    for name, p in (("raw_stack", sources.raw_stack), ("final_xf", sources.original_xf),
                    ("residual_xf", sources.residual_xf), ("tilt_file", sources.original_tlt)):
        h = _sha256_file(p) if p and Path(p).is_file() else None
        if h:
            source_hashes[name] = h

    try:
        manifest = export_revised_imod(
            sources, config=config, layout=layout, backend=backend,
            measured_pixel_size_A=args.measured_pixel_size or float(
                (config.get("geometry", {}) or {}).get("target_pixel_size_A", 0.0)) or None,
            source_hashes=source_hashes,
            creation_command="convertMissalignment export revise " + " ".join(sys.argv[1:]))
    except ExportInputsMissing as exc:
        print(f"ERROR: {exc}")
        return 3
    except RevisionError as exc:
        print(f"ERROR: revised-IMOD export refused: {exc}")
        return 1

    print(f"[export] revised IMOD published: {manifest['physical_export_path']}")
    print(f"[export] compatibility symlink : {manifest['compatibility_symlink_path']}")
    print(f"[export] representability       : {manifest['representability_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
