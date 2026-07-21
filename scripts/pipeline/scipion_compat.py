#!/usr/bin/env python3
"""Optional Scipion compatibility audit for the revised IMOD alignment.

Scipion is a COMPATIBILITY CHECK ONLY — never the production writer, a runtime
dependency, a source of truth, or an absolute reference. This module compares the
shared geometry contract (effective angles, ``.xf`` point mappings, translation
scaling, matrix inversion/order, tilt ordering, ``H_final`` composition) by comparing
transformed POINTS and effective projection geometry, not raw text of ``.xf``/``.tlt``.

Every comparison is classified:

    COMPATIBLE                our mapping and Scipion's agree within tolerance
    EQUIVALENT_REPRESENTATION agree up to a known representation difference
    NOT_COVERED_BY_SCIPION    Scipion's inspected path does not represent this quantity
    INCOMPATIBLE              a real disagreement (diagnostic emitted, never auto-applied)
    UNRESOLVED               could not be evaluated (e.g. Scipion absent, in required mode)

Scipion disagreement never overwrites the repository result. The export works when
Scipion is not installed unless validation is explicitly configured as required.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imod_affine import forward_points_pixels, regular_grid_points  # noqa: E402

COMPAT_CONTRACT_VERSION = 1
CLASSES = ("COMPATIBLE", "EQUIVALENT_REPRESENTATION", "NOT_COVERED_BY_SCIPION",
           "INCOMPATIBLE", "UNRESOLVED")


def scipion_available() -> bool:
    """True iff a Scipion / pwem transform module can be imported."""
    for name in ("pwem", "pyworkflow", "tomo"):
        if importlib.util.find_spec(name) is not None:
            return True
    return False


def _classify_point_agreement(ours: np.ndarray, theirs: np.ndarray, *,
                              tol_px: float) -> tuple[str, dict]:
    ours = np.asarray(ours, dtype=float).reshape(-1, 2)
    theirs = np.asarray(theirs, dtype=float).reshape(-1, 2)
    resid = np.linalg.norm(ours - theirs, axis=1)
    stats = {"rms_px": float(np.sqrt(np.mean(resid ** 2))),
             "max_px": float(np.max(resid)) if resid.size else 0.0}
    if stats["max_px"] <= tol_px:
        return "COMPATIBLE", stats
    # a pure global flip/transpose that still lands within tol after the known
    # Y/X or sign convention would be EQUIVALENT_REPRESENTATION; detect a consistent flip
    flipped = theirs[:, ::-1]
    fresid = np.linalg.norm(ours - flipped, axis=1)
    if float(np.max(fresid)) <= tol_px:
        return "EQUIVALENT_REPRESENTATION", {
            **stats, "note": "agree after X/Y axis-order convention swap"}
    return "INCOMPATIBLE", stats


def audit_revision(revision, *, tolerance_px: float = 0.25,
                   required: bool = False,
                   scipion_mapping_provider=None) -> dict:
    """Compare our per-tilt point mapping against Scipion's for the SAME transforms.

    ``scipion_mapping_provider(tilt_index, points_xy) -> points_xy`` returns where
    Scipion maps the given raw points under the final transform. When Scipion is
    unavailable and ``required`` is False the audit is UNRESOLVED (non-fatal); when
    ``required`` is True the caller decides whether to abort.
    """
    report = {
        "contract_version": COMPAT_CONTRACT_VERSION,
        "scipion_available": scipion_available(),
        "required": bool(required),
        "tolerance_px": float(tolerance_px),
        "status": "UNRESOLVED",
        "comparisons": [],
        "notes": [],
    }
    if scipion_mapping_provider is None:
        report["notes"].append(
            "no Scipion mapping provider supplied; the general Warp non-affine grid and an "
            "independently-recoverable 3D SHIFT are NOT covered by Scipion's inspected path")
        report["status"] = "UNRESOLVED" if required else "NOT_COVERED_BY_SCIPION"
        return report

    og = revision.original_geometry
    grid = regular_grid_points(og.raw_shape_xy, nx=5, ny=5)
    worst = "COMPATIBLE"
    order = {c: i for i, c in enumerate(
        ("COMPATIBLE", "EQUIVALENT_REPRESENTATION", "NOT_COVERED_BY_SCIPION",
         "UNRESOLVED", "INCOMPATIBLE"))}
    for i, final in enumerate(revision.final_transforms):
        ours = forward_points_pixels(grid, final.matrix, final.shift,
                                     og.raw_shape_xy, og.aligned_shape_xy, "imod")
        try:
            theirs = scipion_mapping_provider(i, grid)
        except Exception as exc:  # provider failure is diagnostic, never fatal
            report["comparisons"].append({"tilt_index": i, "status": "UNRESOLVED",
                                          "error": str(exc)})
            if order["UNRESOLVED"] > order[worst]:
                worst = "UNRESOLVED"
            continue
        cls, stats = _classify_point_agreement(ours, theirs, tol_px=tolerance_px)
        entry = {"tilt_index": i, "status": cls, **stats}
        if cls == "INCOMPATIBLE":
            entry["diagnostic"] = {
                "our_mapping_sample": ours[:3].tolist(),
                "scipion_mapping_sample": np.asarray(theirs)[:3].tolist(),
                "suspected_convention_difference":
                    "matrix order / inversion / translation scaling / tilt ordering",
                "coverage": "point-mapping comparison (not raw .xf text)",
            }
        report["comparisons"].append(entry)
        if order[cls] > order[worst]:
            worst = cls
    report["status"] = worst
    return report
