#!/usr/bin/env python3
"""Publish an :class:`ImodAlignmentRevision` to one user-facing IMOD export.

Physical tree (exactly this, nothing else visible)::

    exported_data/imod/<condition_id>/
        configuration/   <series>.xf .residual.xf .tlt .xtilt tilt.com newst.com
        data/            <series>.mrc -> imported raw stack (relative symlink)
        reconstruct_with_imod.sh
        manifest.json
        alignment_change_report.json / .tsv
        alignment_change_summary.txt
        scipion_compatibility.json

There is ONE physical export directory; ``missalignment/runs/<condition_id>/export/imod``
is only a compatibility symlink to it. Source protection is absolute: nothing under
``imported_data/imod`` is ever written, and the reconstruction script refuses to target it.

Pure stdlib + numpy + imod_affine; no IMOD/Scipion/WarpTools/warpylib needed to build the
tree, the reports, the manifest or the command files (only *running* the generated script
needs IMOD).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imod_affine import write_xf  # noqa: E402
from pipeline.imod_revision import (  # noqa: E402
    REVISION_CONTRACT_VERSION, Affine2D, ImodAlignmentRevision, RevisionError,
    RevisionPolicy, tilt_change_metrics,
)

WRITER_SCHEMA_VERSION = 1
VISIBLE_TREE = (
    "configuration", "data", "reconstruct_with_imod.sh", "manifest.json",
    "alignment_change_report.json", "alignment_change_report.tsv",
    "alignment_change_summary.txt", "scipion_compatibility.json",
)


# --------------------------------------------------------------------------- #
# paths + source protection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExportPaths:
    physical_dir: Path
    configuration_dir: Path
    data_dir: Path
    reconstruct_script: Path
    manifest: Path
    report_json: Path
    report_tsv: Path
    summary_txt: Path
    scipion_json: Path
    compat_link: Path

    @classmethod
    def resolve(cls, physical_dir: Path, compat_link: Path) -> "ExportPaths":
        physical_dir = Path(physical_dir)
        return cls(
            physical_dir=physical_dir,
            configuration_dir=physical_dir / "configuration",
            data_dir=physical_dir / "data",
            reconstruct_script=physical_dir / "reconstruct_with_imod.sh",
            manifest=physical_dir / "manifest.json",
            report_json=physical_dir / "alignment_change_report.json",
            report_tsv=physical_dir / "alignment_change_report.tsv",
            summary_txt=physical_dir / "alignment_change_summary.txt",
            scipion_json=physical_dir / "scipion_compatibility.json",
            compat_link=Path(compat_link),
        )


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(other).resolve())
        return True
    except (ValueError, OSError):
        return False


def assert_not_under_imported(path: Path, imported_imod_dir: Path) -> None:
    """Refuse to write anything under the read-only imported IMOD project.

    Tests where the file *lives* (parent resolved, final component kept) so a pre-existing
    data symlink whose TARGET is inside imported_data does not trip the guard on an
    idempotent re-run — only a real write location under imported_data does.
    """
    p = Path(path)
    located = p.parent.resolve() / p.name
    imported = Path(imported_imod_dir).resolve()
    try:
        located.relative_to(imported)
    except ValueError:
        return
    raise RevisionError(
        f"source protection: refusing to write under imported_data/imod ({located}); "
        "the imported project is read-only")


def _relative_symlink(link: Path, target: Path) -> None:
    """Create ``link`` -> ``target`` using a relative path where possible."""
    link = Path(link)
    target = Path(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        # Only replace an empty dir / existing symlink; never delete a populated real dir.
        if link.is_symlink() or not link.exists():
            link.unlink()
        elif link.is_dir() and not any(link.iterdir()):
            link.rmdir()
        else:
            raise RevisionError(
                f"refusing to replace non-empty path with a symlink: {link}")
    try:
        rel = os.path.relpath(target.resolve(), link.parent.resolve())
        link.symlink_to(rel, target_is_directory=target.is_dir())
    except OSError:
        link.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def _sha256_file(path: Path) -> Optional[dict]:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(p), "size": p.stat().st_size, "sha256": h.hexdigest()[:32]}


def _count_rows(path: Path) -> int:
    return sum(1 for ln in Path(path).read_text().splitlines() if ln.strip())


# --------------------------------------------------------------------------- #
# IMOD command-file editing (preserve unrelated options; change only what we must)
# --------------------------------------------------------------------------- #
def update_com_field(text: str, key: str, value: str) -> str:
    """Set ``KEY value`` on the last active occurrence of ``key`` (append if absent).

    Preserves comments, ``$program`` lines and every unrelated option. Matches IMOD/PIP
    ``KEY value`` lines case-insensitively; a trailing ``# comment`` is kept.
    """
    lines = text.splitlines()
    pat = re.compile(rf"(?i)^(\s*){re.escape(key)}(\s+)(\S.*?)(\s*#.*)?$")
    last = -1
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith("$"):
            continue
        if pat.match(raw):
            last = i
    if last >= 0:
        m = pat.match(lines[last])
        indent, sep, comment = m.group(1), m.group(2), (m.group(4) or "")
        lines[last] = f"{indent}{key}{sep}{value}{comment}"
    else:
        # insert before a trailing blank/DONE, else append
        lines.append(f"{key}\t{value}")
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def render_tilt_com(original_text: str, *, input_projections: str, output_file: str,
                    tilt_file: str, positioning: dict, policy: RevisionPolicy) -> str:
    """Revised tilt.com: change only the required fields, preserve everything else.

    Positioning follows the resolved policy: OFFSET/XAXISTILT/SHIFT/THICKNESS are
    PRESERVED (the revision changes the .xf/.tlt alignment, not the reconstruction
    positioning) unless a refined, validated value is supplied. XAXISTILT is only
    updated once its Warp sign is cluster-validated (never here).
    """
    text = original_text or "$tilt\n"
    text = update_com_field(text, "InputProjections", input_projections)
    text = update_com_field(text, "OutputFile", output_file)
    text = update_com_field(text, "TILTFILE", tilt_file)
    pos = positioning or {}
    # preserve_unless_refined: write the (unchanged) values explicitly so the revised
    # command file is self-contained and unambiguous.
    if pos.get("tilt_angle_offset_deg") is not None:
        text = update_com_field(text, "OFFSET", _fmt(pos["tilt_angle_offset_deg"]))
    if pos.get("x_axis_tilt_deg") is not None:
        text = update_com_field(text, "XAXISTILT", _fmt(pos["x_axis_tilt_deg"]))
    sx, sz = pos.get("shift_x_unbinned_px"), pos.get("shift_z_unbinned_px")
    if sx is not None or sz is not None:
        text = update_com_field(text, "SHIFT", f"{_fmt(sx or 0.0)} {_fmt(sz or 0.0)}")
    if pos.get("thickness_unbinned_px") is not None:
        text = update_com_field(text, "THICKNESS", str(int(pos["thickness_unbinned_px"])))
    return text


def render_newst_com(original_text: str, *, input_file: str, output_file: str,
                     transform_file: str) -> str:
    """Revised newst.com: reference the raw stack (via the export data link), the final
    revised .xf and a new revised aligned stack. Preserve binning/interpolation/size."""
    text = original_text or "$newstack\n"
    text = update_com_field(text, "InputFile", input_file)
    text = update_com_field(text, "OutputFile", output_file)
    text = update_com_field(text, "TransformFile", transform_file)
    return text


def _fmt(value: float) -> str:
    v = float(value)
    return str(int(v)) if v == int(v) else f"{v:g}"


# --------------------------------------------------------------------------- #
# reconstruction script
# --------------------------------------------------------------------------- #
RECONSTRUCT_TEMPLATE = r"""#!/usr/bin/env bash
set -euo pipefail

# Revised-IMOD reconstruction for series @@SERIES@@ (condition @@CONDITION_ID@@).
# Generated by convertMissAlignment; runs the REVISED newst.com + tilt.com against the
# ORIGINAL raw stack (via the data/ symlink) and the REVISED .xf/.tlt. It never writes
# under imported_data/imod and never overwrites the imported aligned stack/reconstruction.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="@@DEFAULT_OUT@@"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUT_DIR="$2"; shift 2;;
    -h|--help) echo "usage: $0 [--output-dir DIR]"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

CONFIG="$HERE/configuration"
DATA="$HERE/data"
SERIES="@@SERIES@@"
IMPORTED_IMOD="@@IMPORTED_IMOD@@"

# 1. required IMOD executables
for exe in newstack tilt; do
  command -v "$exe" >/dev/null 2>&1 || { echo "ERROR: IMOD '$exe' not on PATH" >&2; exit 3; }
done

# 2. raw-stack symlink resolves
RAW="$DATA/$SERIES.mrc"
[[ -e "$RAW" ]] || { echo "ERROR: raw-stack link does not resolve: $RAW" >&2; exit 4; }

# 3. output must not be imported_data/imod
mkdir -p "$OUT_DIR"
OUT_REAL="$(cd "$OUT_DIR" && pwd -P)"
case "$OUT_REAL/" in
  "$IMPORTED_IMOD"/*) echo "ERROR: refusing to write under imported_data/imod" >&2; exit 5;;
esac

# 4. source hashes match manifest.json
python3 - "$HERE/manifest.json" "$RAW" <<'PYEOF'
import hashlib, json, sys
manifest, raw = sys.argv[1], sys.argv[2]
m = json.load(open(manifest))
want = (m.get("source_hashes", {}) or {}).get("raw_stack", {})
if want.get("sha256"):
    h = hashlib.sha256()
    with open(raw, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    got = h.hexdigest()[:len(want["sha256"])]
    if got != want["sha256"]:
        sys.exit("ERROR: raw stack hash %s != recorded %s; imported project changed" % (got, want["sha256"]))
print("[reconstruct] source hash OK")
PYEOF

# 5. row-count checks: .xf and .tlt rows must match the stack section count
SECTIONS="$(header -size "$RAW" 2>/dev/null | awk '{print $3}')"
XF_ROWS="$(grep -cve '^[[:space:]]*$' "$CONFIG/$SERIES.xf")"
TLT_ROWS="$(grep -cve '^[[:space:]]*$' "$CONFIG/$SERIES.tlt")"
if [[ -n "${SECTIONS:-}" ]]; then
  [[ "$XF_ROWS"  == "$SECTIONS" ]] || { echo "ERROR: .xf rows $XF_ROWS != stack sections $SECTIONS" >&2; exit 6; }
  [[ "$TLT_ROWS" == "$SECTIONS" ]] || { echo "ERROR: .tlt rows $TLT_ROWS != stack sections $SECTIONS" >&2; exit 6; }
fi

ALI="$OUT_DIR/$SERIES.missalign_ali.mrc"
REC="$OUT_DIR/$SERIES.missalign_rec.mrc"

echo "[reconstruct] newstack -> $ALI"
newstack -InputFile "$RAW" -OutputFile "$ALI" -TransformFile "$CONFIG/$SERIES.xf"

echo "[reconstruct] tilt -> $REC"
# prefer the generated command file (PIP) so its options are not duplicated here
( cd "$CONFIG" && submfg tilt.com ) 2>/dev/null \
  || tilt -InputProjections "$ALI" -OutputFile "$REC" -TILTFILE "$CONFIG/$SERIES.tlt"

echo "[reconstruct] done: $REC"
"""


def render_reconstruct_script(*, series: str, condition_id: str,
                              imported_imod_dir: Path, default_out: str) -> str:
    return (RECONSTRUCT_TEMPLATE
            .replace("@@SERIES@@", series)
            .replace("@@CONDITION_ID@@", condition_id)
            .replace("@@IMPORTED_IMOD@@", str(Path(imported_imod_dir).resolve()))
            .replace("@@DEFAULT_OUT@@", default_out))


# --------------------------------------------------------------------------- #
# change report
# --------------------------------------------------------------------------- #
def _unchanged(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def build_change_report(revision: ImodAlignmentRevision, *,
                        aligned_pixel_size_A: float, raw_pixel_size_A: float) -> dict:
    """Per-tilt physical-effect report (detector-grid displacement, not just coeffs)."""
    og = revision.original_geometry
    per_tilt = []
    included = revision.refined_geometry.included
    acq = (revision.refined_geometry.acquisition_index
           or list(range(revision.n_tilts)))
    offset = float((revision.original_positioning or {}).get("tilt_angle_offset_deg", 0.0) or 0.0)
    rev_offset = float((revision.revised_positioning or {}).get(
        "tilt_angle_offset_deg", offset) or 0.0)

    disp_norms = []
    for i in range(revision.n_tilts):
        orig, final, delta = (og.original_transforms[i],
                              revision.final_transforms[i],
                              revision.residual_transforms[i])
        metrics = tilt_change_metrics(
            orig, final, delta, raw_shape_xy=og.raw_shape_xy,
            aligned_shape_xy=og.aligned_shape_xy, aligned_pixel_size_A=aligned_pixel_size_A)
        disp_norms.append(metrics["rms_displacement_px"])
        o_ang = revision.original_tilt_angles_deg[i]
        r_ang = revision.revised_tilt_angles_deg[i]
        row = {
            "tilt_index": i,
            "acquisition_index": int(acq[i]) if i < len(acq) else i,
            "included": bool(included[i]) if i < len(included) else True,
            "original_tilt_angle_deg": float(o_ang),
            "revised_tilt_angle_deg": float(r_ang),
            "original_effective_angle_deg": float(o_ang + offset),
            "revised_effective_angle_deg": float(r_ang + rev_offset),
            "delta_effective_angle_deg": float((r_ang + rev_offset) - (o_ang + offset)),
            "original_xf": [round(v, 7) for v in orig.to_row()],
            "residual_xf": [round(v, 7) for v in delta.to_row()],
            "final_xf": [round(v, 7) for v in final.to_row()],
            "representability_class": revision.representability.tilt_class[i],
            "affine_fit_rms_residual_px": round(revision.representability.rms_residual_px[i], 6),
            "affine_fit_max_residual_px": round(revision.representability.max_residual_px[i], 6),
        }
        row.update({k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in metrics.items()})
        per_tilt.append(row)

    inc = [t for t in per_tilt if t["included"]]
    modified = [t for t in per_tilt if t["rms_displacement_px"] > 1e-6]
    max_tilt = max(per_tilt, key=lambda t: t["max_displacement_px"]) if per_tilt else None
    o_eff = [t["original_effective_angle_deg"] for t in per_tilt]
    r_eff = [t["revised_effective_angle_deg"] for t in per_tilt]

    def _pos_field(table, key):
        return (table or {}).get(key)

    project = {
        "series": revision.series,
        "n_tilts": revision.n_tilts,
        "n_included": len(inc),
        "n_excluded": revision.n_tilts - len(inc),
        "n_modified": len(modified),
        "n_unchanged": revision.n_tilts - len(modified),
        "global_rms_displacement_px": float(np.sqrt(np.mean(np.square(disp_norms)))) if disp_norms else 0.0,
        "global_max_displacement_px": float(np.max([t["max_displacement_px"] for t in per_tilt])) if per_tilt else 0.0,
        "tilt_with_max_displacement": (max_tilt["tilt_index"] if max_tilt else None),
        "original_effective_angle_range_deg": [min(o_eff), max(o_eff)] if o_eff else None,
        "revised_effective_angle_range_deg": [min(r_eff), max(r_eff)] if r_eff else None,
        "pixel_sizes_A": {"raw_unbinned": raw_pixel_size_A, "aligned": aligned_pixel_size_A},
        "representability": revision.representability.to_manifest(),
        "positioning_changes": {
            field: {
                "original": _pos_field(revision.original_positioning, field),
                "revised": _pos_field(revision.revised_positioning, field) if revision.revised_positioning else _pos_field(revision.original_positioning, field),
                "unchanged": _unchanged(
                    _pos_field(revision.original_positioning, field) or 0.0,
                    (_pos_field(revision.revised_positioning, field)
                     if revision.revised_positioning else _pos_field(revision.original_positioning, field)) or 0.0),
            }
            for field in ("tilt_angle_offset_deg", "x_axis_tilt_deg",
                          "shift_x_unbinned_px", "shift_z_unbinned_px", "thickness_unbinned_px")
        },
    }
    return {"contract_version": REVISION_CONTRACT_VERSION, "project": project, "per_tilt": per_tilt}


_TSV_COLUMNS = (
    "tilt_index", "acquisition_index", "included",
    "original_tilt_angle_deg", "revised_tilt_angle_deg",
    "original_effective_angle_deg", "revised_effective_angle_deg", "delta_effective_angle_deg",
    "centre_displacement_A", "mean_displacement_A", "rms_displacement_A",
    "p95_displacement_px", "max_displacement_A",
    "residual_rotation_deg", "scale_x", "scale_y", "isotropic_scale", "shear", "determinant",
    "affine_fit_rms_residual_px", "affine_fit_max_residual_px", "representability_class",
)


def change_report_tsv(report: dict) -> str:
    rows = ["\t".join(_TSV_COLUMNS)]
    for t in report["per_tilt"]:
        vals = []
        for c in _TSV_COLUMNS:
            v = t.get(c)
            if isinstance(v, (list, tuple)):
                v = ";".join(f"{x:g}" for x in v)
            vals.append(str(v))
        rows.append("\t".join(vals))
    return "\n".join(rows) + "\n"


def change_report_summary(report: dict) -> str:
    p = report["project"]
    lines = [
        f"Revised IMOD alignment — series {p['series']}",
        "=" * 56,
        f"tilts:            {p['n_tilts']} ({p['n_included']} included, {p['n_excluded']} excluded)",
        f"modified tilts:   {p['n_modified']} ({p['n_unchanged']} unchanged)",
        f"global RMS displacement: {p['global_rms_displacement_px']:.4f} px",
        f"global max displacement: {p['global_max_displacement_px']:.4f} px"
        + (f" (tilt {p['tilt_with_max_displacement']})" if p['tilt_with_max_displacement'] is not None else ""),
        f"representability: {p['representability']['worst_class']}",
        "",
        "positioning (original -> revised):",
    ]
    for field, ch in p["positioning_changes"].items():
        tag = "unchanged" if ch["unchanged"] else "CHANGED"
        lines.append(f"  {field:24s} {ch['original']} -> {ch['revised']}   [{tag}]")
    if p["original_effective_angle_range_deg"]:
        lo, hi = p["original_effective_angle_range_deg"]
        rlo, rhi = p["revised_effective_angle_range_deg"]
        lines += ["", f"effective-angle range: [{lo:.2f}, {hi:.2f}] -> [{rlo:.2f}, {rhi:.2f}] deg"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# cache identity
# --------------------------------------------------------------------------- #
def export_cache_key(*, source_geometry_hash: str, refined_geometry_hash: str,
                     positioning_hash: str, volume_frame_contract_version: int,
                     policy: RevisionPolicy, imod_version: str = "") -> str:
    payload = {
        "writer_schema": WRITER_SCHEMA_VERSION,
        "revision_contract": REVISION_CONTRACT_VERSION,
        "source_geometry_hash": source_geometry_hash,
        "refined_geometry_hash": refined_geometry_hash,
        "positioning_hash": positioning_hash,
        "volume_frame_contract_version": int(volume_frame_contract_version),
        "policy": policy.policy_hash_fields(),
        "imod_version": imod_version,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# the writer
# --------------------------------------------------------------------------- #
def write_revision_export(
    revision: ImodAlignmentRevision, paths: ExportPaths, *,
    policy: RevisionPolicy,
    imported_imod_dir: Path,
    raw_stack_source: Path,
    original_tilt_com: str = "",
    original_newst_com: str = "",
    original_xtilt_text: Optional[str] = None,
    source_hashes: Optional[dict] = None,
    measured_pixel_size_A: Optional[float] = None,
    condition_id: str = "",
    scipion_report: Optional[dict] = None,
    extra_provenance: Optional[dict] = None,
) -> dict:
    """Write the single physical export tree + compatibility symlink; return the manifest.

    Idempotent: re-running with the same content overwrites the same files in place and
    never creates numbered/timestamped directories.
    """
    og = revision.original_geometry
    series = revision.series
    imported_imod_dir = Path(imported_imod_dir)

    # source protection: the export dir must not live under imported_data/imod
    assert_not_under_imported(paths.physical_dir, imported_imod_dir)
    paths.configuration_dir.mkdir(parents=True, exist_ok=True)
    paths.data_dir.mkdir(parents=True, exist_ok=True)

    # -- configuration/ ---------------------------------------------------
    final_xf = paths.configuration_dir / f"{series}.xf"
    residual_xf = paths.configuration_dir / f"{series}.residual.xf"
    tlt = paths.configuration_dir / f"{series}.tlt"
    xtilt = paths.configuration_dir / f"{series}.xtilt"
    tilt_com = paths.configuration_dir / "tilt.com"
    newst_com = paths.configuration_dir / "newst.com"
    for p in (final_xf, residual_xf, tlt, xtilt, tilt_com, newst_com):
        assert_not_under_imported(p, imported_imod_dir)

    write_xf(final_xf, np.stack([t.matrix for t in revision.final_transforms]),
             np.stack([t.shift for t in revision.final_transforms]))
    if policy.write_residual_xf:
        write_xf(residual_xf, np.stack([t.matrix for t in revision.residual_transforms]),
                 np.stack([t.shift for t in revision.residual_transforms]))

    tlt.write_text("".join(f"{a:.2f}\n" for a in revision.revised_tilt_angles_deg))
    tlt_unchanged = np.allclose(revision.revised_tilt_angles_deg,
                                revision.original_tilt_angles_deg, atol=1e-6)

    # .xtilt: materialise (unchanged) copy unless a refined per-view x-axis tilt exists
    refined_xtilt = revision.refined_geometry.revised_x_axis_tilt_per_view_deg
    if refined_xtilt is not None:
        xtilt.write_text("".join(f"{v:.4f}\n" for v in refined_xtilt))
        xtilt_unchanged = (og.x_axis_tilt_per_view_deg is not None
                           and np.allclose(refined_xtilt, og.x_axis_tilt_per_view_deg, atol=1e-6))
    elif og.x_axis_tilt_per_view_deg is not None:
        xtilt.write_text("".join(f"{v:.4f}\n" for v in og.x_axis_tilt_per_view_deg))
        xtilt_unchanged = True
    elif original_xtilt_text:
        xtilt.write_text(original_xtilt_text if original_xtilt_text.endswith("\n")
                         else original_xtilt_text + "\n")
        xtilt_unchanged = True
    else:
        # no per-view x-axis tilt at all: materialise zeros so absence is explicit
        xtilt.write_text("".join("0.0000\n" for _ in revision.revised_tilt_angles_deg))
        xtilt_unchanged = True

    rec_out = f"{series}.missalign_rec.mrc"
    ali_out = f"{series}.missalign_ali.mrc"
    tilt_com.write_text(render_tilt_com(
        original_tilt_com, input_projections=ali_out, output_file=rec_out,
        tilt_file=f"{series}.tlt",
        positioning=(revision.revised_positioning or revision.original_positioning),
        policy=policy))
    newst_com.write_text(render_newst_com(
        original_newst_com, input_file=f"../data/{series}.mrc",
        output_file=ali_out, transform_file=f"{series}.xf"))

    # -- data/ : relative symlink to the imported raw stack (never copied) ----
    raw_link = paths.data_dir / f"{series}.mrc"
    assert_not_under_imported(raw_link, imported_imod_dir)
    _relative_symlink(raw_link, Path(raw_stack_source))

    # -- reconstruct_with_imod.sh ----------------------------------------
    default_out = str(paths.physical_dir / "reconstruction")
    paths.reconstruct_script.write_text(render_reconstruct_script(
        series=series, condition_id=condition_id, imported_imod_dir=imported_imod_dir,
        default_out=default_out))
    paths.reconstruct_script.chmod(0o755)

    # -- reports ----------------------------------------------------------
    report = build_change_report(
        revision, aligned_pixel_size_A=og.aligned_pixel_size_A,
        raw_pixel_size_A=og.raw_pixel_size_A)
    paths.report_json.write_text(json.dumps(report, indent=2) + "\n")
    paths.report_tsv.write_text(change_report_tsv(report))
    paths.summary_txt.write_text(change_report_summary(report))
    paths.scipion_json.write_text(json.dumps(
        scipion_report or {"status": "NOT_RUN",
                           "reason": "Scipion validation not requested or unavailable"},
        indent=2) + "\n")

    # -- compatibility symlink (single physical copy) --------------------
    _relative_symlink(paths.compat_link, paths.physical_dir)

    # -- manifest ---------------------------------------------------------
    file_hashes = {name: _sha256_file(p) for name, p in {
        "final_xf": final_xf, "residual_xf": residual_xf, "tlt": tlt, "xtilt": xtilt,
        "tilt_com": tilt_com, "newst_com": newst_com,
    }.items() if p.exists()}
    manifest = {
        "schema_version": WRITER_SCHEMA_VERSION,
        "revision_contract_version": REVISION_CONTRACT_VERSION,
        "condition_id": condition_id,
        "series": series,
        "reconstruction_angpix_A": measured_pixel_size_A,
        "pixel_sizes_A": {"raw_unbinned": og.raw_pixel_size_A,
                          "aligned": og.aligned_pixel_size_A},
        "physical_export_path": str(paths.physical_dir),
        "compatibility_symlink_path": str(paths.compat_link),
        "symlink_direction": "run_level -> exported_data (physical)",
        "raw_data_symlink_target": str(Path(raw_stack_source).resolve()),
        "reconstruction_script": str(paths.reconstruct_script),
        "source_hashes": source_hashes or {},
        "geometry_contract": {
            "original_geometry_hash": (revision.provenance or {}).get("original_geometry_hash"),
            "refined_geometry_hash": (revision.provenance or {}).get("refined_geometry_hash"),
            "positioning_hash": (revision.provenance or {}).get("positioning_hash"),
            "volume_frame_contract_version": (revision.provenance or {}).get(
                "volume_frame_contract_version"),
        },
        "final_xf": file_hashes.get("final_xf"),
        "residual_xf": file_hashes.get("residual_xf"),
        "tlt": {**(file_hashes.get("tlt") or {}), "unchanged": bool(tlt_unchanged)},
        "xtilt": {**(file_hashes.get("xtilt") or {}), "unchanged": bool(xtilt_unchanged)},
        "tilt_com": file_hashes.get("tilt_com"),
        "newst_com": file_hashes.get("newst_com"),
        "representability_status": revision.representability.worst_class,
        "policy": policy.to_manifest(),
        "imod_validation_status": "PENDING" if policy.run_imod_reconstruction_validation else "SKIPPED",
        "scipion_compatibility_status": (scipion_report or {}).get("status", "NOT_RUN"),
        "creation_command": (revision.provenance or {}).get("creation_command", ""),
        "software_versions": (revision.provenance or {}).get("software_versions", {}),
        "export_cache_key": (revision.provenance or {}).get("export_cache_key"),
        "final_xf_semantics": "complete revised raw->aligned transform (H_final = DeltaH @ H_original)",
        "residual_xf_semantics": "diagnostic MissAlignment residual (DeltaH) only; NOT a complete raw->aligned transform",
    }
    if extra_provenance:
        manifest["provenance_extra"] = extra_provenance
    paths.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
