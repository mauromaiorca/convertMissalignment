#!/usr/bin/env python3
"""Export MissAlignment/warpylib offsets to an IMOD .xf file.

The exporter can operate strictly by row count/order, or match XML tilts to
IMOD rows by tilt angle.  Angle matching is intended for the common case in
which Warp/MissAlignment contains fewer projections than the original IMOD
stack because one or more tilts were excluded.

When ``--unmatched-template-policy keep-original`` is selected, unmatched IMOD
rows retain their original matrix and translation.  A TSV mapping report makes
this explicit and records every updated and retained row.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from warpylib import TiltSeries


def load_table(path: Path, minimum_columns: int, label: str) -> np.ndarray:
    try:
        data = np.loadtxt(path, dtype=float)
    except Exception as exc:
        raise SystemExit(f"ERROR: cannot read {label} {path}: {exc}") from exc
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or data.shape[1] < minimum_columns:
        raise SystemExit(
            f"ERROR: {label} must contain at least {minimum_columns} columns: {path}"
        )
    return data


def load_template_xf(path: Path) -> np.ndarray:
    return load_table(path, 6, "template .xf")[:, :6].copy()


def load_angles(path: Path) -> np.ndarray:
    try:
        values = np.loadtxt(path, dtype=float)
    except Exception as exc:
        raise SystemExit(f"ERROR: cannot read template .tlt {path}: {exc}") from exc
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise SystemExit(f"ERROR: template .tlt is empty or invalid: {path}")
    return values


def to_numpy_1d(value: object, label: str) -> np.ndarray:
    try:
        array = value.detach().cpu().float().numpy()  # type: ignore[attr-defined]
    except AttributeError:
        array = np.asarray(value, dtype=float)
    array = np.asarray(array, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise SystemExit(f"ERROR: XML field {label} is empty or contains non-finite values")
    return array


def strict_mapping(xml_angles: np.ndarray, template_angles: np.ndarray | None) -> list[tuple[int, int, float]]:
    n_xml = len(xml_angles)
    if template_angles is not None and len(template_angles) != n_xml:
        raise SystemExit(
            f"ERROR: XML has {n_xml} tilts, template .tlt has {len(template_angles)} rows"
        )
    return [
        (i, i, float(abs(xml_angles[i] - template_angles[i])) if template_angles is not None else 0.0)
        for i in range(n_xml)
    ]


def angle_mapping(
    xml_angles: np.ndarray,
    template_angles: np.ndarray,
    tolerance: float,
) -> list[tuple[int, int, float]]:
    """Return unique XML-index -> template-index matches based on tilt angle."""
    if tolerance <= 0:
        raise SystemExit("ERROR: --angle-tolerance-deg must be > 0")

    candidates: list[tuple[float, int, int]] = []
    for xml_index, xml_angle in enumerate(xml_angles):
        for template_index, template_angle in enumerate(template_angles):
            delta = abs(float(xml_angle - template_angle))
            if delta <= tolerance:
                candidates.append((delta, xml_index, template_index))

    # Select the globally closest available pairs first. Tilt-series angles are
    # normally unique and separated by much more than the configured tolerance.
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    assigned_xml: set[int] = set()
    assigned_template: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for delta, xml_index, template_index in candidates:
        if xml_index in assigned_xml or template_index in assigned_template:
            continue
        assigned_xml.add(xml_index)
        assigned_template.add(template_index)
        matches.append((xml_index, template_index, delta))

    unmatched_xml = [i for i in range(len(xml_angles)) if i not in assigned_xml]
    if unmatched_xml:
        lines = []
        for i in unmatched_xml[:20]:
            nearest = int(np.argmin(np.abs(template_angles - xml_angles[i])))
            delta = abs(float(template_angles[nearest] - xml_angles[i]))
            lines.append(
                f"  XML index {i}, angle {xml_angles[i]:.6f}: nearest IMOD index "
                f"{nearest}, angle {template_angles[nearest]:.6f}, delta {delta:.6f} deg"
            )
        detail = "\n".join(lines)
        raise SystemExit(
            "ERROR: not all XML tilts could be uniquely matched to template .tlt "
            f"within {tolerance:.6g} deg.\n{detail}"
        )

    matches.sort(key=lambda item: item[0])
    return matches


def write_xf(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in data:
            handle.write(
                f"{row[0]:12.7f}{row[1]:12.7f}"
                f"{row[2]:12.7f}{row[3]:12.7f}"
                f"{row[4]:12.3f}{row[5]:12.3f}\n"
            )


def write_mapping_report(
    path: Path,
    *,
    matches: Iterable[tuple[int, int, float]],
    xml_angles: np.ndarray,
    template_angles: np.ndarray,
    unmatched_template: list[int],
    unmatched_policy: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "status",
                "xml_index_0based",
                "xml_angle_deg",
                "imod_index_0based",
                "imod_angle_deg",
                "abs_angle_delta_deg",
                "action",
            ]
        )
        for xml_index, template_index, delta in sorted(matches, key=lambda item: item[1]):
            writer.writerow(
                [
                    "matched",
                    xml_index,
                    f"{xml_angles[xml_index]:.8f}",
                    template_index,
                    f"{template_angles[template_index]:.8f}",
                    f"{delta:.8f}",
                    "translation_replaced_from_XML",
                ]
            )
        for template_index in unmatched_template:
            writer.writerow(
                [
                    "unmatched_IMOD",
                    "",
                    "",
                    template_index,
                    f"{template_angles[template_index]:.8f}",
                    "",
                    "original_transform_retained" if unmatched_policy == "keep-original" else "error",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MissAlignment/warpylib XML offsets to IMOD .xf"
    )
    parser.add_argument("--xml", required=True, type=Path)
    parser.add_argument("--template-xf", required=True, type=Path)
    parser.add_argument("--template-tlt", type=Path, default=None)
    parser.add_argument("--out-xf", type=Path, default=None)
    parser.add_argument("--raw-pixel-size", required=True, type=float)
    parser.add_argument(
        "--tilt-count-policy",
        choices=["strict", "match-by-angle"],
        default="strict",
        help="strict requires equal counts/order; match-by-angle maps XML tilts using .tlt angles",
    )
    parser.add_argument(
        "--angle-tolerance-deg",
        type=float,
        default=0.05,
        help="maximum angle difference used only by match-by-angle; strict mode maps by row order",
    )
    parser.add_argument(
        "--unmatched-template-policy",
        choices=["error", "keep-original"],
        default="error",
        help="action for IMOD rows absent from the XML",
    )
    parser.add_argument("--mapping-report", type=Path, default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate and print the mapping without writing an .xf file",
    )
    args = parser.parse_args()

    if args.raw_pixel_size <= 0:
        raise SystemExit("ERROR: --raw-pixel-size must be > 0")
    if not args.check_only and args.out_xf is None:
        raise SystemExit("ERROR: --out-xf is required unless --check-only is used")

    template = load_template_xf(args.template_xf)
    template_angles: np.ndarray | None = None
    if args.template_tlt is not None:
        template_angles = load_angles(args.template_tlt)
        if len(template_angles) != len(template):
            raise SystemExit(
                f"ERROR: template .tlt has {len(template_angles)} rows, "
                f"template .xf has {len(template)} rows"
            )

    tilt_series = TiltSeries(args.xml)
    offset_y_A = to_numpy_1d(tilt_series.tilt_axis_offset_y, "tilt_axis_offset_y")
    offset_x_A = to_numpy_1d(tilt_series.tilt_axis_offset_x, "tilt_axis_offset_x")
    xml_angles = to_numpy_1d(tilt_series.angles, "angles")
    if not (len(offset_y_A) == len(offset_x_A) == len(xml_angles)):
        raise SystemExit(
            "ERROR: XML angle and offset arrays have inconsistent lengths: "
            f"angles={len(xml_angles)}, y={len(offset_y_A)}, x={len(offset_x_A)}"
        )

    if args.tilt_count_policy == "strict":
        if len(xml_angles) != len(template):
            raise SystemExit(
                f"ERROR: XML has {len(xml_angles)} tilts, template .xf has {len(template)} rows. "
                "Use --tilt-count-policy match-by-angle only after supplying the corresponding template .tlt."
            )
        # MissAlignment may refine tilt angles. With equal row counts, the XML
        # arrays retain stack order, so row order is the authoritative mapping.
        # Angle differences are reported below but are not used to remap rows.
        matches = strict_mapping(xml_angles, template_angles)
    else:
        if template_angles is None:
            raise SystemExit(
                "ERROR: --template-tlt is required with --tilt-count-policy match-by-angle"
            )
        matches = angle_mapping(xml_angles, template_angles, args.angle_tolerance_deg)

    matched_template = {template_index for _, template_index, _ in matches}
    unmatched_template = [i for i in range(len(template)) if i not in matched_template]
    if unmatched_template and args.unmatched_template_policy == "error":
        assert template_angles is not None
        detail = ", ".join(
            f"{i} ({template_angles[i]:.6f} deg)" for i in unmatched_template[:30]
        )
        raise SystemExit(
            "ERROR: template IMOD rows are absent from the XML: "
            f"{detail}. Set --unmatched-template-policy keep-original only if these tilts "
            "were deliberately excluded from MissAlignment."
        )

    print(f"XML:             {args.xml}")
    print(f"Template XF:     {args.template_xf}")
    if args.template_tlt is not None:
        print(f"Template TLT:    {args.template_tlt}")
    print(f"XML tilts:       {len(xml_angles)}")
    print(f"IMOD rows:       {len(template)}")
    print(f"Matched rows:    {len(matches)}")
    print(f"Mapping policy:  {args.tilt_count_policy}")
    if args.tilt_count_policy == "match-by-angle":
        print(f"Angle tolerance: {args.angle_tolerance_deg:.6g} deg")
    elif template_angles is not None:
        row_deltas = np.abs(xml_angles - template_angles)
        print("Final XML angle change relative to template .tlt (diagnostic only):")
        print(
            f"  abs delta min/mean/max: {row_deltas.min():.6f} / "
            f"{row_deltas.mean():.6f} / {row_deltas.max():.6f} deg"
        )
    if unmatched_template:
        assert template_angles is not None
        print("Unmatched IMOD rows:")
        for index in unmatched_template:
            print(
                f"  index {index:4d}, angle {template_angles[index]:10.6f} deg: "
                "original transform retained"
            )

    if args.mapping_report is not None:
        if template_angles is None:
            # For strict mode without a .tlt, provide synthetic NaNs rather than
            # claiming that angles were checked.
            template_angles = np.full(len(template), np.nan, dtype=float)
        write_mapping_report(
            args.mapping_report,
            matches=matches,
            xml_angles=xml_angles,
            template_angles=template_angles,
            unmatched_template=unmatched_template,
            unmatched_policy=args.unmatched_template_policy,
        )
        print(f"Mapping report:  {args.mapping_report}")

    if args.check_only:
        print("Mapping validation completed; no .xf file was written.")
        return

    corrected_yx_px = np.stack([offset_y_A, offset_x_A], axis=1) / args.raw_pixel_size
    inv_rot_shift_xy = -np.flip(corrected_yx_px, axis=1)

    output = template.copy()
    for xml_index, template_index, _ in matches:
        rotation = output[template_index, :4].reshape(2, 2)
        shift_xy = rotation @ inv_rot_shift_xy[xml_index]
        output[template_index, 4:6] = shift_xy

    assert args.out_xf is not None
    write_xf(args.out_xf, output)
    delta = output[:, 4:6] - template[:, 4:6]
    updated = sorted(matched_template)
    updated_delta = delta[updated]
    print(f"Output XF:       {args.out_xf}")
    print(f"Raw pixel size:  {args.raw_pixel_size:.6g} A/px")
    print("Translation delta for matched rows, pixels:")
    print(
        f"  dx min/mean/max: {updated_delta[:, 0].min():.3f} / "
        f"{updated_delta[:, 0].mean():.3f} / {updated_delta[:, 0].max():.3f}"
    )
    print(
        f"  dy min/mean/max: {updated_delta[:, 1].min():.3f} / "
        f"{updated_delta[:, 1].mean():.3f} / {updated_delta[:, 1].max():.3f}"
    )


if __name__ == "__main__":
    main()
