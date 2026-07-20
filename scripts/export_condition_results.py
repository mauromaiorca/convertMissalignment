#!/usr/bin/env python3
"""Export the latest MissAlignment XML for one condition to validated IMOD XF files.

The script derives the geometry of the *converted* Warp stack from its
``*.conversion.json`` manifest, while the output IMOD geometry is taken from
the automatically generated aligned stack whenever available.  This keeps
pixel-size changes and eTomo ``SizeToOutput`` cropping explicit.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import mrcfile

RAW_CONDITIONS = {
    "raw_identity",
    "raw_xf",
    "raw_xf_translation",
    "raw_xf_affine_fixed",
}


def latest_xml(warp_dir: Path) -> Path:
    candidates: list[tuple[int, float, Path]] = []
    for path in warp_dir.rglob("*.xml"):
        score = -1
        for part in path.relative_to(warp_dir).parts:
            match = re.fullmatch(r"iter(\d+)", part)
            if match:
                score = max(score, int(match.group(1)))
        candidates.append((score, path.stat().st_mtime, path))
    if not candidates:
        raise SystemExit(f"ERROR: no XML files found under {warp_dir}")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def mrc_header(path: Path) -> tuple[tuple[int, int], float]:
    with mrcfile.open(path, permissive=True) as handle:
        if handle.data.ndim != 3:
            raise SystemExit(f"ERROR: expected a 3-D stack: {path}")
        shape_xy = (int(handle.data.shape[2]), int(handle.data.shape[1]))
        pixel = float(handle.voxel_size.x)
    if pixel <= 0:
        raise SystemExit(f"ERROR: non-positive pixel size in {path}")
    return shape_xy, pixel


def load_conversion_manifest(warp_dir: Path) -> tuple[Path, dict[str, Any]]:
    manifests = sorted(warp_dir.glob("*.conversion.json"))
    if len(manifests) != 1:
        raise SystemExit(
            f"ERROR: expected one conversion manifest in {warp_dir}, found {len(manifests)}"
        )
    path = manifests[0]
    return path, json.loads(path.read_text())


def positive_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"ERROR: missing or invalid {label}: {value!r}") from exc
    if result <= 0:
        raise SystemExit(f"ERROR: {label} must be positive, got {result}")
    return result


def target_output_geometry(params: dict[str, Any]) -> tuple[tuple[int, int], float, str]:
    files = params.get("files", {})
    aligned = files.get("aligned_stack")
    if aligned and Path(aligned).is_file():
        shape, pixel = mrc_header(Path(aligned).resolve())
        return shape, pixel, "aligned_stack header"

    geometry = params.get("geometry", {})
    target_xyz = geometry.get("target_volume_shape_xyz")
    target_pixel = positive_float(
        geometry.get("target_output_pixel_size_A") or geometry.get("raw_pixel_size_A"),
        "target output pixel size",
    )
    if not target_xyz or len(target_xyz) < 2:
        raise SystemExit(
            "ERROR: cannot determine final IMOD X,Y dimensions. Generate ali_identity "
            "or provide target_volume_shape_xyz."
        )
    return (int(target_xyz[0]), int(target_xyz[1])), target_pixel, "target volume geometry"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--warp-dir", type=Path, required=True)
    parser.add_argument(
        "--condition", required=True, choices=sorted(RAW_CONDITIONS | {"ali_identity"})
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=None)
    parser.add_argument("--rms-tolerance-px", type=float, default=0.10)
    parser.add_argument("--max-tolerance-px", type=float, default=0.25)
    parser.add_argument("--allow-approximate", action="store_true")
    args = parser.parse_args()

    params_path = args.params.resolve()
    params = json.loads(params_path.read_text())
    config = params["conditions"][args.condition]
    warp_dir = args.warp_dir.resolve()
    manifest_path, manifest = load_conversion_manifest(warp_dir)

    image_shape_zyx = manifest.get("image_shape_zyx")
    if not image_shape_zyx or len(image_shape_zyx) != 3:
        raise SystemExit(f"ERROR: invalid image_shape_zyx in {manifest_path}")
    converted_shape_xy = (int(image_shape_zyx[2]), int(image_shape_zyx[1]))
    converted_pixel = positive_float(
        manifest.get("output_pixel_size_A"), "conversion output pixel size"
    )

    raw_stack = Path(params["files"]["raw_stack"]).resolve()
    raw_shape_xy, raw_header_pixel = mrc_header(raw_stack)
    raw_pixel = positive_float(
        params.get("geometry", {}).get("raw_pixel_size_A") or raw_header_pixel,
        "original raw pixel size",
    )

    final_shape_xy, final_pixel, final_geometry_source = target_output_geometry(params)
    aligned_stack = params.get("files", {}).get("aligned_stack")
    if aligned_stack and Path(aligned_stack).is_file():
        _, aligned_header_pixel = mrc_header(Path(aligned_stack).resolve())
    else:
        aligned_header_pixel = final_pixel
    aligned_pixel = positive_float(
        params.get("geometry", {}).get("aligned_pixel_size_A") or aligned_header_pixel,
        "original aligned pixel size",
    )

    xml = args.xml.resolve() if args.xml else latest_xml(warp_dir)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    series = params["series_name"]
    exporter = Path(__file__).resolve().parent / "warp_to_imod_affine.py"
    final_xf = out_dir / f"{series}_{args.condition}_raw_to_final.xf"
    report_json = out_dir / f"{series}_{args.condition}_affine_validation.json"
    report_tsv = out_dir / f"{series}_{args.condition}_affine_validation.tsv"

    source_frame = "raw" if args.condition in RAW_CONDITIONS else "ali"
    command = [
        sys.executable,
        str(exporter),
        "--xml", str(xml),
        "--source-frame", source_frame,
        "--input-shape", f"{converted_shape_xy[0]},{converted_shape_xy[1]}",
        "--final-shape", f"{final_shape_xy[0]},{final_shape_xy[1]}",
        "--input-pixel-size", str(converted_pixel),
        "--source-xf-pixel-size", str(raw_pixel if source_frame == "raw" else aligned_pixel),
        "--final-pixel-size", str(final_pixel),
        "--out-xf", str(final_xf),
        "--report-json", str(report_json),
        "--report-tsv", str(report_tsv),
        "--rms-tolerance-px", str(args.rms_tolerance_px),
        "--max-tolerance-px", str(args.max_tolerance_px),
    ]

    residual_xf: Path | None = None
    if source_frame == "ali":
        original_xf = Path(params["files"]["final_xf"]).resolve()
        residual_xf = out_dir / f"{series}_{args.condition}_ali_residual.xf"
        command.extend([
            "--original-xf", str(original_xf),
            "--raw-shape", f"{raw_shape_xy[0]},{raw_shape_xy[1]}",
            "--original-raw-pixel-size", str(raw_pixel),
            "--original-ali-pixel-size", str(aligned_pixel),
            "--out-residual-xf", str(residual_xf),
        ])
    if args.allow_approximate:
        command.append("--allow-approximate")

    subprocess.run(command, check=True)
    provenance = {
        "schema_version": 2,
        "condition": args.condition,
        "source_frame": source_frame,
        "xml": str(xml),
        "conversion_manifest": str(manifest_path),
        "converted_stack": manifest.get("output_stack"),
        "converted_shape_xy": list(converted_shape_xy),
        "converted_pixel_size_A": converted_pixel,
        "original_raw_stack": str(raw_stack),
        "original_raw_shape_xy": list(raw_shape_xy),
        "original_raw_pixel_size_A": raw_pixel,
        "original_aligned_stack": str(aligned_stack) if aligned_stack else None,
        "original_aligned_pixel_size_A": aligned_pixel,
        "final_shape_xy": list(final_shape_xy),
        "final_pixel_size_A": final_pixel,
        "final_geometry_source": final_geometry_source,
        "output_xf": str(final_xf),
        "residual_xf": str(residual_xf) if residual_xf else None,
        "validation_json": str(report_json),
        "validation_tsv": str(report_tsv),
        "command": command,
    }
    (out_dir / "export_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(final_xf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
