#!/usr/bin/env python3
"""
auto_missalign_from_etomo.py

Given one eTomo/IMOD tilt-series directory, infer the main parameters needed for
MissAlignment, prepare Warp-compatible staging folders, optionally run
etomo_to_warp.py, and write MissAlignment config/run commands.

Typical use:

  conda activate missalign
  python auto_missalign_from_etomo.py \
      --etomo-dir ./lam8_ts_004 \
      --converter ./etomo_to_warp.py \
      --out-dir ./missalign_lam8_auto

The script creates up to three conditions:

  raw_xf        raw stack + final IMOD .xf alignment
  ali_identity  aligned IMOD stack + identity .xf
  raw_identity  raw stack + identity .xf

Do not use an already aligned stack together with the original .xf, because this
would apply the same alignment twice.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

try:
    import mrcfile
except Exception as exc:  # pragma: no cover
    print("ERROR: could not import mrcfile. Run: python -m pip install mrcfile", file=sys.stderr)
    raise exc


@dataclass
class MRCInfo:
    path: str
    shape_zyx: list[int]
    size_xyz: list[int]
    dtype: str
    voxel_size: list[float]
    cell_dimensions: list[float]
    origin: list[float]


@dataclass
class AutoParams:
    etomo_dir: str
    basename: str
    tilt_axis_angle: Optional[float]
    raw_stack: Optional[str]
    raw_stack_size_xyz: Optional[list[int]]
    raw_pixel_size_A: Optional[float]
    aligned_stack: Optional[str]
    aligned_stack_size_xyz: Optional[list[int]]
    aligned_pixel_size_A: Optional[float]
    final_tilt_file: Optional[str]
    final_transform_file: Optional[str]
    etomo_thickness_unbinned_px: Optional[int]
    output_pixel_size_A: Optional[float]
    target_volume_output_px_xyz: Optional[list[int]]
    raw_volume_shape_for_converter_xyz: Optional[list[int]]
    aligned_volume_shape_for_converter_xyz: Optional[list[int]]
    warnings: list[str]


TEXT_EXTENSIONS = {
    ".com", ".log", ".txt", ".mdoc", ".adoc", ".info",
    ".tlt", ".rawtlt", ".xf", ".prexf", ".prexg", ".aln",
}


TILT_AXIS_PATTERNS = [
    # IMOD .com common forms
    r"(?im)^\s*(?:-)?(?:TiltAxisAngle|TILTAXISANGLE|RotationAngle|ROTATIONANGLE)\s+([-+]?\d+(?:\.\d+)?)",
    # logs/free text
    r"(?im)tilt\s*axis(?:\s*angle)?\s*(?:=|:)?\s*([-+]?\d+(?:\.\d+)?)",
    r"(?im)axis\s*angle\s*(?:=|:)?\s*([-+]?\d+(?:\.\d+)?)",
]

THICKNESS_PATTERNS = [
    r"(?im)^\s*(?:-)?(?:THICKNESS|Thickness|thickness)\s+([0-9]+(?:\.\d+)?)",
    r"(?im)thickness\s*(?:=|:)\s*([0-9]+(?:\.\d+)?)",
]

PIXEL_SIZE_PATTERNS = [
    r"(?im)^\s*PixelSpacing\s*=\s*([0-9]+(?:\.\d+)?)",
    r"(?im)^\s*PixelSize\s*=\s*([0-9]+(?:\.\d+)?)",
    r"(?im)pixel\s*size\s*(?:=|:)\s*([0-9]+(?:\.\d+)?)",
]


EXCLUDE_STACK_KEYWORDS = (
    "_rec", "_full_rec", "_even", "_odd", "test", "proj", "_Imod",
)


def read_text(path: Path, max_bytes: int = 3_000_000) -> str:
    try:
        data = path.read_bytes()[:max_bytes]
    except Exception:
        return ""
    return data.decode("utf-8", errors="ignore")


def numeric_line_count(path: Path) -> int:
    n = 0
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # Count lines whose first token starts as a number.
            if re.match(r"^[-+]?\d", s):
                n += 1
    return n


def xf_line_count(path: Path) -> int:
    n = 0
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) >= 6 and all(re.match(r"^[-+]?\d", t) for t in toks[:6]):
                n += 1
    return n


def get_mrc_info(path: Path) -> MRCInfo:
    with mrcfile.open(path, permissive=True) as m:
        shape_zyx = [int(x) for x in m.data.shape]
        if len(shape_zyx) != 3:
            raise ValueError(f"Expected a 3D MRC stack/volume, got shape {shape_zyx}: {path}")
        size_xyz = [shape_zyx[2], shape_zyx[1], shape_zyx[0]]
        voxel = [float(m.voxel_size.x), float(m.voxel_size.y), float(m.voxel_size.z)]
        cell = [float(m.header.cella.x), float(m.header.cella.y), float(m.header.cella.z)]
        origin = [float(m.header.origin.x), float(m.header.origin.y), float(m.header.origin.z)]
        return MRCInfo(
            path=str(path),
            shape_zyx=shape_zyx,
            size_xyz=size_xyz,
            dtype=str(m.data.dtype),
            voxel_size=voxel,
            cell_dimensions=cell,
            origin=origin,
        )


def voxel_x_or_none(info: Optional[MRCInfo]) -> Optional[float]:
    if info is None:
        return None
    x = info.voxel_size[0]
    if not math.isfinite(x) or x <= 0:
        return None
    return x


def parse_float_patterns(paths: Iterable[Path], patterns: list[str]) -> list[tuple[float, Path, str]]:
    hits = []
    for p in paths:
        text = read_text(p)
        if not text:
            continue
        for pattern in patterns:
            for m in re.finditer(pattern, text):
                try:
                    value = float(m.group(1))
                except Exception:
                    continue
                line = text[max(0, m.start() - 80): m.end() + 80].replace("\n", " ")
                hits.append((value, p, line.strip()))
    return hits


def choose_tilt_axis(paths: list[Path]) -> Optional[float]:
    # Prefer .com files, then tiltalign/align logs, then any other text file.
    priority_names = ["tilt.com", "align.com", "tiltalign.log", "align.log", "taAngles.log"]
    ordered = []
    for name in priority_names:
        ordered.extend([p for p in paths if p.name == name])
    ordered.extend([p for p in paths if p not in ordered])
    hits = parse_float_patterns(ordered, TILT_AXIS_PATTERNS)
    if not hits:
        return None
    # Pick the first plausible value; IMOD values are usually within [-180, 180].
    for val, _p, _line in hits:
        if -180 <= val <= 180:
            return float(val)
    return float(hits[0][0])


def choose_thickness(paths: list[Path]) -> Optional[int]:
    # Prefer tilt.com because it contains the actual reconstruction command.
    priority_names = ["tilt.com", "tilt.log", "align.com", "align.log", "tomopitch.log"]
    ordered = []
    for name in priority_names:
        ordered.extend([p for p in paths if p.name == name])
    ordered.extend([p for p in paths if p not in ordered])
    hits = parse_float_patterns(ordered, THICKNESS_PATTERNS)
    if not hits:
        return None
    # Pick largest plausible thickness; logs often repeat values.
    values = [int(round(v)) for v, _p, _line in hits if 1 <= v < 100000]
    return max(values) if values else None


def parse_mdoc_pixel_size(etomo_dir: Path, base: str) -> Optional[float]:
    candidates = [etomo_dir / f"{base}.mrc.mdoc", *etomo_dir.glob("*.mdoc")]
    seen = set()
    ordered = []
    for p in candidates:
        if p.exists() and p not in seen:
            ordered.append(p)
            seen.add(p)
    hits = parse_float_patterns(ordered, PIXEL_SIZE_PATTERNS)
    if not hits:
        return None
    vals = [v for v, _p, _line in hits if v > 0]
    if not vals:
        return None
    # In SerialEM mdoc, PixelSpacing is normally constant; use median-like centre.
    vals_sorted = sorted(vals)
    return float(vals_sorted[len(vals_sorted) // 2])


def text_files(etomo_dir: Path) -> list[Path]:
    files = []
    for p in etomo_dir.rglob("*"):
        if p.is_file() and p.suffix in TEXT_EXTENSIONS:
            files.append(p)
    return files


def find_first_existing(candidates: list[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def choose_raw_stack(etomo_dir: Path, base: str) -> Optional[Path]:
    candidates = [etomo_dir / f"{base}.mrc", etomo_dir / f"{base}.st"]
    hit = find_first_existing(candidates)
    if hit:
        return hit
    stacks = []
    for p in list(etomo_dir.glob("*.mrc")) + list(etomo_dir.glob("*.st")):
        name = p.name.lower()
        if any(k.lower() in name for k in EXCLUDE_STACK_KEYWORDS):
            continue
        if "_ali" in name or "preali" in name:
            continue
        stacks.append(p)
    return sorted(stacks)[0] if stacks else None


def choose_aligned_stack(etomo_dir: Path, base: str) -> Optional[Path]:
    candidates = [
        etomo_dir / f"{base}_ali.mrc",
        etomo_dir / f"{base}_ali.st",
        etomo_dir / f"{base}_preali.mrc",
        etomo_dir / f"{base}_preali.st",
    ]
    return find_first_existing(candidates)


def choose_tilt_file(etomo_dir: Path, base: str) -> Optional[Path]:
    candidates = [
        etomo_dir / f"{base}.tlt",
        etomo_dir / f"{base}_fid.tlt",
        etomo_dir / f"{base}.rawtlt",
    ]
    return find_first_existing(candidates)


def choose_transform_file(etomo_dir: Path, base: str) -> Optional[Path]:
    candidates = [
        etomo_dir / f"{base}.xf",
        etomo_dir / f"{base}_fid.xf",
        etomo_dir / f"{base}.tltxf",
    ]
    return find_first_existing(candidates)


def choose_rec_volume(etomo_dir: Path, base: str) -> Optional[Path]:
    candidates = [
        etomo_dir / f"{base}_rec.mrc",
        etomo_dir / f"{base}_full_rec.mrc",
        etomo_dir / f"{base}_even_rec.mrc",
        etomo_dir / f"{base}_odd_rec.mrc",
    ]
    return find_first_existing(candidates)


def round_int(x: float) -> int:
    return int(round(float(x)))


def infer_params(etomo_dir: Path, args: argparse.Namespace) -> tuple[AutoParams, dict[str, Optional[MRCInfo]]]:
    warnings: list[str] = []
    etomo_dir = etomo_dir.resolve()
    base = args.basename or etomo_dir.name
    txts = text_files(etomo_dir)

    raw_stack = Path(args.raw_stack).resolve() if args.raw_stack else choose_raw_stack(etomo_dir, base)
    aligned_stack = Path(args.aligned_stack).resolve() if args.aligned_stack else choose_aligned_stack(etomo_dir, base)
    tilt_file = Path(args.tilt_file).resolve() if args.tilt_file else choose_tilt_file(etomo_dir, base)
    xf_file = Path(args.xf_file).resolve() if args.xf_file else choose_transform_file(etomo_dir, base)
    rec_volume = Path(args.rec_volume).resolve() if args.rec_volume else choose_rec_volume(etomo_dir, base)

    raw_info = get_mrc_info(raw_stack) if raw_stack and raw_stack.exists() else None
    aligned_info = get_mrc_info(aligned_stack) if aligned_stack and aligned_stack.exists() else None
    rec_info = get_mrc_info(rec_volume) if rec_volume and rec_volume.exists() else None

    raw_px = args.raw_pixel_size or voxel_x_or_none(raw_info) or parse_mdoc_pixel_size(etomo_dir, base)
    aligned_px = args.aligned_pixel_size or voxel_x_or_none(aligned_info)

    if raw_info is None:
        warnings.append("Could not find/read a raw stack, e.g. <basename>.mrc or <basename>.st.")
    if aligned_info is None:
        warnings.append("Could not find/read an aligned stack, e.g. <basename>_ali.mrc; ali_identity will be skipped.")
    if tilt_file is None:
        warnings.append("Could not find a tilt-angle file; expected <basename>.tlt, <basename>_fid.tlt or <basename>.rawtlt.")
    if xf_file is None:
        warnings.append("Could not find a transform file; expected <basename>.xf or <basename>_fid.xf.")

    tilt_axis = args.tilt_axis_angle if args.tilt_axis_angle is not None else choose_tilt_axis(txts)
    if tilt_axis is None:
        warnings.append("Could not infer tilt-axis angle. Pass --tilt-axis-angle explicitly.")

    thickness = args.thickness if args.thickness is not None else choose_thickness(txts)
    if thickness is None:
        warnings.append("Could not infer eTomo thickness from tilt.com/logs. Pass --thickness or --target-volume-shape.")

    # Output pixel size: use aligned stack pixel size if available; otherwise raw pixel size.
    out_px = args.output_pixel_size or aligned_px or raw_px
    if out_px is None:
        warnings.append("Could not infer output pixel size. Pass --output-pixel-size explicitly.")

    # Target volume at output pixel size.
    target_vol = None
    if args.target_volume_shape:
        target_vol = args.target_volume_shape
    elif rec_info is not None:
        target_vol = rec_info.size_xyz
    elif aligned_info is not None and raw_px and out_px and thickness:
        # aligned stack XY is usually already the output scale; Z from unbinned thickness.
        target_x, target_y = aligned_info.size_xyz[0], aligned_info.size_xyz[1]
        target_z = round_int(thickness * raw_px / out_px)
        target_vol = [target_x, target_y, target_z]
    elif raw_info is not None and raw_px and out_px and thickness:
        scale = raw_px / out_px
        target_vol = [round_int(raw_info.size_xyz[0] * scale), round_int(raw_info.size_xyz[1] * scale), round_int(thickness * scale)]

    if target_vol is None:
        warnings.append("Could not infer target volume shape. Pass --target-volume-shape X Y Z.")

    raw_vol = None
    aligned_vol = None
    if target_vol and raw_px and out_px:
        raw_vol = [round_int(v * out_px / raw_px) for v in target_vol]
    if target_vol and aligned_px and out_px:
        aligned_vol = [round_int(v * out_px / aligned_px) for v in target_vol]
    elif target_vol:
        aligned_vol = target_vol

    # Validate tilt/xf counts where possible.
    for label, info in [("raw", raw_info), ("aligned", aligned_info)]:
        if info and tilt_file and tilt_file.exists():
            nt = numeric_line_count(tilt_file)
            if nt != info.shape_zyx[0]:
                warnings.append(f"{label} stack has {info.shape_zyx[0]} slices, but tilt file has {nt} numeric lines: {tilt_file.name}")
        if info and xf_file and xf_file.exists() and label == "raw":
            nx = xf_line_count(xf_file)
            if nx != info.shape_zyx[0]:
                warnings.append(f"raw stack has {info.shape_zyx[0]} slices, but xf file has {nx} transform lines: {xf_file.name}")

    params = AutoParams(
        etomo_dir=str(etomo_dir),
        basename=base,
        tilt_axis_angle=tilt_axis,
        raw_stack=str(raw_stack) if raw_stack else None,
        raw_stack_size_xyz=raw_info.size_xyz if raw_info else None,
        raw_pixel_size_A=raw_px,
        aligned_stack=str(aligned_stack) if aligned_stack else None,
        aligned_stack_size_xyz=aligned_info.size_xyz if aligned_info else None,
        aligned_pixel_size_A=aligned_px,
        final_tilt_file=str(tilt_file) if tilt_file else None,
        final_transform_file=str(xf_file) if xf_file else None,
        etomo_thickness_unbinned_px=thickness,
        output_pixel_size_A=out_px,
        target_volume_output_px_xyz=target_vol,
        raw_volume_shape_for_converter_xyz=raw_vol,
        aligned_volume_shape_for_converter_xyz=aligned_vol,
        warnings=warnings,
    )
    infos = {"raw": raw_info, "aligned": aligned_info, "rec": rec_info}
    return params, infos


def safe_unlink_or_rmtree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output path already exists: {path}. Use --overwrite to replace it.")
        safe_unlink_or_rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def write_identity_xf(path: Path, ntilts: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for _ in range(ntilts):
            f.write("1.0 0.0 0.0 1.0 0.0 0.0\n")


def create_staging(params: AutoParams, infos: dict[str, Optional[MRCInfo]], out_dir: Path, copy_files: bool, conditions: list[str]) -> dict[str, Path]:
    staging_root = out_dir / "staging"
    staging_root.mkdir(parents=True, exist_ok=True)

    tilt_file = Path(params.final_tilt_file) if params.final_tilt_file else None
    xf_file = Path(params.final_transform_file) if params.final_transform_file else None
    raw_stack = Path(params.raw_stack) if params.raw_stack else None
    ali_stack = Path(params.aligned_stack) if params.aligned_stack else None

    created: dict[str, Path] = {}

    def make_condition(condition: str, stack: Path, transform: Optional[Path], identity: bool, ntilts: int) -> None:
        root = staging_root / condition / f"TS_{params.basename}_{condition}"
        root.mkdir(parents=True, exist_ok=True)
        prefix = f"TS_{params.basename}_{condition}"
        if tilt_file is None:
            raise RuntimeError("No tilt file available; cannot create staging data.")
        link_or_copy(tilt_file, root / f"{prefix}.rawtlt", copy_files)
        link_or_copy(stack, root / f"{prefix}.st", copy_files)
        if identity:
            write_identity_xf(root / f"{prefix}.xf", ntilts)
        else:
            if transform is None:
                raise RuntimeError("No transform file available; cannot create raw_xf staging data.")
            link_or_copy(transform, root / f"{prefix}.xf", copy_files)
        created[condition] = staging_root / condition

    if "raw_xf" in conditions and raw_stack and infos["raw"]:
        make_condition("raw_xf", raw_stack, xf_file, False, infos["raw"].shape_zyx[0])
    if "ali_identity" in conditions and ali_stack and infos["aligned"]:
        make_condition("ali_identity", ali_stack, None, True, infos["aligned"].shape_zyx[0])
    if "raw_identity" in conditions and raw_stack and infos["raw"]:
        make_condition("raw_identity", raw_stack, None, True, infos["raw"].shape_zyx[0])

    return created


def write_config(path: Path, training_directory: Path, mode: str = "realistic") -> None:
    if mode == "smoke":
        yaml = f"""general:
  training_directory: {training_directory}
  apply_ctf: False
  iteration_settings:
    - {{ downsample: 1, alignment: global }}
  seed: 45132

model_training:
  model_architecture: 'default'
  model_checkpoint: null
  loss_margin: 0.5
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  max_epochs_per_iteration: 2
  warmup_steps: 20
  multistep_lr_scheduler:
    milestones: [1]
    gamma: 0.5

data_loading:
  batch_size: 8
  patch_size: 64
  steps_per_epoch: 50

shift_generation:
  trajectory_probability: .5
  trajectory_max_shift: 4.0
  jitter_probability: .5
  jitter_max_std: 1.0
  outlier_probability: .3
  outlier_max_shift: 6.0
  fracture_probability: .3
  fracture_max_shift: 6.0

tilt_series_alignment:
  patch_size: 64
  patch_overlap: 0.25
  batch_size: 8
"""
    else:
        yaml = f"""general:
  training_directory: {training_directory}
  apply_ctf: False
  iteration_settings:
    - {{ downsample: 2, alignment: anchoring }}
    - {{ downsample: 1, alignment: global }}
    - {{ downsample: 1, alignment: global }}
  seed: 45132

model_training:
  model_architecture: 'default'
  model_checkpoint: null
  loss_margin: 0.5
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  max_epochs_per_iteration: 10
  warmup_steps: 200
  multistep_lr_scheduler:
    milestones: [5]
    gamma: 0.5

data_loading:
  batch_size: 16
  patch_size: 96
  steps_per_epoch: 500

shift_generation:
  trajectory_probability: .5
  trajectory_max_shift: 10.0
  jitter_probability: .5
  jitter_max_std: 2.0
  outlier_probability: .5
  outlier_max_shift: 20.0
  fracture_probability: .5
  fracture_max_shift: 20.0

tilt_series_alignment:
  patch_size: 96
  patch_overlap: 0.1
  batch_size: 16
"""
    path.write_text(yaml)


def run_converter(converter: Path, input_dir: Path, output_dir: Path, tilt_axis: float, volume_shape: list[int], output_pixel_size: float, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Converter output already exists: {output_dir}. Use --overwrite.")
        safe_unlink_or_rmtree(output_dir)
    cmd = [
        sys.executable,
        str(converter),
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--tilt-axis-angle", str(tilt_axis),
        "--volume-shape", str(volume_shape[0]), str(volume_shape[1]), str(volume_shape[2]),
        "--output-pixel-size", str(output_pixel_size),
    ]
    print("\nRunning converter:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def condition_volume_shape(condition: str, params: AutoParams) -> Optional[list[int]]:
    if condition in {"raw_xf", "raw_identity"}:
        return params.raw_volume_shape_for_converter_xyz
    if condition == "ali_identity":
        return params.aligned_volume_shape_for_converter_xyz
    return None


def write_reports(params: AutoParams, infos: dict[str, Optional[MRCInfo]], out_dir: Path) -> None:
    report = []
    report.append("# MissAlignment/eTomo auto-parameter report\n")
    report.append(f"eTomo directory:        {params.etomo_dir}")
    report.append(f"basename:              {params.basename}")
    report.append(f"tilt-axis angle:       {params.tilt_axis_angle} deg")
    report.append(f"raw stack:             {params.raw_stack}")
    report.append(f"raw stack size XYZ:    {params.raw_stack_size_xyz}")
    report.append(f"raw pixel size:        {params.raw_pixel_size_A} Å/px")
    report.append(f"aligned stack:         {params.aligned_stack}")
    report.append(f"aligned stack size XYZ:{params.aligned_stack_size_xyz}")
    report.append(f"aligned pixel size:    {params.aligned_pixel_size_A} Å/px")
    report.append(f"final tilt file:       {params.final_tilt_file}")
    report.append(f"final transform file:  {params.final_transform_file}")
    report.append(f"eTomo thickness:       {params.etomo_thickness_unbinned_px} unbinned px")
    report.append(f"output pixel size:     {params.output_pixel_size_A} Å/px")
    report.append(f"target volume output:  {params.target_volume_output_px_xyz} px at output pixel size")
    report.append(f"raw volume converter:  {params.raw_volume_shape_for_converter_xyz} px at raw pixel size")
    report.append(f"ali volume converter:  {params.aligned_volume_shape_for_converter_xyz} px at aligned/input pixel size")
    if params.warnings:
        report.append("\nWarnings:")
        for w in params.warnings:
            report.append(f"  - {w}")
    report.append("\nMRC headers:")
    for key, info in infos.items():
        report.append(f"\n[{key}]")
        report.append(json.dumps(asdict(info) if info else None, indent=2))

    (out_dir / "metadata_summary.txt").write_text("\n".join(report) + "\n")
    payload = asdict(params)
    payload["mrc_headers"] = {k: (asdict(v) if v else None) for k, v in infos.items()}
    (out_dir / "metadata_summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def write_command_files(params: AutoParams, out_dir: Path, converter: Path, conditions: list[str]) -> None:
    conv_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    run_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# For eTomo-converted tiltstacks from etomo_to_warp.py, do NOT use --prepare-stacks.",
        "# Disable TorchDynamo/TorchInductor to avoid Triton/compiler-wrapper failures on HPC.",
        "export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}",
        "export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}",
        "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}",
        "",
        "# Activate conda manually before running this script, e.g.:",
        "# source /path/to/conda/etc/profile.d/conda.sh",
        "# conda activate /path/to/missalignment-environment",
        "",
    ]
    for condition in conditions:
        vol = condition_volume_shape(condition, params)
        if vol is None or params.tilt_axis_angle is None or params.output_pixel_size_A is None:
            continue
        input_dir = out_dir / "staging" / condition
        warp_dir = out_dir / f"warp_{condition}"
        conv_lines.append(
            f"{sys.executable} {converter} --input-dir {input_dir} --output-dir {warp_dir} "
            f"--tilt-axis-angle {params.tilt_axis_angle} --volume-shape {vol[0]} {vol[1]} {vol[2]} "
            f"--output-pixel-size {params.output_pixel_size_A}"
        )
        conv_lines.append("")
        run_lines.append(
            f"miss-alignment --config-file {warp_dir / 'config.yaml'} "
            f"--training-devices 0 --reconstruction-devices 0 "
            f"--dataloaders-per-trainer 1 --start-at-iteration 0 "
            f"2>&1 | tee {warp_dir / 'missalignment.log'}"
        )
        run_lines.append("")
    conv = out_dir / "run_conversions.sh"
    run = out_dir / "run_missalignment.sh"
    conv.write_text("\n".join(conv_lines) + "\n")
    run.write_text("\n".join(run_lines) + "\n")
    conv.chmod(0o755)
    run.chmod(0o755)


def parse_conditions(s: str) -> list[str]:
    allowed = {"raw_xf", "ali_identity", "raw_identity"}
    values = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in values if x not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(f"Unknown condition(s): {bad}. Allowed: {sorted(allowed)}")
    return values


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--etomo-dir", required=True, type=Path, help="Input eTomo/IMOD directory, e.g. ./lam8_ts_004")
    p.add_argument("--converter", default=Path("./etomo_to_warp.py"), type=Path, help="Path to MissAlignment examples/etomo_to_warp.py")
    p.add_argument("--out-dir", default=None, type=Path, help="Output project directory. Default: missalign_auto_<basename>")
    p.add_argument("--basename", default=None, help="Tilt-series basename. Default: name of --etomo-dir")
    p.add_argument("--conditions", type=parse_conditions, default=parse_conditions("raw_xf,ali_identity,raw_identity"), help="Comma-separated conditions to prepare")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output directories")
    p.add_argument("--copy", action="store_true", help="Copy stacks instead of symlinking them")
    p.add_argument("--no-convert", action="store_true", help="Only infer parameters and create staging/commands; do not run etomo_to_warp.py")
    p.add_argument("--config-mode", choices=["smoke", "realistic"], default="realistic", help="Type of MissAlignment config to write")

    # Optional overrides.
    p.add_argument("--tilt-axis-angle", type=float, default=None)
    p.add_argument("--thickness", type=int, default=None, help="eTomo thickness in raw/unbinned pixels")
    p.add_argument("--output-pixel-size", type=float, default=None, help="Output pixel size in Å/px")
    p.add_argument("--raw-pixel-size", type=float, default=None, help="Override raw stack pixel size in Å/px")
    p.add_argument("--aligned-pixel-size", type=float, default=None, help="Override aligned stack pixel size in Å/px")
    p.add_argument("--target-volume-shape", nargs=3, type=int, metavar=("X", "Y", "Z"), default=None, help="Target output volume in pixels at output pixel size")
    p.add_argument("--raw-stack", type=Path, default=None, help="Override raw stack path")
    p.add_argument("--aligned-stack", type=Path, default=None, help="Override aligned stack path")
    p.add_argument("--tilt-file", type=Path, default=None, help="Override final tilt-angle file")
    p.add_argument("--xf-file", type=Path, default=None, help="Override final transform file")
    p.add_argument("--rec-volume", type=Path, default=None, help="Override reconstruction volume used for target shape")

    args = p.parse_args()

    etomo_dir = args.etomo_dir.resolve()
    if not etomo_dir.exists():
        raise FileNotFoundError(etomo_dir)
    base = args.basename or etomo_dir.name
    out_dir = args.out_dir or Path(f"missalign_auto_{base}")
    out_dir = out_dir.resolve()
    ensure_empty_dir(out_dir, args.overwrite)

    params, infos = infer_params(etomo_dir, args)
    write_reports(params, infos, out_dir)

    print("\nInferred parameters:")
    print((out_dir / "metadata_summary.txt").read_text())

    missing_fatal = []
    if params.tilt_axis_angle is None:
        missing_fatal.append("tilt-axis angle")
    if params.output_pixel_size_A is None:
        missing_fatal.append("output pixel size")
    if params.final_tilt_file is None:
        missing_fatal.append("tilt file")
    if "raw_xf" in args.conditions and params.final_transform_file is None:
        missing_fatal.append("transform .xf file for raw_xf")
    if missing_fatal:
        raise RuntimeError("Missing required values: " + ", ".join(missing_fatal))

    created = create_staging(params, infos, out_dir, args.copy, args.conditions)

    converter = args.converter.resolve()
    if not converter.exists() and not args.no_convert:
        raise FileNotFoundError(f"Converter not found: {converter}. Use --converter or --no-convert.")

    converted_conditions = []
    if not args.no_convert:
        for condition, input_root in created.items():
            vol = condition_volume_shape(condition, params)
            if vol is None:
                print(f"Skipping {condition}: could not infer converter volume shape.")
                continue
            warp_dir = out_dir / f"warp_{condition}"
            run_converter(
                converter=converter,
                input_dir=input_root,
                output_dir=warp_dir,
                tilt_axis=float(params.tilt_axis_angle),
                volume_shape=vol,
                output_pixel_size=float(params.output_pixel_size_A),
                overwrite=args.overwrite,
            )
            write_config(warp_dir / "config.yaml", warp_dir.resolve(), mode=args.config_mode)
            converted_conditions.append(condition)
    else:
        print("\n--no-convert set: staging created, but converter was not run.")

    # If not converted, still write configs into intended output dirs after user runs conversion.
    for condition in created:
        warp_dir = out_dir / f"warp_{condition}"
        if warp_dir.exists() and not (warp_dir / "config.yaml").exists():
            write_config(warp_dir / "config.yaml", warp_dir.resolve(), mode=args.config_mode)

    write_command_files(params, out_dir, converter, list(created.keys()))

    print("\nDone.")
    print(f"Project directory: {out_dir}")
    print(f"Metadata report:   {out_dir / 'metadata_summary.txt'}")
    print(f"Metadata JSON:     {out_dir / 'metadata_summary.json'}")
    print(f"Conversion cmds:   {out_dir / 'run_conversions.sh'}")
    print(f"MissAlign cmds:    {out_dir / 'run_missalignment.sh'}")
    if converted_conditions:
        print("Converted conditions:", ", ".join(converted_conditions))
    print("\nTo run MissAlignment after conversion:")
    print(f"  bash {out_dir / 'run_missalignment.sh'}")


if __name__ == "__main__":
    main()
