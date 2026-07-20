#!/usr/bin/env python3
"""Prepare condition-specific staging folders and convert IMOD data to Warp.

This version supports the recommended geometrically distinct conditions:

* ``raw_identity``
* ``raw_xf_translation`` (plus legacy alias ``raw_xf``)
* ``raw_xf_affine_fixed``
* ``ali_identity`` (aligned stack generated automatically when requested)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import mrcfile

CONDITIONS = [
    "raw_identity",
    "raw_xf",
    "raw_xf_translation",
    "raw_xf_affine_fixed",
    "ali_identity",
]
DEFAULT_CONDITIONS = ["raw_xf_affine_fixed", "ali_identity"]
DATA_EXTENSIONS = {".mrc", ".st", ".ali", ".rec", ".map"}


def load_params(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise SystemExit(f"ERROR: output path exists: {path}\nUse --overwrite to replace it.")
    path.mkdir(parents=True, exist_ok=True)


def ntilts_from_stack(stack_path: Path) -> int:
    with mrcfile.open(stack_path, permissive=True) as handle:
        if handle.data.ndim != 3:
            raise SystemExit(f"ERROR: expected 3-D stack: {stack_path}")
        return int(handle.data.shape[0])


def line_count(path: Path) -> int:
    with path.open("r", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def write_identity_xf(path: Path, ntilts: int) -> None:
    with path.open("w") as handle:
        for _ in range(ntilts):
            handle.write("1.0 0.0 0.0 1.0 0.0 0.0\n")


def stage_input_file(src: Path, dst: Path, *, copy_data: bool = False) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            raise SystemExit(f"ERROR: refusing to replace directory while staging: {dst}")
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in DATA_EXTENSIONS and not copy_data:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def run_generate_ali(
    params_path: Path,
    helper: Path,
    output: Path,
    module_mode: str,
    imod_module: str,
    module_init_script: str,
    overwrite: bool,
) -> None:
    command = [
        sys.executable,
        str(helper),
        "--params",
        str(params_path),
        "--output",
        str(output),
        "--module-mode",
        module_mode,
        "--imod-module",
        imod_module,
    ]
    if module_init_script:
        command.extend(["--module-init-script", module_init_script])
    if overwrite:
        command.append("--overwrite")
    print("Running:", " ".join(shlex.quote(x) for x in command))
    subprocess.run(command, check=True)


def ensure_ali_condition(
    params_path: Path,
    params: dict[str, Any],
    *,
    generate: bool,
    generator: Path,
    generated_path: Path,
    module_mode: str,
    imod_module: str,
    module_init_script: str,
    overwrite: bool,
) -> dict[str, Any]:
    condition = params["conditions"]["ali_identity"]
    stack_value = condition.get("stack")
    if stack_value and Path(stack_value).is_file():
        return params
    if not generate:
        raise SystemExit(
            "ERROR: ali_identity was requested but no aligned stack exists and automatic "
            "generation is disabled. Enable generate_aligned_stack in the TOML or supply aligned_stack."
        )
    if not generator.is_file():
        raise SystemExit(f"ERROR: aligned-stack generator missing: {generator}")
    run_generate_ali(
        params_path,
        generator,
        generated_path,
        module_mode,
        imod_module,
        module_init_script,
        overwrite,
    )
    return load_params(params_path)


def validate_condition(condition: str, config: dict[str, Any]) -> None:
    for key in ("stack", "tilt_file", "xf_file", "volume_shape_xyz", "output_pixel_size_A"):
        if config.get(key) in (None, ""):
            raise SystemExit(f"ERROR: condition {condition} is missing {key} in params JSON")


def make_staging_for_condition(
    params: dict[str, Any], condition: str, out_dir: Path, copy_files: bool
) -> tuple[Path, str, dict[str, Any]]:
    config = params["conditions"][condition]
    validate_condition(condition, config)
    series = params["series_name"]
    ts_name = f"TS_{series}_{condition}"
    condition_input_dir = out_dir / "staging" / condition
    ts_dir = condition_input_dir / ts_name
    ts_dir.mkdir(parents=True, exist_ok=True)

    stack_src = Path(config["stack"])
    tilt_src = Path(config["tilt_file"])
    if not stack_src.is_file():
        raise SystemExit(f"ERROR: stack not found for {condition}: {stack_src}")
    if not tilt_src.is_file():
        raise SystemExit(f"ERROR: tilt file not found for {condition}: {tilt_src}")

    stack_dst = ts_dir / f"{ts_name}.st"
    tilt_dst = ts_dir / f"{ts_name}.rawtlt"
    xf_dst = ts_dir / f"{ts_name}.xf"
    source_xf_dst = ts_dir / f"{ts_name}.source.xf"
    stage_input_file(stack_src, stack_dst, copy_data=copy_files)
    stage_input_file(tilt_src, tilt_dst, copy_data=True)

    ntilts = ntilts_from_stack(stack_src)
    if line_count(tilt_src) != ntilts:
        raise SystemExit(
            f"ERROR: {condition}: stack has {ntilts} tilts but tilt file has {line_count(tilt_src)} rows"
        )

    if config["xf_file"] == "IDENTITY":
        write_identity_xf(xf_dst, ntilts)
    else:
        xf_src = Path(config["xf_file"])
        if not xf_src.is_file() or line_count(xf_src) != ntilts:
            raise SystemExit(f"ERROR: invalid XF for {condition}: {xf_src}")
        stage_input_file(xf_src, xf_dst, copy_data=True)

    source_xf = config.get("source_xf_file")
    if source_xf and source_xf != "IDENTITY":
        source_path = Path(source_xf)
        if source_path.is_file():
            if line_count(source_path) != ntilts:
                raise SystemExit(
                    f"ERROR: source XF for {condition} has {line_count(source_path)} rows; expected {ntilts}"
                )
            stage_input_file(source_path, source_xf_dst, copy_data=True)

    return condition_input_dir, ts_name, config


def converter_command(
    converter: Path,
    input_dir: Path,
    output_dir: Path,
    tilt_axis: float,
    config: dict[str, Any],
    output_pixel_size: float,
    grid_shape_xy: tuple[int, int],
) -> list[str]:
    volume = config["volume_shape_xyz"]
    return [
        sys.executable,
        str(converter),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--tilt-axis-angle",
        str(float(tilt_axis)),
        "--volume-shape",
        *(str(int(x)) for x in volume),
        "--output-pixel-size",
        str(float(output_pixel_size)),
        "--alignment-mode",
        str(config.get("alignment_mode", "translation")),
        "--axis-frame",
        str(config.get("axis_frame", "raw")),
        "--movement-grid-shape",
        str(grid_shape_xy[0]),
        str(grid_shape_xy[1]),
    ]


def shell_quote(command: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--tilt-axis-angle", type=float, default=None)
    parser.add_argument("--output-pixel-size", type=float, default=None)
    parser.add_argument("--movement-grid-shape", type=int, nargs=2, default=(5, 5))
    parser.add_argument("--generate-aligned-stack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aligned-stack-output", type=Path, default=None)
    parser.add_argument("--aligned-stack-generator", type=Path, default=None)
    parser.add_argument("--module-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--imod-module", default="imod")
    parser.add_argument("--module-init-script", default="")
    args = parser.parse_args()

    params_path = args.params.resolve()
    params = load_params(params_path)
    converter = args.converter.resolve()
    if not converter.is_file():
        raise SystemExit(f"ERROR: converter not found: {converter}")

    out_dir = args.out_dir.resolve()
    ensure_clean_dir(out_dir, args.overwrite)

    if "ali_identity" in args.conditions:
        generator = (
            args.aligned_stack_generator.resolve()
            if args.aligned_stack_generator
            else converter.parent / "generate_aligned_stack.py"
        )
        generated_path = (
            args.aligned_stack_output.resolve()
            if args.aligned_stack_output
            else out_dir.parent / "generated_inputs" / f"{params['series_name']}_aligned.ali"
        )
        params = ensure_ali_condition(
            params_path,
            params,
            generate=args.generate_aligned_stack,
            generator=generator,
            generated_path=generated_path,
            module_mode=args.module_mode,
            imod_module=args.imod_module,
            module_init_script=args.module_init_script,
            overwrite=args.overwrite,
        )

    tilt_axis = args.tilt_axis_angle
    if tilt_axis is None:
        tilt_axis = params.get("geometry", {}).get("tilt_axis_angle_deg")
    if tilt_axis is None:
        raise SystemExit("ERROR: tilt-axis angle is missing")

    commands: list[list[str]] = []
    for condition in args.conditions:
        input_dir, _ts_name, config = make_staging_for_condition(
            params, condition, out_dir, args.copy
        )
        warp_dir = out_dir / f"warp_{condition}"
        warp_dir.mkdir(parents=True, exist_ok=True)
        pixel = (
            args.output_pixel_size
            if args.output_pixel_size is not None
            else config["output_pixel_size_A"]
        )
        commands.append(
            converter_command(
                converter,
                input_dir,
                warp_dir,
                float(tilt_axis),
                config,
                float(pixel),
                tuple(int(x) for x in args.movement_grid_shape),
            )
        )

    run_script = out_dir / "run_conversions.sh"
    with run_script.open("w") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for command in commands:
            handle.write(shell_quote(command) + "\n")
    run_script.chmod(0o755)

    print(f"Wrote staging and conversion script: {run_script}")
    for command in commands:
        print(shell_quote(command))
    if args.run:
        for command in commands:
            print("\nRunning:", shell_quote(command))
            subprocess.run(command, check=True)
        print("\nConversion complete.")
    else:
        print(f"\nRun conversions with: bash {run_script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
