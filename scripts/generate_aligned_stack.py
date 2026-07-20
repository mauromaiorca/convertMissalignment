#!/usr/bin/env python3
"""Generate a local aligned IMOD stack from raw projections and an original .xf.

The script is designed for project setup: large source data remain untouched,
while the generated ``.ali`` is written inside the output project.  It also
updates the extracted parameter JSON so that the ``ali_identity`` condition is
immediately usable by the converter.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import mrcfile


def shell_module_prelude(init_script: str, module_name: str) -> str:
    lines = ["set -euo pipefail"]
    if init_script:
        lines.append(f"source {shlex.quote(init_script)}")
    else:
        lines.extend(
            [
                "if ! type module >/dev/null 2>&1; then",
                "  for f in /etc/profile /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do",
                "    if [ -r \"$f\" ]; then source \"$f\" >/dev/null 2>&1 || true; fi",
                "    type module >/dev/null 2>&1 && break",
                "  done",
                "fi",
            ]
        )
    lines.append("type module >/dev/null 2>&1 || { echo 'module command unavailable' >&2; exit 127; }")
    lines.append(f"module load {shlex.quote(module_name)}")
    return "\n".join(lines)


def run_command(
    command: list[str], *, module_mode: str, module_name: str, init_script: str,
    stdin_path: Path | None = None, cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    executable_available = shutil.which(command[0]) is not None
    if module_mode == "never":
        if not executable_available:
            raise SystemExit(
                f"ERROR: {command[0]} is not in PATH and module_mode=never. "
                "Load IMOD manually or set [external_tools].module_mode='auto'."
            )
        with stdin_path.open("r") if stdin_path else open(os.devnull, "r") as stream:
            return subprocess.run(command, text=True, stdin=stream if stdin_path else None, capture_output=True, cwd=cwd)
    if executable_available and module_mode == "auto":
        with stdin_path.open("r") if stdin_path else open(os.devnull, "r") as stream:
            return subprocess.run(command, text=True, stdin=stream if stdin_path else None, capture_output=True, cwd=cwd)

    script = shell_module_prelude(init_script, module_name)
    script += "\n" + " ".join(shlex.quote(part) for part in command)
    if stdin_path is not None:
        script += " < " + shlex.quote(str(stdin_path))
    return subprocess.run(["bash", "-lc", script], text=True, capture_output=True, cwd=cwd)


def prepare_newst_standard_input(
    source_com: Path, raw: Path, xf: Path, output: Path
) -> tuple[Path, dict[str, Any]] | None:
    """Create a local PIP input preserving eTomo newst.com geometry options."""
    if not source_com.is_file():
        return None
    lines = source_com.read_text(errors="replace").splitlines()
    command_index = None
    for index, line in enumerate(lines):
        if re.search(r"(?i)^\s*\$?newstack\b.*(?:-StandardInput|-Standard)", line):
            command_index = index
            break
    if command_index is None:
        return None
    block: list[str] = []
    for line in lines[command_index + 1 :]:
        if re.match(r"^\s*\$", line):
            break
        block.append(line)
    if not block:
        return None

    overrides = {
        "inputfile": f"InputFile {raw}",
        "outputfile": f"OutputFile {output}",
        "transformfile": f"TransformFile {xf}",
    }
    found: set[str] = set()
    rewritten: list[str] = []
    size_to_output: list[int] | None = None
    bin_by_factor: float | None = None
    for line in block:
        match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*(?:=|\s)\s*(.*?)\s*$", line)
        key = match.group(1).lower() if match else ""
        value = match.group(2) if match else ""
        if key in overrides:
            if key not in found:
                rewritten.append(overrides[key])
                found.add(key)
            continue
        if key == "sizetooutput":
            nums = [int(x) for x in re.findall(r"[-+]?\d+", value)[:2]]
            if len(nums) == 2:
                size_to_output = nums
        if key == "binbyfactor":
            nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
            if nums:
                bin_by_factor = float(nums[0])
        rewritten.append(line)
    prefix = [overrides[k] for k in ("inputfile", "outputfile", "transformfile") if k not in found]
    generated = output.with_suffix(output.suffix + ".newstack.in")
    generated.write_text("\n".join(prefix + rewritten).rstrip() + "\n")
    return generated, {
        "method": "newst.com StandardInput",
        "source_newst_com": str(source_com.resolve()),
        "generated_standard_input": str(generated.resolve()),
        "size_to_output_xy": size_to_output,
        "bin_by_factor": bin_by_factor,
    }


def mrc_header(path: Path) -> dict[str, Any]:
    with mrcfile.open(path, permissive=True) as handle:
        shape_zyx = [int(x) for x in handle.data.shape]
        voxel = [
            float(handle.voxel_size.x),
            float(handle.voxel_size.y),
            float(handle.voxel_size.z),
        ]
        return {
            "path": str(path.resolve()),
            "exists": True,
            "shape_zyx": shape_zyx,
            "header_nxyz": [shape_zyx[2], shape_zyx[1], shape_zyx[0]],
            "voxel_size_A": voxel,
            "dtype": str(handle.data.dtype),
            "generated_by": "generate_aligned_stack.py/newstack",
        }


def update_params(params_path: Path, output_stack: Path, provenance: dict[str, Any] | None = None) -> None:
    params = json.loads(params_path.read_text())
    info = mrc_header(output_stack)
    ntilts = info["shape_zyx"][0]
    final_tilt = params.get("files", {}).get("final_tilt")
    if final_tilt:
        with Path(final_tilt).open("r", errors="replace") as handle:
            n_angles = sum(1 for line in handle if line.strip())
        if n_angles != ntilts:
            raise SystemExit(
                f"ERROR: generated aligned stack has {ntilts} sections but final tilt file has {n_angles} rows"
            )

    pixel = float(info["voxel_size_A"][0])
    if pixel <= 0:
        raw_pixel = float(params["geometry"]["raw_pixel_size_A"])
        bin_factor = float((provenance or {}).get("bin_by_factor") or 1.0)
        pixel = raw_pixel * bin_factor
        info["voxel_size_A"] = [pixel, pixel, float(info["voxel_size_A"][2])]

    params.setdefault("files", {})["aligned_stack"] = str(output_stack.resolve())
    params.setdefault("mrc_headers", {})["aligned_stack"] = info
    params.setdefault("counts", {})["aligned_stack_tilts"] = ntilts
    params.setdefault("geometry", {})["aligned_pixel_size_A"] = pixel

    target_pixel = params["geometry"].get("target_output_pixel_size_A") or pixel
    target_xyz = params["geometry"].get("target_volume_shape_xyz")
    if target_xyz and target_pixel:
        physical = [float(target_xyz[i]) * float(target_pixel) for i in range(3)]
        aligned_xyz = [int(round(value / pixel)) for value in physical]
    else:
        aligned_xyz = target_xyz
    params["geometry"]["aligned_volume_shape_xyz_for_converter"] = aligned_xyz

    condition = params.setdefault("conditions", {}).setdefault("ali_identity", {})
    condition.update(
        {
            "stack": str(output_stack.resolve()),
            "tilt_file": params.get("files", {}).get("final_tilt"),
            "xf_file": "IDENTITY",
            "volume_shape_xyz": aligned_xyz,
            "output_pixel_size_A": target_pixel,
            "alignment_mode": "identity",
            "axis_frame": "aligned",
            "source_xf_file": params.get("files", {}).get("final_xf"),
            "description": "automatically generated IMOD aligned stack + identity metadata",
        }
    )
    params.setdefault("generated_inputs", {})["aligned_stack"] = {
        "path": str(output_stack.resolve()),
        "source_raw_stack": params.get("files", {}).get("raw_stack"),
        "source_xf": params.get("files", {}).get("final_xf"),
        "program": "IMOD newstack",
        "provenance": provenance or {},
    }
    params_path.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--module-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--imod-module", default="imod")
    parser.add_argument("--module-init-script", default="")
    parser.add_argument("--mode", type=int, default=2, help="MRC mode used by the direct fallback; mode 2 is float32")
    parser.add_argument("--newst-com", type=Path, default=None, help="Optional explicit eTomo newst.com.")
    parser.add_argument("--use-newst-com", action=argparse.BooleanOptionalAction, default=True, help="Preserve eTomo newstack options such as SizeToOutput whenever newst.com is available.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    params_path = args.params.resolve()
    params = json.loads(params_path.read_text())
    raw = Path(params["files"]["raw_stack"]).resolve()
    xf = Path(params["files"]["final_xf"]).resolve()
    if not raw.is_file():
        raise SystemExit(f"ERROR: raw stack not found: {raw}")
    if not xf.is_file():
        raise SystemExit(f"ERROR: transform file not found: {xf}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        print(f"Aligned stack already exists; reusing: {output}")
        update_params(params_path, output, {"method": "reused existing aligned stack"})
        return 0
    if output.exists():
        output.unlink()

    imod_dir = Path(params.get("imod_dir") or raw.parent).resolve()
    source_newst = args.newst_com.resolve() if args.newst_com else imod_dir / "newst.com"
    prepared = prepare_newst_standard_input(source_newst, raw, xf, output) if args.use_newst_com else None
    raw_info_pre = mrc_header(raw)
    raw_pixel_pre = float(raw_info_pre["voxel_size_A"][0])
    expected_aligned_pixel = params.get("geometry", {}).get("target_output_pixel_size_A")
    expected_aligned_pixel = float(expected_aligned_pixel) if expected_aligned_pixel else None
    if prepared is not None and expected_aligned_pixel and raw_pixel_pre > 0:
        _, prepared_provenance = prepared
        newst_bin = float(prepared_provenance.get("bin_by_factor") or 1.0)
        effective_pixel = raw_pixel_pre * newst_bin
        tolerance = max(1e-4, expected_aligned_pixel * 1e-3)
        if abs(effective_pixel - expected_aligned_pixel) > tolerance:
            raise SystemExit(
                "ERROR: eTomo newst.com is inconsistent with the selected raw stack: "
                f"raw pixel={raw_pixel_pre} Å, BinByFactor={newst_bin}, "
                f"implied aligned pixel={effective_pixel} Å, expected={expected_aligned_pixel} Å. "
                "This usually means newst.com was written for a different input stack/binning. "
                "Provide the matching raw stack or an explicit aligned stack rather than applying binning twice."
            )
    target_xyz = params.get("geometry", {}).get("target_volume_shape_xyz")
    direct_command = [
        "newstack", "-input", str(raw), "-output", str(output),
        "-xform", str(xf), "-mode", str(args.mode),
    ]
    if target_xyz and len(target_xyz) >= 2:
        direct_command.extend(["-size", f"{int(target_xyz[0])},{int(target_xyz[1])}"])
    attempts: list[tuple[list[str], Path | None, Path, dict[str, Any]]] = []
    if prepared is not None:
        standard_input, provenance = prepared
        attempts.append((["newstack", "-StandardInput"], standard_input, imod_dir, provenance))
    attempts.append((direct_command, None, raw.parent, {
        "method": "direct newstack fallback",
        "size_to_output_xy": [int(target_xyz[0]), int(target_xyz[1])] if target_xyz and len(target_xyz) >= 2 else None,
    }))
    attempts.append((["newstack", "-xform", str(xf), "-mode", str(args.mode), str(raw), str(output)], None, raw.parent, {"method": "positional newstack fallback"}))

    print("Generating aligned stack with IMOD newstack")
    print(f"  raw:       {raw}")
    print(f"  xf:        {xf}")
    print(f"  newst.com: {source_newst if source_newst.is_file() else 'not found; direct fallback'}")
    print(f"  output:    {output}")
    if args.dry_run:
        command, stdin_path, cwd, provenance = attempts[0]
        print("  command:", " ".join(shlex.quote(x) for x in command))
        print("  stdin:", stdin_path or "none")
        print("  cwd:", cwd)
        print("  provenance:", provenance)
        return 0

    errors: list[str] = []
    used_provenance: dict[str, Any] | None = None
    for command, stdin_path, cwd, provenance in attempts:
        if output.exists():
            output.unlink()
        result = run_command(
            command, module_mode=args.module_mode, module_name=args.imod_module,
            init_script=args.module_init_script, stdin_path=stdin_path, cwd=cwd,
        )
        if result.returncode == 0 and output.is_file():
            used_provenance = {**provenance, "command": command, "cwd": str(cwd)}
            break
        errors.append(
            f"command: {' '.join(shlex.quote(x) for x in command)}\n"
            f"stdin: {stdin_path}\ncwd: {cwd}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    else:
        raise SystemExit("ERROR: IMOD newstack failed.\n\n" + "\n\n".join(errors))

    raw_info = mrc_header(raw)
    ali_info = mrc_header(output)
    if ali_info["voxel_size_A"][0] <= 0 and raw_info["voxel_size_A"][0] > 0:
        bin_factor = float((used_provenance or {}).get("bin_by_factor") or 1.0)
        inferred_pixel = raw_info["voxel_size_A"][0] * bin_factor
        with mrcfile.open(output, mode="r+", permissive=True) as handle:
            handle.voxel_size = inferred_pixel
        ali_info = mrc_header(output)
    if expected_aligned_pixel and ali_info["voxel_size_A"][0] > 0:
        tolerance = max(1e-4, expected_aligned_pixel * 1e-3)
        if abs(ali_info["voxel_size_A"][0] - expected_aligned_pixel) > tolerance:
            raise SystemExit(
                "ERROR: generated aligned stack pixel size does not match the project geometry: "
                f"expected {expected_aligned_pixel} Å/px, got {ali_info['voxel_size_A'][0]} Å/px."
            )
    if raw_info["shape_zyx"][0] != ali_info["shape_zyx"][0]:
        raise SystemExit(
            "ERROR: aligned stack tilt count differs from raw stack: "
            f"raw={raw_info['shape_zyx'][0]}, ali={ali_info['shape_zyx'][0]}"
        )
    expected_xy = (used_provenance or {}).get("size_to_output_xy")
    if expected_xy:
        actual_xy = [ali_info["shape_zyx"][2], ali_info["shape_zyx"][1]]
        expected_xy = [int(expected_xy[0]), int(expected_xy[1])]
        if actual_xy != expected_xy:
            raise SystemExit(
                "ERROR: generated aligned stack dimensions do not match eTomo newst.com: "
                f"expected X,Y={expected_xy}, got X,Y={actual_xy}. "
                "Inspect the generated .newstack.in file and newst.com before continuing."
            )
    if used_provenance is not None:
        used_provenance["raw_header"] = raw_info
        used_provenance["aligned_header"] = ali_info
    update_params(params_path, output, used_provenance)
    print(f"Created aligned stack: {output}")
    print(f"  shape Z,Y,X: {ali_info['shape_zyx']}")
    print(f"  pixel size:  {ali_info['voxel_size_A'][0]} Å/px")
    print(f"Updated params: {params_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
