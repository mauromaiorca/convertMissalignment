#!/usr/bin/env python3
"""
Patch IMOD/eTomo command files in a reconstruction working directory.

Current functions:
  - Patch newst.com to read the raw stack from an absolute path.
  - Patch newst.com / tilt.com to use local parameter/output files.
  - Optionally add/replace UseGPU in tilt.com.
  - Optionally add/replace arbitrary directives in newst.com or tilt.com.

Designed to be generic so more IMOD parameters can be added later without
rewriting the shell setup script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CANONICAL_KEYS = {
    "inputfile": "InputFile",
    "inputprojections": "InputProjections",
    "outputfile": "OutputFile",
    "transformfile": "TransformFile",
    "tiltfile": "TiltFile",
    "usegpu": "UseGPU",
    "binbyfactor": "BinByFactor",
}


def parse_key_value(items: Iterable[str] | None) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not items:
        return result
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid KEY=VALUE argument: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise SystemExit(f"Invalid empty key in: {item!r}")
        result[key.lower()] = value
    return result


def directive_key(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("#") or stripped.startswith("$"):
        return None
    return stripped.split(maxsplit=1)[0].lower()


def patch_standard_input_file(path: Path, replacements: Dict[str, str], append_missing: bool = True) -> None:
    """Patch IMOD StandardInput-style directive lines in a .com file.

    Existing directive keys are replaced case-insensitively. Duplicate directives
    for keys we are replacing are collapsed to one line. Comments, blank lines,
    and command launcher lines starting with '$' are preserved.
    """
    lines = path.read_text(errors="ignore").splitlines()
    seen: set[str] = set()
    output: List[str] = []

    for line in lines:
        key = directive_key(line)
        if key is not None and key in replacements:
            if key not in seen:
                canonical = CANONICAL_KEYS.get(key, line.strip().split(maxsplit=1)[0])
                output.append(f"{canonical}\t{replacements[key]}")
                seen.add(key)
            # Drop duplicate occurrences of the same replaced directive.
            continue
        output.append(line)

    if append_missing:
        for key, value in replacements.items():
            if key not in seen:
                canonical = CANONICAL_KEYS.get(key, key)
                output.append(f"{canonical}\t{value}")

    path.write_text("\n".join(output).rstrip() + "\n")


def tilt_uses_inputprojections_or_inputfile(path: Path) -> str:
    """Return which projection-input directive tilt.com currently uses/prefer."""
    keys = [directive_key(line) for line in path.read_text(errors="ignore").splitlines()]
    if "inputprojections" in keys:
        return "inputprojections"
    if "inputfile" in keys:
        return "inputfile"
    return "inputprojections"


def patch_xtilt_reference(tilt_path: Path, local_xtilt: str) -> None:
    """If tilt.com has an X-axis tilt file directive/reference, make it local.

    This is intentionally conservative: only lines containing 'xtilt' are
    modified, and comments/launcher lines are ignored.
    """
    lines = tilt_path.read_text(errors="ignore").splitlines()
    patched: List[str] = []
    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if stripped and not stripped.startswith("#") and not stripped.startswith("$") and "xtilt" in low:
            parts = stripped.split(maxsplit=1)
            key = parts[0]
            patched.append(f"{key}\t{local_xtilt}")
        else:
            patched.append(line)
    tilt_path.write_text("\n".join(patched).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch IMOD/eTomo newst.com and tilt.com files for reproducible reconstructions."
    )
    parser.add_argument("--dir", required=True, type=Path, help="Working directory containing newst.com and tilt.com")
    parser.add_argument("--basename", required=True, help="Dataset basename, e.g. lam8_ts_004")
    parser.add_argument("--raw-stack", required=True, type=Path, help="Absolute path to the raw tilt stack")

    parser.add_argument("--use-gpu", action="store_true", help="Add/replace UseGPU in tilt.com")
    parser.add_argument("--gpu-id", default="0", help="Value for UseGPU, default: 0")

    parser.add_argument("--newst-set", action="append", default=[], metavar="KEY=VALUE",
                        help="Add/replace an arbitrary directive in newst.com. Can be repeated.")
    parser.add_argument("--tilt-set", action="append", default=[], metavar="KEY=VALUE",
                        help="Add/replace an arbitrary directive in tilt.com. Can be repeated.")
    parser.add_argument("--newst-bin", type=int, default=None,
                        help="Shortcut for --newst-set BinByFactor=N")

    args = parser.parse_args()

    work_dir = args.dir.resolve()
    newst = work_dir / "newst.com"
    tilt = work_dir / "tilt.com"
    raw_stack = args.raw_stack.resolve()

    for path in (work_dir, newst, tilt, raw_stack):
        if not path.exists():
            raise SystemExit(f"ERROR: missing required path: {path}")

    base = args.basename
    local_xf = f"{base}.xf"
    local_tlt = f"{base}.tlt"
    local_xtilt = f"{base}.xtilt"
    local_ali = f"{base}_ali.mrc"
    local_rec = f"{base}.rec"

    newst_replacements: Dict[str, str] = {
        "inputfile": str(raw_stack),
        "transformfile": local_xf,
        "outputfile": local_ali,
    }
    newst_replacements.update(parse_key_value(args.newst_set))
    if args.newst_bin is not None:
        newst_replacements["binbyfactor"] = str(args.newst_bin)

    tilt_input_key = tilt_uses_inputprojections_or_inputfile(tilt)
    tilt_replacements: Dict[str, str] = {
        tilt_input_key: local_ali,
        "tiltfile": local_tlt,
        "outputfile": local_rec,
    }
    if args.use_gpu:
        tilt_replacements["usegpu"] = str(args.gpu_id)
    tilt_replacements.update(parse_key_value(args.tilt_set))

    patch_standard_input_file(newst, newst_replacements)
    patch_standard_input_file(tilt, tilt_replacements)

    if (work_dir / local_xtilt).exists():
        patch_xtilt_reference(tilt, local_xtilt)

    print(f"Patched IMOD scripts in: {work_dir}")
    print(f"  newst.com: InputFile={raw_stack}, TransformFile={local_xf}, OutputFile={local_ali}")
    print(f"  tilt.com:  {tilt_input_key}={local_ali}, TiltFile={local_tlt}, OutputFile={local_rec}")
    if args.use_gpu:
        print(f"  tilt.com:  UseGPU={args.gpu_id}")
    if args.newst_bin is not None:
        print(f"  newst.com: BinByFactor={args.newst_bin}")
    if args.newst_set:
        print(f"  newst.com extra settings: {', '.join(args.newst_set)}")
    if args.tilt_set:
        print(f"  tilt.com extra settings: {', '.join(args.tilt_set)}")


if __name__ == "__main__":
    main()
