#!/usr/bin/env python3
"""Create isolated Warp project snapshots for MissAlignment smoke and full runs."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

EXCLUDE_NAMES = {"models", "model.ckpt"}
EXCLUDE_PREFIXES = ("iter",)
EXCLUDE_SUFFIXES = ("_alignment_loss.json",)


def _copy_entry(src: Path, dst: Path) -> None:
    if src.name in EXCLUDE_NAMES or src.name.startswith(EXCLUDE_PREFIXES) or src.name.endswith(EXCLUDE_SUFFIXES):
        return
    if src.is_symlink():
        dst.symlink_to(os.readlink(src), target_is_directory=src.is_dir())
        return
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            _copy_entry(child, dst / child.name)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Tilt stacks are immutable and can be linked; metadata and XML are copied.
    if src.suffix.lower() in {".st", ".mrc", ".mrcs"} and "tiltstack" in src.parts:
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            try:
                os.link(src, dst)
                return
            except OSError:
                pass
    shutil.copy2(src, dst)


def _validate_project(path: Path) -> dict:
    xmls = sorted(path.glob("*.xml"))
    stacks = sorted((path / "tiltstack").glob("*/*.st")) if (path / "tiltstack").is_dir() else []
    if len(xmls) != 1:
        raise RuntimeError(f"expected exactly one root Warp XML in {path}, found {len(xmls)}")
    if not stacks:
        raise RuntimeError(f"no tiltstack/*/*.st found in {path}")
    if xmls[0].stat().st_size <= 0:
        raise RuntimeError(f"empty Warp XML: {xmls[0]}")
    return {"xml": str(xmls[0]), "tiltstacks": [str(p) for p in stacks]}


def clone(source: Path, target: Path) -> dict:
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        _copy_entry(child, target / child.name)
    return _validate_project(target)


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def prepare_snapshots(
    source: Path,
    pre: Path,
    smoke: Path,
    full: Path,
    manifest_path: Path,
    *,
    force: bool = False,
) -> dict:
    source = Path(source).resolve()
    if not source.is_dir():
        raise RuntimeError(f"source Warp project missing: {source}")
    source_info = _validate_project(source)

    targets = (Path(pre), Path(smoke), Path(full))
    if any(path.exists() or path.is_symlink() for path in targets):
        if not force:
            if Path(manifest_path).is_file():
                existing = json.loads(Path(manifest_path).read_text())
                if Path(existing.get("source", "")).resolve() == source:
                    for target in targets:
                        _validate_project(target)
                    return {"execution": "reused", **existing}
            raise RuntimeError(
                "MissAlignment snapshots already exist but cannot be safely reused; "
                "inspect them or rerun with --force"
            )

    pre_info = clone(source, Path(pre))
    smoke_info = clone(Path(pre), Path(smoke))
    full_info = clone(Path(pre), Path(full))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_info": source_info,
        "pre_missalign": {"path": str(pre), **pre_info},
        "smoke": {"path": str(smoke), **smoke_info},
        "full": {"path": str(full), **full_info},
        "policy": {
            "writable_xml_copied": True,
            "tilt_stacks_linked_when_possible": True,
            "smoke_and_full_share_writable_metadata": False,
        },
    }
    atomic_json(Path(manifest_path), manifest)
    return {"execution": "created", **manifest}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--pre", required=True, type=Path)
    ap.add_argument("--smoke", required=True, type=Path)
    ap.add_argument("--full", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    args = ap.parse_args()
    manifest = prepare_snapshots(
        args.source, args.pre, args.smoke, args.full, args.manifest, force=True
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
