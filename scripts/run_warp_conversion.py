#!/usr/bin/env python3
"""Phase-2a: convert the prepared IMOD tilt series into a Warp training directory (§9).

This is the job that POPULATES ``layout.training_dir`` so the MissAlignment smoke/full
jobs do not run against an empty directory. It consumes ONLY the canonical warp staging
manifest written by ``prepare`` (no rediscovery, §5): the condition's input stack, the
REAL source ``.xf`` (or identity for ``ali_identity``, §7), the alignment mode, axis frame,
tilt-axis angle and the TARGET volume geometry.

Memory safety (§10): the input stack is SYMLINKED into the staging dir, never read into
memory or copied byte-for-byte. ``etomo_to_warp`` reads the ``.st`` from disk.

Post-conversion validation (2.10): the source target shape is explicitly labelled
as IMOD reconstruction MRC storage order, mapped to the Warp reconstruction XYZ frame, and compared
with the produced XML ``VolumeDimensionsAngstrom``. The directory must also pass
``check_warp_dir``.

Requires warpylib (cluster only). Idempotent: if the training dir already holds a converted
XML + marker it reports success and exits 0.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def positioning_marker_current(validation: dict, manifest_positioning_hash: str) -> bool:
    """A conversion marker is positioning-current iff its recorded positioning hash
    matches the staging manifest's. A pre-contract marker records no hash (defaults to
    ``"none"``); if the manifest now carries a real positioning it is treated as stale."""
    return str(validation.get("positioning_hash", "none")) == str(manifest_positioning_hash)


def _count_rows(p: Path) -> int:
    return sum(1 for ln in p.read_text().splitlines() if ln.strip())



def _publish_v8_dataset(requested_training_dir: Path, manifest: dict) -> None:
    """Publish data/metadata when the target is a v8 ``.warp_project`` path."""
    if requested_training_dir.name != ".warp_project":
        return
    dataset_dir = requested_training_dir.parent
    if dataset_dir.parent.name != "warp_data":
        return
    project_root = dataset_dir.parent.parent
    sys.path.insert(0, str(HERE))
    from pipeline.project_publish import publish_warp_dataset
    from pipeline.runlayout import RunLayout

    layout = RunLayout.from_settings(
        out_dir=project_root,
        basename=str(manifest.get("series_name") or "series"),
        condition=str(manifest.get("condition") or "raw_xf_affine_fixed"),
        refinement_mode=str(manifest.get("refinement_mode") or "standard"),
        dataset_id=dataset_dir.name,
    ).create()
    publish_warp_dataset(layout)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert a prepared tilt series to a Warp training dir (§9).")
    ap.add_argument("--staging-manifest", required=True, type=Path)
    ap.add_argument("--training-dir", type=Path, default=None,
                    help="Override the manifest's training_dir (defaults to it).")
    ap.add_argument("--grid-shape", type=int, nargs=2, default=(5, 5), metavar=("NX", "NY"))
    ap.add_argument("--vol-tolerance", type=float, default=0.02)
    ap.add_argument("--force", action="store_true", help="Reconvert even if already done.")
    args = ap.parse_args()

    if not args.staging_manifest.is_file():
        print(f"ERROR: staging manifest not found: {args.staging_manifest}")
        return 2
    man = json.loads(args.staging_manifest.read_text())
    # Canonical IMOD tilt.com positioning carried by the staging manifest (§ positioning
    # propagation). A conversion marker written before this contract, or one whose recorded
    # positioning hash differs, is stale and must be reconverted.
    manifest_positioning_table = man.get("imod_positioning")
    manifest_positioning_hash = man.get("positioning_hash") or "none"
    # The IMOD->Warp tilt-angle sign is part of the conversion identity: a marker written with
    # a different sign (or before the sign contract existed) is stale even without positioning.
    manifest_tilt_angle_sign = int(man.get("imod_to_warp_tilt_angle_sign", -1))
    requested_training_dir = Path(args.training_dir or man["training_dir"]).absolute()
    training_dir = requested_training_dir.resolve()
    training_dir.mkdir(parents=True, exist_ok=True)

    # Idempotency is contract-aware. A legacy marker without the explicit
    # IMOD-MRC -> Warp volume-frame mapping is stale and must not be trusted.
    validation_path = training_dir / "conversion_validation.json"
    validation = {}
    if validation_path.is_file():
        try:
            validation = json.loads(validation_path.read_text())
        except Exception:
            validation = {}
    validation_frame = validation.get("volume_frame") or {}
    validation_version = int(validation.get("volume_frame_contract_version", 0))
    validation_quarter_turn_k = int(
        validation_frame.get("projection_quarter_turn_k", 0)
    ) % 4
    # Contract v1 is safe only for identity/translation-like conversions where
    # no odd detector quarter turn was applied. Contract-v1 affine conversions
    # encoded an incorrect Warp volume X/Y swap and must be regenerated.
    frame_contract_is_usable = (
        validation_version >= 2
        or (validation_version == 1 and validation_quarter_turn_k % 2 == 0)
    )
    positioning_is_current = positioning_marker_current(validation, manifest_positioning_hash)
    # A pre-contract marker records no sign; default it to the historical +1 so it is treated
    # as stale versus the -1 default and reconverted.
    sign_is_current = int(validation.get("imod_to_warp_tilt_angle_sign", 1)) == manifest_tilt_angle_sign
    # Per-view TiltAxisAngle convention: a marker made with the fixed align.com value (no version)
    # defaults to 0 and is stale versus the current per-view-.xf convention.
    from imod_affine import WARP_AXIS_ANGLE_CONVENTION_VERSION
    axis_convention_current = (
        int(validation.get("warp_axis_angle_convention_version", 0)) == WARP_AXIS_ANGLE_CONVENTION_VERSION)
    conversion_is_current = (
        (training_dir / "_converted.marker").is_file()
        and bool(list(training_dir.glob("*.xml")))
        and frame_contract_is_usable
        and bool(validation.get("warp_volume_shape_xyz"))
        and positioning_is_current
        and sign_is_current
        and axis_convention_current
    )
    if (training_dir / "_converted.marker").is_file() and not axis_convention_current:
        print(
            "[warp-conversion] tilt-axis-angle convention changed "
            f"(marker v{validation.get('warp_axis_angle_convention_version', 0)} != "
            f"v{WARP_AXIS_ANGLE_CONVENTION_VERSION}); reconverting per-view axis angles"
        )
    if (training_dir / "_converted.marker").is_file() and not sign_is_current:
        print(
            "[warp-conversion] tilt-angle sign changed "
            f"(marker={validation.get('imod_to_warp_tilt_angle_sign', 'none')} != "
            f"manifest={manifest_tilt_angle_sign}); reconverting"
        )
    if (training_dir / "_converted.marker").is_file() and not positioning_is_current:
        print(
            "[warp-conversion] positioning contract changed "
            f"(marker={validation.get('positioning_hash', 'none')[:12]} != "
            f"manifest={manifest_positioning_hash[:12]}); reconverting"
        )
    if not args.force and conversion_is_current:
        print(f"[warp-conversion] already converted with current volume-frame contract: {training_dir}")
        _publish_v8_dataset(requested_training_dir, man)
        return 0
    if (training_dir / "_converted.marker").is_file() and not conversion_is_current:
        print(
            "[warp-conversion] legacy/stale conversion marker detected; "
            "reconverting with detector-frame-only quarter-turn contract v2"
        )

    stack = Path(man["input_stack"]) if man.get("input_stack") else None
    tilt_file = Path(man["tilt_file"]) if man.get("tilt_file") else None
    if not stack or not stack.is_file():
        print(f"ERROR: input stack missing: {stack}")
        return 2
    if not tilt_file or not tilt_file.is_file():
        print(f"ERROR: tilt file missing: {tilt_file}")
        return 2

    series = man.get("series_name") or stack.stem
    condition = man["condition"]
    ts_name = f"TS_{series}_{condition}"
    ts_dir = training_dir.parent / "staging" / ts_name
    ts_dir.mkdir(parents=True, exist_ok=True)

    # §10 memory safety: link the stack, never copy the multi-GB source stack.
    st = ts_dir / f"{ts_name}.st"
    if st.is_symlink() or st.exists():
        st.unlink()
    try:
        st.symlink_to(stack.resolve())
    except OSError:
        try:
            st.hardlink_to(stack.resolve())
        except OSError as exc:
            print(f"ERROR: could not link input stack into staging dir: {exc}")
            return 2
    (ts_dir / f"{ts_name}.rawtlt").write_text(tilt_file.read_text())
    n_t = _count_rows(tilt_file)

    sys.path.insert(0, str(HERE))
    from imod_affine import read_xf, write_xf  # noqa: E402
    import numpy as np  # noqa: E402

    staged_xf = man.get("staged_xf")
    if staged_xf and not man.get("is_identity"):
        if not Path(staged_xf).is_file():
            print(f"ERROR: staged .xf missing: {staged_xf}")
            return 2
        A, d = read_xf(staged_xf)
        if len(A) != n_t:
            print(f"ERROR: staged .xf has {len(A)} rows != {n_t} tilts (§7)")
            return 2
        write_xf(ts_dir / f"{ts_name}.xf", A, d)
        write_xf(ts_dir / f"{ts_name}.source.xf", A, d)
        print(f"[warp-conversion] staged REAL source .xf ({man.get('warp_alignment_mode')})")
    else:
        write_xf(ts_dir / f"{ts_name}.xf", np.stack([np.eye(2)] * n_t), np.zeros((n_t, 2)))
        write_xf(ts_dir / f"{ts_name}.source.xf", np.stack([np.eye(2)] * n_t), np.zeros((n_t, 2)))
        print("[warp-conversion] staged identity .xf (ali_identity)")

    try:
        import etomo_to_warp as e2w
    except ModuleNotFoundError as exc:
        print(f"ERROR: warpylib/etomo_to_warp unavailable ({exc}); this job is cluster-only.")
        return 3

    target_shape_imod_mrc_xyz = tuple(man["target_volume_shape_xyz"])
    out_pix = float(man["target_pixel_size_A"])
    # Rehydrate the canonical positioning object (OFFSET/XAXISTILT/SHIFT) and apply it in
    # the converter. Absent table -> None keeps the prior no-positioning behaviour.
    positioning = None
    if manifest_positioning_table:
        from geometry.imod_positioning import from_toml_table
        positioning = from_toml_table(manifest_positioning_table)
    from geometry.imod_positioning import (
        IMOD_TO_WARP_TILT_ANGLE_SIGN, tilt_angle_convention_manifest,
        tilt_view_order_identity, validate_tilt_angle_sign)
    level_angle_x_sign = int(man.get("level_angle_x_sign", -1))
    imod_tilt_angle_sign = validate_tilt_angle_sign(
        man.get("imod_to_warp_tilt_angle_sign", IMOD_TO_WARP_TILT_ANGLE_SIGN))
    e2w.process_tilt_series(
        folder_path=ts_dir, output_directory=training_dir,
        tilt_axis_angle=float(man["tilt_axis_angle_deg"]),
        volume_shape=target_shape_imod_mrc_xyz,
        output_pixel_size=out_pix, alignment_mode=man["warp_alignment_mode"],
        axis_frame=man["axis_frame"], grid_shape_xy=tuple(args.grid_shape),
        positioning=positioning, level_angle_x_sign=level_angle_x_sign,
        imod_to_warp_tilt_angle_sign=imod_tilt_angle_sign)

    xmls = list(training_dir.glob("*.xml"))
    if not xmls:
        print(f"ERROR: conversion produced no XML in {training_dir}")
        return 1

    conversion_manifest_path = training_dir / f"{ts_name}.conversion.json"
    if not conversion_manifest_path.is_file():
        print(f"ERROR: conversion manifest missing: {conversion_manifest_path}")
        return 1
    conversion_manifest = json.loads(conversion_manifest_path.read_text())
    volume_frame = conversion_manifest.get("volume_frame") or {}
    warp_shape = tuple(
        int(v) for v in (
            volume_frame.get("reconstruction_shape_warp_xyz")
            or volume_frame.get("current_shape_warp_xyz")
            or conversion_manifest.get("warp_volume_shape_xyz")
            or []
        )
    )
    if len(warp_shape) != 3 or any(v <= 0 for v in warp_shape):
        print(
            "ERROR: conversion manifest lacks a valid Warp reconstruction XYZ shape; "
            "refusing legacy axis inference"
        )
        return 1
    conversion_pixel = float(conversion_manifest.get("output_pixel_size_A") or out_pix)

    # Physical-volume invariant in the Warp reconstruction XYZ frame. The
    # detector quarter turn is not applied to these 3-D volume extents.
    expected = [s * conversion_pixel for s in warp_shape]
    for xmlp in xmls:
        m = re.search(r'VolumeDimensionsAngstrom="([^"]+)"', xmlp.read_text())
        if m:
            vol = [float(x) for x in m.group(1).split(",")]
            for got, exp in zip(vol, expected):
                if exp and abs(got - exp) / abs(exp) > args.vol_tolerance:
                    print(f"ERROR: {xmlp.name} VolumeDimensionsAngstrom {vol} != target "
                          f"{[round(e, 1) for e in expected]} (2.10)")
                    return 1
    # reuse the existing structural validator if available
    try:
        run03 = HERE / "03_run_missalignment.py"
        if run03.is_file():
            import importlib.util
            spec = importlib.util.spec_from_file_location("run_missalignment_03", run03)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "check_warp_dir"):
                mod.check_warp_dir(training_dir)
    except Exception as exc:
        print(f"ERROR: warp dir failed structural validation: {exc}")
        return 1

    (training_dir / "_converted.marker").write_text("ok\n")
    (training_dir / "conversion_validation.json").write_text(json.dumps({
        "training_dir": str(training_dir), "ts_dir": str(ts_dir), "xml_count": len(xmls),
        "condition": condition, "alignment_mode": man["warp_alignment_mode"],
        "axis_frame": man["axis_frame"], "is_identity": bool(man.get("is_identity")),
        "schema_version": 2,
        "target_volume_shape_imod_mrc_xyz": list(target_shape_imod_mrc_xyz),
        "target_pixel_size_A": out_pix,
        "warp_volume_shape_xyz": list(warp_shape),
        "warp_volume_pixel_size_A": conversion_pixel,
        "volume_frame_contract_version": int(volume_frame.get("contract_version", 0)),
        "volume_frame": volume_frame,
        "conversion_manifest": str(conversion_manifest_path),
        "imod_positioning": manifest_positioning_table,
        "positioning_hash": manifest_positioning_hash,
        "positioning_applied": bool(positioning is not None),
        "level_angle_x_sign": level_angle_x_sign,
        "imod_to_warp_tilt_angle_sign": imod_tilt_angle_sign,
        "tilt_view_order": tilt_view_order_identity(n_t),
        "tilt_angle_convention": tilt_angle_convention_manifest(imod_tilt_angle_sign),
        # per-view TiltAxisAngle convention + hash of all final Warp axis angles (cache identity)
        "warp_axis_angle_convention_version": int(
            (conversion_manifest.get("tilt_axis_angle_provenance") or {}).get(
                "warp_axis_angle_convention_version", 0)),
        "tilt_axis_angles_hash": (conversion_manifest.get("tilt_axis_angle_provenance") or {}).get(
            "tilt_axis_angles_hash"),
        "warp_tilt_axis_angles_deg": conversion_manifest.get("warp_tilt_axis_angles_deg"),
        "volume_invariant_ok": True}, indent=2) + "\n")
    _publish_v8_dataset(requested_training_dir, man)
    print(f"[warp-conversion] OK: {len(xmls)} XML(s) in {training_dir} (validated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
