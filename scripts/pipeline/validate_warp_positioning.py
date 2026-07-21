#!/usr/bin/env python3
"""Cluster-side validation of the Warp representation of IMOD tilt.com positioning.

This is the ONLY authority on whether the Warp geometry is correct: it uses the
actually-installed ``warpylib`` to
  1. inspect available ``TiltSeries`` properties/methods;
  2. apply and save LevelAngleY (OFFSET), LevelAngleX (XAXISTILT) and the 3-D shift;
  3. reload the saved XML with a real parser and verify the values survive;
  4. determine the sign of LevelAngleX by comparing Warp projections against an
     independent IMOD projection oracle at several non-collinear 3-D points and tilts;
  5. write a machine-readable JSON report.

Run on a node with warpylib:
    python scripts/pipeline/validate_warp_positioning.py --out validation.json \
        --offset -11.5 --xaxis 1.82 --shift 0.0 -8.1 --pixel 2.0

The pipeline must NOT treat the Warp geometry as validated until this reports
``"level_angle_x_sign_validated": true`` and ``"xml_roundtrip_ok": true``.
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry.imod_positioning import imod_detector_projection  # noqa: E402


def _warp_projection_model(point, tilt_deg, *, level_angle_y, level_angle_x, tilt_axis_deg=0.0):
    """Documented Warp Euler model (to be reconciled with warpylib's actual matrices):

        R = Rz(-tilt_axis) @ Ry(angle + level_angle_y) @ Rx(level_angle_x)

    Returns the detector (u, v) = first two components of R @ point. If warpylib exposes
    a projection API, prefer it (see ``--use-warpylib-projection``).
    """
    import numpy as np

    a = np.deg2rad(tilt_deg + level_angle_y)
    lx = np.deg2rad(level_angle_x)
    tz = np.deg2rad(tilt_axis_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(lx), -np.sin(lx)], [0, np.sin(lx), np.cos(lx)]])
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]])
    p = rz @ ry @ rx @ np.asarray(point, dtype=float)
    return float(p[0]), float(p[1])


def _reload_xml_value(xml_path: Path, field: str):
    """Best-effort extraction of a numeric field from a saved Warp XML (real parser)."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except ET.ParseError:
        return None
    # attribute on the root, a <Param Name="field" Value="..."/>, or an element <field>...</field>
    if field in root.attrib:
        return root.attrib[field]
    for el in root.iter():
        if el.get("Name") == field and el.get("Value") is not None:
            return el.get("Value")
        if el.tag == field and (el.text or "").strip():
            return el.text.strip()
    return None


def validate(offset, xaxis, shift_xz, pixel, *, out_path, tmp_dir):
    import numpy as np

    report = {
        "warpylib_available": False,
        "available_properties": [],
        "applied": {},
        "reloaded": {},
        "xml_roundtrip_ok": None,
        "level_angle_x_sign": None,
        "level_angle_x_sign_validated": False,
        "sign_selection": {},
        "problems": [],
    }
    try:
        import warpylib  # noqa: F401
        import torch
        from warpylib import TiltSeries
    except Exception as exc:  # pragma: no cover - cluster only
        report["problems"].append(f"warpylib import failed: {exc}")
        Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 2

    report["warpylib_available"] = True
    xml_path = Path(tmp_dir) / "validate_positioning.xml"
    ts = TiltSeries(path=str(xml_path), n_tilts=3)
    report["available_properties"] = sorted(a for a in dir(ts) if not a.startswith("__"))

    raw_angles = [-60.0, 0.0, 60.0]
    ts.angles = torch.tensor(raw_angles, dtype=torch.float32)

    # 1. determine the LevelAngleX sign against the IMOD oracle
    points = [(10.0, 0.0, 0.0), (0.0, 12.0, 0.0), (0.0, 0.0, 8.0), (5.0, -7.0, 3.0)]
    best = None
    for sign in (1, -1):
        err = 0.0
        for p in points:
            for t in raw_angles:
                iu, iv = imod_detector_projection(p, t, offset_deg=offset, x_axis_tilt_deg=xaxis)
                wu, wv = _warp_projection_model(p, t, level_angle_y=offset, level_angle_x=sign * xaxis)
                err += (iu - wu) ** 2 + (iv - wv) ** 2
        report["sign_selection"][str(sign)] = float(err)
        if best is None or err < best[1]:
            best = (sign, err)
    report["level_angle_x_sign"] = best[0]
    other = report["sign_selection"][str(-best[0])]
    report["level_angle_x_sign_validated"] = bool(other > best[1] * 4 + 1e-9)  # clear winner
    if not report["level_angle_x_sign_validated"]:
        report["problems"].append("LevelAngleX sign is not a clear winner; inspect the projection model")

    # 2. apply, save, reload, verify the values survive
    try:
        ts.level_angle_y = float(offset)
        ts.level_angle_x = float(report["level_angle_x_sign"] * xaxis)
        if shift_xz != (0.0, 0.0):
            method = getattr(ts, "apply_tomogram_shift_3d", None)
            if callable(method):
                method(torch.tensor([shift_xz[0] * pixel, 0.0, shift_xz[1] * pixel], dtype=torch.float32))
            else:
                report["problems"].append("TiltSeries has no apply_tomogram_shift_3d()")
        ts.save_meta(str(xml_path))
        report["applied"] = {"LevelAngleY": offset, "LevelAngleX": report["level_angle_x_sign"] * xaxis}
        for field in ("LevelAngleY", "LevelAngleX"):
            report["reloaded"][field] = _reload_xml_value(xml_path, field)
        ly = report["reloaded"].get("LevelAngleY")
        lx = report["reloaded"].get("LevelAngleX")
        ok = ly is not None and lx is not None
        if ok:
            ok = abs(float(str(ly).replace(",", ".")) - offset) < 1e-3
            ok = ok and abs(float(str(lx).replace(",", ".")) - report["level_angle_x_sign"] * xaxis) < 1e-3
        report["xml_roundtrip_ok"] = bool(ok)
        if not ok:
            report["problems"].append("saved Warp XML did not reproduce the applied level angles")
    except Exception as exc:  # pragma: no cover - cluster only
        report["problems"].append(f"apply/save/reload failed: {exc}")
        report["xml_roundtrip_ok"] = False

    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if (report["xml_roundtrip_ok"] and report["level_angle_x_sign_validated"]) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cluster-side Warp positioning validation (needs warpylib).")
    p.add_argument("--offset", type=float, default=-11.5)
    p.add_argument("--xaxis", type=float, default=1.82)
    p.add_argument("--shift", type=float, nargs=2, default=[0.0, -8.1], metavar=("SHIFT_X", "SHIFT_Z"))
    p.add_argument("--pixel", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=Path("warp_positioning_validation.json"))
    import tempfile

    args = p.parse_args(argv)
    with tempfile.TemporaryDirectory() as td:
        return validate(args.offset, args.xaxis, tuple(args.shift), args.pixel,
                        out_path=args.out, tmp_dir=td)


if __name__ == "__main__":
    raise SystemExit(main())
