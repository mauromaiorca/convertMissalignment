#!/usr/bin/env python3
"""Serialization and exact IMOD ``.xf`` export for constrained models.

All four constrained models are *exactly* representable as one IMOD ``.xf`` row
per tilt (``a11 a12 a21 a22 dx dy``), because each yields a closed-form affine
matrix per tilt. The export therefore writes the exact constrained matrix, not
a fitted movement grid. The existing grid-fit exporter
(``warp_to_imod_affine.py``) is retained only as an independent verification
oracle.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from . import coordinate_frames as cf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imod_affine import write_xf  # noqa: E402


def params_to_dict(model, params, tilt_angles=None) -> dict:
    p = model.as_tensor(params).detach().cpu().numpy()
    out = {
        "model": model.name,
        "param_names": list(model.param_names),
        "n_tilts": int(p.shape[0]),
        "params": p.tolist(),
    }
    if tilt_angles is not None:
        out["tilt_angles_deg"] = [float(a) for a in np.asarray(tilt_angles).reshape(-1)]
    return out


def params_from_dict(data: dict):
    from .registry import get_model
    model = get_model(data["model"])
    import torch  # local; torch required for models
    params = torch.tensor(data["params"], dtype=model.dtype)
    return model, params


def homogeneous_to_xf_rows(
    H_phys: np.ndarray,
    in_shape_xy: Sequence[int],
    out_shape_xy: Sequence[int],
    p_in_A: float,
    p_out_A: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-tilt absolute-physical homogeneous -> IMOD ``.xf`` (A[N,2,2], d[N,2])."""
    H = np.asarray(H_phys, dtype=float)
    if H.ndim == 2:
        H = H[None]
    mats, shifts = [], []
    for h in H:
        A, d = cf.abs_physical_to_imod_xf(h, in_shape_xy, out_shape_xy, p_in_A, p_out_A)
        mats.append(A)
        shifts.append(d)
    return np.stack(mats), np.stack(shifts)


def write_xf_from_homogeneous(
    path: Path,
    H_phys: np.ndarray,
    in_shape_xy: Sequence[int],
    out_shape_xy: Sequence[int],
    p_in_A: float,
    p_out_A: float,
) -> tuple[np.ndarray, np.ndarray]:
    mats, shifts = homogeneous_to_xf_rows(H_phys, in_shape_xy, out_shape_xy, p_in_A, p_out_A)
    write_xf(path, mats, shifts)
    return mats, shifts


def write_params_json(path: Path, model, params, tilt_angles=None, extra: dict | None = None) -> None:
    data = params_to_dict(model, params, tilt_angles)
    if extra:
        data.update(extra)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
