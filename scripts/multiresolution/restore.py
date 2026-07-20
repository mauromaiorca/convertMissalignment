#!/usr/bin/env python3
"""Restore a constrained residual from the working grid to the source grid.

Everything is carried in the verified ``(n-1)/2`` pixel-homogeneous convention.
The model residual is first expressed as a working-frame ``.xf`` (via the
audited ``alignment_models.serialization``), converted to a working-aligned
pixel-homogeneous matrix, then transferred to the source grid by conjugation
with ``G_a`` and composed with ``H0_source``:

    DeltaH_source  = G_a @ DeltaH_working @ inv(G_a)
    Hfinal_source  = DeltaH_source @ H0_source        (canonical order)

The exporter writes ``Hfinal_source`` as the final source ``.xf``. The restore
is the COMPLETE homogeneous transform, never a rescaled parameter vector.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imod_affine import homogeneous_to_xf, read_xf, write_xf, xf_to_homogeneous  # noqa: E402

from . import transfer as T
from .grid2d import Grid2D


def model_residual_xf_rows(model, params, working_ali_shape, working_pixel_A):
    """Working-aligned residual as IMOD .xf rows (uses (n-1)/2 via serialization)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import alignment_models as am  # noqa: F401
    from alignment_models import coordinate_frames as cf
    from alignment_models.serialization import homogeneous_to_xf_rows
    n = model.as_tensor(params).shape[0]
    centre = cf.physical_center_xy(working_ali_shape, working_pixel_A)
    H = model.homogeneous_physical(params, np.tile(centre, (n, 1))).detach().cpu().numpy()
    return homogeneous_to_xf_rows(H, working_ali_shape, working_ali_shape, working_pixel_A, working_pixel_A)


def restore_residual_to_source(
    *, model, params,
    source_raw: Grid2D, source_ali: Grid2D, working_raw: Grid2D, working_ali: Grid2D,
    source_h0_xf: Path | None,
):
    """Restore a per-tilt working residual to source-grid Hfinal and intermediates.

    Returns a dict of per-tilt matrices (lists of 3x3) and the final source
    ``.xf`` rows ``(A[N,2,2], d[N,2])``.
    """
    n = model.as_tensor(params).shape[0]
    G_r = working_raw.mapping_to(source_raw)   # working raw -> source raw
    G_a = working_ali.mapping_to(source_ali)   # working aligned -> source aligned

    # Working residual as pixel-homogeneous (n-1)/2 matrices.
    A_res, d_res = model_residual_xf_rows(model, params, working_ali.shape_xy, working_ali.pixel_size_xy_A[0])
    dH_working = [xf_to_homogeneous(A_res[i], d_res[i], working_ali.shape_xy, working_ali.shape_xy) for i in range(n)]

    # Source H0 (raw->aligned). Identity if not provided.
    if source_h0_xf is not None and Path(source_h0_xf).is_file():
        A0, d0 = read_xf(source_h0_xf)
        if len(A0) != n:
            raise ValueError(f"source .xf has {len(A0)} rows, residual has {n} tilts")
        H0_source = [xf_to_homogeneous(A0[i], d0[i], source_raw.shape_xy, source_ali.shape_xy) for i in range(n)]
    else:
        H0_source = [np.eye(3) for _ in range(n)]

    Q_wa = working_ali.Q
    Q_sa = source_ali.Q
    out = {
        "G_r": G_r.tolist(), "G_a": G_a.tolist(),
        "deltaH_working": [], "deltaH_physical": [], "deltaH_source": [],
        "h0_source": [], "hfinal_source": [],
    }
    final_A, final_d = [], []
    for i in range(n):
        dHw = dH_working[i]
        dH_phys = T.deltaH_physical(dHw, Q_wa)              # grid-independent physical residual
        dH_src = T.deltaH_source(dHw, G_a)                  # source-aligned residual
        Hf = dH_src @ H0_source[i]                          # Hfinal = DeltaH @ H0
        A_xf, d_xf = homogeneous_to_xf(Hf, source_raw.shape_xy, source_ali.shape_xy)
        final_A.append(A_xf); final_d.append(d_xf)
        out["deltaH_working"].append(dHw.tolist())
        out["deltaH_physical"].append(dH_phys.tolist())
        out["deltaH_source"].append(dH_src.tolist())
        out["h0_source"].append(H0_source[i].tolist())
        out["hfinal_source"].append(Hf.tolist())
    out["final_xf_matrices"] = np.stack(final_A).tolist()
    out["final_xf_shifts"] = np.stack(final_d).tolist()
    return out, np.stack(final_A), np.stack(final_d)


def write_source_xf(path: Path, A: np.ndarray, d: np.ndarray) -> None:
    write_xf(path, A, d)
