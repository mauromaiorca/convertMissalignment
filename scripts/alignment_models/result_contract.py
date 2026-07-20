#!/usr/bin/env python3
"""Canonical constrained-result contract (spec §24).

A production constrained run writes, atomically (temp + rename):

  constrained_alignment.json   -- the canonical, deterministic result
  constrained_alignment.pt     -- the parameter tensors (torch)
  stage_history.json           -- per-stage transitions + parameter mappings
  run_manifest.json            -- environment/inputs/hashes/seed/status

A partial/failed run is marked ``completion_status != "completed"`` so finalize
can refuse it. The reader validates schema + expected model + tilt count before
the result is consumed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CANONICAL_FILES = ("constrained_alignment.json", "constrained_alignment.pt",
                   "stage_history.json", "run_manifest.json")


class ResultContractError(ValueError):
    pass


def _atomic_text(path: Path, text: str) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_constrained_result(
    out_dir: Path, *, model: str, params, tilt_angles, scopes: dict, gauge: dict,
    regularization: dict, working_raw_grid: dict | None, working_aligned_grid: dict | None,
    input_hashes: dict | None, warp_project_hash: str | None,
    loss_history: list, gradient_summary: dict | None, stage_history: list,
    software_versions: dict, cuda_info: dict | None, seed: int | None,
    start_time: str, end_time: str, completion_status: str,
    units: dict | None = None, frame: str = "aligned_physical",
    param_names: tuple | None = None, checkpoint_provenance: dict | None = None,
) -> dict:
    """Write the canonical result files atomically; return the JSON dict."""
    out_dir = Path(out_dir)
    try:
        import torch
        p = torch.as_tensor(params)
        params_list = p.detach().cpu().tolist()
        has_torch = True
    except Exception:
        params_list = [list(map(float, row)) for row in params]
        has_torch = False
    angles = [float(a) for a in (tilt_angles if tilt_angles is not None else [])]

    result = {
        "schema_version": SCHEMA_VERSION, "model": model, "frame": frame,
        "units": units or {"translation": "angstrom", "rotation": "radian", "scale": "log_scale"},
        "parameter_names": list(param_names) if param_names else None,
        "tilt_angles": angles, "n_tilts": len(params_list),
        "free_parameters": params_list, "expanded_per_tilt_parameters": params_list,
        "scopes": scopes, "gauge": gauge, "regularization": regularization,
        "working_raw_grid": working_raw_grid, "working_aligned_grid": working_aligned_grid,
        "input_hashes": input_hashes or {}, "warp_project_hash": warp_project_hash,
        "loss_history_summary": _loss_summary(loss_history),
        "gradient_summary": gradient_summary or {},
        "checkpoint_provenance": checkpoint_provenance or {},
        "software_versions": software_versions, "cuda_information": cuda_info or {},
        "random_seed": seed, "start_time": start_time, "end_time": end_time,
        "completion_status": completion_status,
    }
    _atomic_text(out_dir / "constrained_alignment.json", json.dumps(result, indent=2, default=str) + "\n")
    _atomic_text(out_dir / "stage_history.json", json.dumps(
        {"schema_version": SCHEMA_VERSION, "model": model, "stages": stage_history}, indent=2, default=str) + "\n")
    _atomic_text(out_dir / "run_manifest.json", json.dumps({
        "schema_version": SCHEMA_VERSION, "model": model, "completion_status": completion_status,
        "start_time": start_time, "end_time": end_time, "seed": seed,
        "software_versions": software_versions, "cuda_information": cuda_info or {},
        "input_hashes": input_hashes or {}, "warp_project_hash": warp_project_hash,
        "canonical_files": list(CANONICAL_FILES),
    }, indent=2, default=str) + "\n")
    # .pt (atomic): only when torch present
    if has_torch:
        import torch
        ptmp = out_dir / "constrained_alignment.pt.tmp"
        torch.save({"model": model, "params": p.detach().cpu(),
                    "tilt_angles": angles, "schema_version": SCHEMA_VERSION}, ptmp)
        os.replace(ptmp, out_dir / "constrained_alignment.pt")
    return result


def _loss_summary(loss_history: list) -> dict:
    if not loss_history:
        return {"n": 0}
    vals = [float(x) for x in loss_history]
    return {"n": len(vals), "first": vals[0], "last": vals[-1],
            "min": min(vals), "max": max(vals)}


@dataclass
class ConstrainedResultRef:
    result_dir: str
    json: dict


def read_constrained_result(result_dir: Path, *, expected_model: str | None = None,
                            expected_n_tilts: int | None = None,
                            require_completed: bool = True) -> ConstrainedResultRef:
    """Read + validate the canonical result. Raises on schema/model/tilt mismatch
    or an incomplete run. This is what finalize ``--result auto`` consumes — it
    NEVER picks the latest XML by mtime."""
    result_dir = Path(result_dir)
    jpath = result_dir / "constrained_alignment.json"
    if not jpath.is_file():
        raise ResultContractError(
            f"canonical result missing: {jpath}. finalize --result auto requires the "
            "constrained result contract; a bare MissAlignment XML is not accepted.")
    data = json.loads(jpath.read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ResultContractError(f"unsupported schema_version {data.get('schema_version')} "
                                  f"(supported {SCHEMA_VERSION})")
    if require_completed and data.get("completion_status") != "completed":
        raise ResultContractError(
            f"run not completed (status={data.get('completion_status')!r}); refusing to finalize a "
            "partial/failed result.")
    if expected_model and data.get("model") != expected_model:
        raise ResultContractError(f"model mismatch: result {data.get('model')!r} != expected {expected_model!r}")
    if expected_n_tilts is not None and data.get("n_tilts") != expected_n_tilts:
        raise ResultContractError(f"tilt-count mismatch: result {data.get('n_tilts')} != expected {expected_n_tilts}")
    for p in data.get("free_parameters", []):
        if not all(_finite(x) for x in p):
            raise ResultContractError("non-finite parameter in result")
    return ConstrainedResultRef(result_dir=str(result_dir), json=data)


def _finite(x) -> bool:
    try:
        import math
        return math.isfinite(float(x))
    except Exception:
        return False
