#!/usr/bin/env python3
"""Capability probing for external dependencies.

Each capability resolves to one of four states (spec §11):
  available_and_tested  -- importable/executable AND a cheap self-test passed
  available_not_tested  -- importable/executable, no self-test run
  unavailable_local     -- not present here (may be present on the cluster)
  cluster_only          -- known to be GPU/cluster-only by design

Used during project setup (to decide local-vs-cluster execution) and by the cluster
capability probe (which records the real cluster state).
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Optional

AVAILABLE_TESTED = "available_and_tested"
AVAILABLE = "available_not_tested"
UNAVAILABLE = "unavailable_local"
CLUSTER_ONLY = "cluster_only"


@dataclass
class Capability:
    name: str
    state: str
    detail: str = ""
    path: Optional[str] = None
    version: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def available(self) -> bool:
        return self.state in (AVAILABLE, AVAILABLE_TESTED)


def _module(name: str) -> Optional[str]:
    spec = importlib.util.find_spec(name)
    return spec.origin if spec else None


def probe_python_module(name: str, *, cluster_only: bool = False) -> Capability:
    try:
        origin = _module(name)
    except Exception as exc:  # importing a parent package can fail
        return Capability(name, UNAVAILABLE, detail=f"find_spec error: {exc}")
    if origin:
        return Capability(name, AVAILABLE, path=origin)
    return Capability(name, CLUSTER_ONLY if cluster_only else UNAVAILABLE,
                      detail="module not importable here")


def probe_executable(name: str, *, version_flag: Optional[str] = None,
                     cluster_only: bool = False) -> Capability:
    path = shutil.which(name)
    if not path:
        return Capability(name, CLUSTER_ONLY if cluster_only else UNAVAILABLE,
                          detail="not on PATH")
    ver = None
    if version_flag:
        try:
            cp = subprocess.run([name, version_flag], text=True, capture_output=True, timeout=20)
            ver = (cp.stdout or cp.stderr).strip().splitlines()[0] if (cp.stdout or cp.stderr) else None
        except Exception:
            ver = None
    return Capability(name, AVAILABLE, path=path, version=ver)


def probe_torch_cuda() -> Capability:
    try:
        import torch
    except Exception:
        return Capability("torch", UNAVAILABLE, detail="torch not importable")
    try:
        has = bool(torch.cuda.is_available())
        ndev = torch.cuda.device_count() if has else 0
        return Capability("torch_cuda", AVAILABLE_TESTED if has else AVAILABLE,
                          detail=f"cuda_available={has} devices={ndev}",
                          version=torch.__version__)
    except Exception as exc:
        return Capability("torch_cuda", AVAILABLE, detail=f"probe error: {exc}",
                          version=getattr(__import__('torch'), '__version__', None))


# The canonical dependency set the workflow cares about.
def probe_all(*, load_torch: bool = False) -> dict:
    caps = {
        "warpylib": probe_python_module("warpylib", cluster_only=True),
        "torch_projectors": probe_python_module("torch_projectors", cluster_only=True),
        "miss_alignment_module": probe_python_module("miss_alignment", cluster_only=True),
        "miss_alignment_exe": probe_executable("miss-alignment", cluster_only=True),
        "torch": probe_python_module("torch"),
        "torch_cuda": (
            probe_torch_cuda() if load_torch else
            Capability("torch_cuda", CLUSTER_ONLY, detail="not probed during preparation")
        ),
        "mrcfile": probe_python_module("mrcfile"),
        "newstack": probe_executable("newstack"),
        "tilt": probe_executable("tilt"),
        "ctfphaseflip": probe_executable("ctfphaseflip"),
        "sbatch": probe_executable("sbatch", cluster_only=True),
    }
    return {k: v.to_dict() for k, v in caps.items()}


def can_convert_warp() -> Capability:
    """Whether eTomo->Warp conversion can run in this process (needs warpylib)."""
    return probe_python_module("warpylib", cluster_only=True)
