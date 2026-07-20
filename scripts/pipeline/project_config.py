#!/usr/bin/env python3
"""The single canonical project configuration: one schema, one loader.

Resolves defect 2.1 (two dialects), 2.4 (xtilt vs tltxf), 2.7 (condition vs Warp
alignment mode vs MissAlignment refinement mode) and 2.11 (tilt axis source).

A *resolved* config (``[provenance].resolved = true``) is the immutable product of
the ``init`` step: every source path is absolute and present, geometry is measured
(no empty strings), and the three mode concepts are kept in distinct fields. Every
later phase loads this and re-discovers nothing.

Three concepts that MUST stay separate (2.7):
- ``condition``            initial IMOD->Warp condition: raw_identity | raw_xf |
                           raw_xf_translation | raw_xf_affine_fixed | ali_identity
- ``warp_alignment_mode``  the etomo_to_warp converter mode: identity | translation |
                           full-affine | quarter-turn-affine (derived from the condition)
- ``refinement_mode``      MissAlignment refinement: smoke | standard | translation |
                           affine2d | rigid | similarity
"""
from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 2

# condition -> Warp converter alignment mode (2.7). The Warp mode is NEVER the
# MissAlignment refinement mode. Values verified against the real testABC
# etomo_missalign_params.json (per-condition alignment_mode/axis_frame).
WARP_MODE_FOR_CONDITION = {
    "raw_identity": "identity",
    "ali_identity": "identity",
    "raw_xf": "translation",            # testABC: raw_xf -> translation (NOT full-affine)
    "raw_xf_translation": "translation",
    "raw_xf_affine_fixed": "quarter-turn-affine",
}
# condition -> Warp axis frame (testABC: raw_xf_affine_fixed/ali_identity = aligned).
AXIS_FRAME_FOR_CONDITION = {
    "raw_identity": "raw",
    "ali_identity": "aligned",
    "raw_xf": "raw",
    "raw_xf_translation": "raw",
    "raw_xf_affine_fixed": "aligned",
}
WARP_ALIGNMENT_MODES = ("identity", "translation", "full-affine", "quarter-turn-affine")
AXIS_FRAMES = ("raw", "aligned")


def axis_frame_for(condition: str) -> str:
    if condition not in AXIS_FRAME_FOR_CONDITION:
        raise ConfigError(f"unknown condition {condition!r} for axis_frame")
    return AXIS_FRAME_FOR_CONDITION[condition]


def volume_invariant_ok(volume_dims_angstrom, target_shape_xyz, target_pixel_A, *, tol=0.02) -> bool:
    """Defect 2.10: the Warp VolumeDimensionsAngstrom must equal
    target_volume_shape_xyz * target_pixel_size_A within ``tol`` (fractional).

    The real testABC XML FAILS this (it wrote raw_shape * output_pixel = 2x target),
    which is the bug; a corrected converter passes it.
    """
    if not (volume_dims_angstrom and target_shape_xyz and target_pixel_A):
        return False
    expected = [float(s) * float(target_pixel_A) for s in target_shape_xyz]
    for got, exp in zip(volume_dims_angstrom, expected):
        if exp == 0:
            return False
        if abs(float(got) - exp) / abs(exp) > tol:
            return False
    return True


def assert_volume_invariant(volume_dims_angstrom, target_shape_xyz, target_pixel_A, *, tol=0.02):
    if not volume_invariant_ok(volume_dims_angstrom, target_shape_xyz, target_pixel_A, tol=tol):
        expected = [round(float(s) * float(target_pixel_A), 1) for s in target_shape_xyz]
        raise ConfigError(
            f"VolumeDimensionsAngstrom {list(volume_dims_angstrom)} != target "
            f"{target_shape_xyz} x {target_pixel_A}A = {expected} (defect 2.10). The physical "
            "volume must not depend on raw-vs-binned input; pass the TARGET shape at the TARGET "
            "pixel, not the raw shape at the output pixel.")
REFINEMENT_MODES = ("smoke", "standard", "translation", "affine2d", "rigid", "similarity")
STOCK_REFINEMENT_MODES = ("smoke", "standard", "translation", "affine2d")  # 2.15: no fork needed
RESULT_BACKENDS = ("warp_xml", "constrained_json")


class ConfigError(ValueError):
    pass


class ModeUnavailableError(ConfigError):
    pass


def constrained_fork_status() -> dict:
    """Whether the rigid/similarity constrained fork is really available (2.15).

    Available iff the fork's dispatcher imports AND exposes the modes, OR the
    delivered patch is pinned to a real upstream commit. Locally neither holds, so
    rigid/similarity are reported unavailable (and gated before submission).
    """
    status = {"available": False, "reason": "", "supported": []}
    try:
        import importlib
        mod = importlib.import_module("miss_alignment.constrained_integration")
        sup = list(getattr(mod, "SUPPORTED_ALIGNMENTS", ()))
        if {"rigid", "similarity"} <= set(sup):
            return {"available": True, "reason": "miss_alignment.constrained_integration imported",
                    "supported": sup}
        status["reason"] = f"fork imported but modes missing: {sup}"
        status["supported"] = sup
        return status
    except Exception:
        pass
    # check the delivered patch pin
    pin = Path(__file__).resolve().parents[2] / "cluster_integration" / "missalignment_patch" / "PINNED_VERSION"
    if pin.is_file():
        txt = pin.read_text()
        for line in txt.splitlines():
            if line.startswith("MISSALIGN_COMMIT="):
                commit = line.split("=", 1)[1].strip()
                if commit and commit != "UNPINNED_SET_ON_CLUSTER":
                    status["reason"] = f"patch pinned to {commit} (apply + probe on cluster)"
                    # pinned but not yet imported here -> treat as unavailable locally,
                    # the cluster preflight (probe.sh) is the real gate.
                    status["reason"] += "; not importable in this process"
                    return status
        status["reason"] = "fork not importable and patch is UNPINNED_SET_ON_CLUSTER"
    else:
        status["reason"] = "fork not importable and no patch deliverable found"
    return status


def assert_refinement_mode_available(mode: str, *, allow_override: bool = False) -> None:
    """Raise a CLEAR error before job generation/submission if the mode is unavailable.

    Stock modes (smoke/standard/translation/affine2d) are always allowed. rigid and
    similarity require the constrained fork; if it is not available, refuse unless
    explicitly overridden.
    """
    if mode in STOCK_REFINEMENT_MODES:
        return
    if mode not in REFINEMENT_MODES:
        raise ConfigError(f"unknown refinement_mode {mode!r}; valid: {REFINEMENT_MODES}")
    st = constrained_fork_status()
    if st["available"]:
        return
    if allow_override:
        return
    raise ModeUnavailableError(
        f"refinement_mode {mode!r} requires the constrained MissAlignment fork, which is NOT "
        f"available ({st['reason']}). Install it on the cluster "
        f"(cluster_integration/missalignment_patch/install.sh) and pin MISSALIGN_COMMIT, or choose "
        f"a stock mode {STOCK_REFINEMENT_MODES}. Refusing to prepare/submit a job that will fail "
        "(defect 2.15).")


def warp_alignment_mode_for(condition: str) -> str:
    if condition not in WARP_MODE_FOR_CONDITION:
        raise ConfigError(f"unknown condition {condition!r}; known: {sorted(WARP_MODE_FOR_CONDITION)}")
    return WARP_MODE_FOR_CONDITION[condition]


@dataclass
class SourcePaths:
    raw_stack: Optional[str] = None
    aligned_stack: Optional[str] = None
    final_xf_file: Optional[str] = None
    final_tilt_file: Optional[str] = None
    raw_tilt_file: Optional[str] = None
    xtilt_file: Optional[str] = None       # SEPARATE from tltxf (2.4): XTILTFILE for recon
    tltxf_file: Optional[str] = None       # SEPARATE: a transform, never the final .xf
    defocus_file: Optional[str] = None
    mdoc_file: Optional[str] = None
    newst_com: Optional[str] = None
    tilt_com: Optional[str] = None
    ctf_com: Optional[str] = None
    source_reconstruction: Optional[str] = None


@dataclass
class Geometry:
    tilt_axis_angle_deg: Optional[float] = None
    tilt_axis_source: Optional[str] = None
    raw_shape_xyz: Optional[list] = None
    raw_pixel_size_A: Optional[float] = None
    aligned_shape_xyz: Optional[list] = None
    aligned_pixel_size_A: Optional[float] = None
    # IMOD reconstruction MRC storage order: X, Y(thickness), Z(detector vertical).
    # Warp conversion applies the explicit X,Z,Y mapping and any detector-plane
    # quarter turn before writing VolumeDimensionsAngstrom.
    target_volume_shape_xyz: Optional[list] = None
    target_volume_frame: Optional[str] = None
    target_pixel_size_A: Optional[float] = None
    target_volume_physical_A: Optional[list] = None
    target_volume_source: Optional[str] = None


@dataclass
class ClusterConfig:
    profile: str = "maxwell"
    environment: Optional[str] = None       # conda/venv to activate on PATH (2.14)
    module_init_script: Optional[str] = None
    imod_module: Optional[str] = None
    imod_bin_dir: Optional[str] = None
    warp_module: Optional[str] = None
    warp_tools_executable: Optional[str] = None
    warp_worker_executable: Optional[str] = None
    partition: str = "vds"
    constraint: Optional[str] = "V100"
    gres: Optional[str] = None
    cpu_partition: Optional[str] = None
    time: str = "7-00:00:00"
    cpus: int = 16
    memory: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    nodelist: Optional[str] = None
    cuda_visible_devices: Optional[str] = None
    omp_num_threads: Optional[int] = None
    generate_slurm: bool = True
    submit: bool = False


@dataclass
class ResolvedProjectConfig:
    basename: str
    data_root: str
    output_dir: str
    sources: SourcePaths
    geometry: Geometry
    conditions: list
    warp_alignment_modes: dict          # condition -> warp mode
    refinement_mode: str                # SEPARATE concept (2.7)
    result_backend: str
    ctf_mode: str
    extra_projection_binning: int
    cluster: ClusterConfig
    resolved: bool = False
    raw: dict = field(default_factory=dict)   # the full original TOML, for round-trip

    # -- accessors ---------------------------------------------------------
    def warp_mode(self, condition: str) -> str:
        return self.warp_alignment_modes.get(condition) or warp_alignment_mode_for(condition)

    def condition_input(self, condition: str) -> "ConditionInput":
        """§7: the concrete (stack, .xf, mode, axis_frame, grids) for this condition."""
        return build_condition_input(self, condition)

    def require_resolved(self) -> "ResolvedProjectConfig":
        if not self.resolved:
            raise ConfigError(
                "this config is not RESOLVED. Run `prepare_imod_to_warp.py init SETTINGS.toml` "
                "first to discover+measure once and write the canonical resolved TOML; later "
                "phases consume only that (no rediscovery).")
        return self

    def to_dict(self) -> dict:
        return {
            "project": {"basename": self.basename, "schema_version": SCHEMA_VERSION},
            "paths": {"data_root": self.data_root, "output_dir": self.output_dir},
            "input": {k: v for k, v in asdict(self.sources).items() if v is not None},
            "geometry": {k: v for k, v in asdict(self.geometry).items() if v is not None},
            "conversion": {"initial_conditions": self.conditions,
                           "condition_modes": self.warp_alignment_modes},
            "multiresolution": {"extra_projection_binning": self.extra_projection_binning},
            "ctf": {"mode": self.ctf_mode},
            "missalignment": {"refinement_mode": self.refinement_mode,
                              "result_backend": self.result_backend},
            "cluster": {k: v for k, v in asdict(self.cluster).items() if v is not None},
            "provenance": {"resolved": self.resolved},
        }


# --------------------------------------------------------------------------- #
# legacy normalization (2.1): accept the setup_missalign_project.py dialect
# --------------------------------------------------------------------------- #
def normalize_legacy(cfg: dict) -> dict:
    """Map the legacy [project].data_dir/out_dir + [input].conditions + [slurm]
    dialect onto the canonical keys. Idempotent for already-canonical TOMLs."""
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
    proj = out.get("project", {})
    paths = out.setdefault("paths", {})
    if "data_root" not in paths and proj.get("data_dir"):
        paths["data_root"] = proj["data_dir"]
    if "output_dir" not in paths and proj.get("out_dir"):
        paths["output_dir"] = proj["out_dir"]
    inp = out.get("input", {})
    conv = out.setdefault("conversion", {})
    if "initial_conditions" not in conv and inp.get("conditions"):
        conv["initial_conditions"] = inp["conditions"]
    # legacy [slurm] -> [cluster]
    sl = out.get("slurm", {})
    if sl:
        cl = out.setdefault("cluster", {})
        for lk, ck in (("gpu_partition", "partition"), ("gpu_constraint", "constraint"),
                       ("nodelist", "nodelist"), ("standard_time", "time"),
                       ("standard_cpus", "cpus")):
            if lk in sl and ck not in cl:
                cl[ck] = sl[lk]
    # legacy env fields: [external_tools].imod_module/module_init_script and
    # [paths].missalign_environment (the real testABC layout).
    ext = out.get("external", {}) or out.get("external_tools", {})
    cl = out.setdefault("cluster", {})
    if ext:
        for lk, ck in (("missalign_environment", "environment"),
                       ("module_init_script", "module_init_script"),
                       ("imod_module", "imod_module")):
            if lk in ext and ck not in cl:
                cl[ck] = ext[lk]
    for src in (paths, proj):
        if src.get("missalign_environment") and "environment" not in cl:
            cl["environment"] = src["missalign_environment"]
    return out


def from_dict(cfg: dict) -> ResolvedProjectConfig:
    cfg = normalize_legacy(cfg)
    proj = cfg.get("project", {})
    paths = cfg.get("paths", {})
    inp = cfg.get("input", {})
    geom = cfg.get("geometry", {})
    conv = cfg.get("conversion", {})
    mr = cfg.get("multiresolution", {})
    ctf = cfg.get("ctf", {})
    ma = cfg.get("missalignment", {})
    cl = cfg.get("cluster", {})
    prov = cfg.get("provenance", {})

    conditions = conv.get("initial_conditions", ["ali_identity"])
    modes = dict(conv.get("condition_modes", {}))
    for c in conditions:
        if c not in modes:                       # only derive when not explicitly set
            modes[c] = warp_alignment_mode_for(c)

    sources = SourcePaths(**{k: inp.get(k) for k in SourcePaths().__dict__})
    geometry = Geometry(**{k: geom.get(k) for k in Geometry().__dict__})
    cluster = ClusterConfig(**{k: cl.get(k, getattr(ClusterConfig(), k)) for k in ClusterConfig().__dict__})

    return ResolvedProjectConfig(
        basename=proj.get("basename") or proj.get("name") or "series",
        data_root=paths.get("data_root", ""), output_dir=paths.get("output_dir", "."),
        sources=sources, geometry=geometry, conditions=list(conditions),
        warp_alignment_modes=modes,
        refinement_mode=ma.get("refinement_mode", "standard"),
        result_backend=ma.get("result_backend", "warp_xml"),
        ctf_mode=ctf.get("mode", "off"),
        extra_projection_binning=int(mr.get("extra_projection_binning", 1)),
        cluster=cluster, resolved=bool(prov.get("resolved", False)), raw=cfg)


def load(path: Path) -> ResolvedProjectConfig:
    with Path(path).open("rb") as fh:
        cfg = tomllib.load(fh)
    return from_dict(cfg)


# --------------------------------------------------------------------------- #
# §7 condition input adapter
# --------------------------------------------------------------------------- #
# The SINGLE authority that maps a condition onto the concrete inputs the Warp
# converter must consume. No phase may re-decide this. Two hard rules it enforces:
#   * an *_identity condition NEVER consumes a real .xf (initial_xf == None == identity);
#   * an *_xf* condition NEVER synthesizes an identity .xf — it must consume the real
#     source raw->aligned .xf (final_xf_file), failing loudly if absent. (The historical
#     bug fed `working_selected` + an identity .xf to a "full-affine" conversion, silently
#     turning raw_xf_affine_fixed into an identity run.)
CONDITION_STACK_ROLE = {
    "raw_identity": "raw_stack",
    "ali_identity": "aligned_stack",
    "raw_xf": "raw_stack",
    "raw_xf_translation": "raw_stack",
    "raw_xf_affine_fixed": "raw_stack",
}
CONDITION_USES_SOURCE_XF = {
    "raw_identity": False,
    "ali_identity": False,
    "raw_xf": True,
    "raw_xf_translation": True,
    "raw_xf_affine_fixed": True,
}


@dataclass(frozen=True)
class GridSpec:
    name: str
    shape_xy: Optional[tuple]
    pixel_size_A: Optional[float]


@dataclass(frozen=True)
class ConditionInput:
    condition: str
    stack: str                      # absolute path of the input stack the converter consumes
    stack_role: str                 # 'raw_stack' | 'aligned_stack'
    stack_grid: str                 # 'raw' | 'aligned'
    tilt_file: str
    initial_xf: Optional[str]       # the .xf applied at conversion; None == identity
    source_xf: Optional[str]        # the source raw->aligned .xf (None for identity conditions)
    alignment_mode: str             # warp converter mode (identity|translation|full-affine|quarter-turn-affine)
    axis_frame: str                 # 'raw' | 'aligned'
    grids: dict                     # name -> GridSpec (raw/aligned/target)

    @property
    def is_identity(self) -> bool:
        return self.initial_xf is None

    def to_dict(self) -> dict:
        return {"condition": self.condition, "stack": self.stack, "stack_role": self.stack_role,
                "stack_grid": self.stack_grid, "tilt_file": self.tilt_file,
                "initial_xf": self.initial_xf, "source_xf": self.source_xf,
                "alignment_mode": self.alignment_mode, "axis_frame": self.axis_frame,
                "is_identity": self.is_identity,
                "grids": {k: {"shape_xy": list(v.shape_xy) if v.shape_xy else None,
                              "pixel_size_A": v.pixel_size_A} for k, v in self.grids.items()}}


def condition_input_from_paths(condition: str, *, raw_stack=None, aligned_stack=None,
                               final_xf_file=None, final_tilt_file=None, warp_mode=None,
                               geometry: "Geometry | None" = None,
                               require_files: bool = True) -> ConditionInput:
    """Core §7 resolver over explicit paths (the production engine calls this with the
    already-resolved source selection; ``build_condition_input`` wraps a ResolvedProjectConfig)."""
    if condition not in CONDITION_STACK_ROLE:
        raise ConfigError(f"unknown condition {condition!r}; known: {sorted(CONDITION_STACK_ROLE)}")
    stack_role = CONDITION_STACK_ROLE[condition]
    uses_xf = CONDITION_USES_SOURCE_XF[condition]
    alignment_mode = warp_mode or warp_alignment_mode_for(condition)
    axis_frame = axis_frame_for(condition)
    by_role = {"raw_stack": raw_stack, "aligned_stack": aligned_stack}

    stack = by_role[stack_role]
    if not stack or not str(stack).strip():
        raise ConfigError(
            f"condition {condition!r} needs the {stack_role} but it is not set; init must measure "
            f"and record it (this phase rediscovers nothing).")
    if require_files and not Path(stack).is_file():
        raise ConfigError(f"condition {condition!r}: {stack_role} {stack!r} is not a file")
    stack_grid = "raw" if stack_role == "raw_stack" else "aligned"

    if not final_tilt_file or not str(final_tilt_file).strip():
        raise ConfigError(f"condition {condition!r}: final_tilt_file is not set")
    if require_files and not Path(str(final_tilt_file)).is_file():
        raise ConfigError(f"condition {condition!r}: final_tilt_file {final_tilt_file!r} is not a file")

    if uses_xf:
        xf = final_xf_file
        if not xf or not str(xf).strip():
            raise ConfigError(
                f"condition {condition!r} is an *_xf condition and requires the real source "
                f"raw->aligned .xf (final_xf_file); it is absent. Refusing to synthesize an "
                f"identity .xf (that would silently turn {condition!r} into an identity run).")
        if str(xf).endswith(".tltxf"):
            raise ConfigError(
                f"condition {condition!r}: the alignment .xf must be the final .xf, not the .tltxf "
                f"({xf!r}) — the .tltxf is an intermediate transform (2.4).")
        if require_files and not Path(xf).is_file():
            raise ConfigError(f"condition {condition!r}: final_xf_file {xf!r} is not a file")
        initial_xf = source_xf = str(xf)
    else:
        initial_xf = source_xf = None

    g = geometry or Geometry()
    def _grid(name, shape, pix):
        return GridSpec(name, tuple(shape[:2]) if shape else None, pix)
    grids = {"raw": _grid("raw", g.raw_shape_xyz, g.raw_pixel_size_A),
             "aligned": _grid("aligned", g.aligned_shape_xyz, g.aligned_pixel_size_A),
             "target": _grid("target", g.target_volume_shape_xyz, g.target_pixel_size_A)}
    return ConditionInput(
        condition=condition, stack=str(stack), stack_role=stack_role, stack_grid=stack_grid,
        tilt_file=str(final_tilt_file), initial_xf=initial_xf, source_xf=source_xf,
        alignment_mode=alignment_mode, axis_frame=axis_frame, grids=grids)


def build_condition_input(rc: ResolvedProjectConfig, condition: str) -> ConditionInput:
    """§7: resolve a condition's converter inputs from a ResolvedProjectConfig."""
    return condition_input_from_paths(
        condition, raw_stack=rc.sources.raw_stack, aligned_stack=rc.sources.aligned_stack,
        final_xf_file=rc.sources.final_xf_file, final_tilt_file=rc.sources.final_tilt_file,
        warp_mode=rc.warp_mode(condition), geometry=rc.geometry, require_files=True)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate(rc: ResolvedProjectConfig, *, require_resolved: bool = False,
             require_geometry: bool = False) -> list:
    """Return a list of problems (empty == OK). Raises only via require_* flags."""
    problems = []
    if rc.refinement_mode not in REFINEMENT_MODES:
        problems.append(f"refinement_mode {rc.refinement_mode!r} not in {REFINEMENT_MODES}")
    if rc.result_backend not in RESULT_BACKENDS:
        problems.append(f"result_backend {rc.result_backend!r} not in {RESULT_BACKENDS}")
    for c in rc.conditions:
        wm = rc.warp_alignment_modes.get(c)
        if wm not in WARP_ALIGNMENT_MODES:
            problems.append(f"condition {c!r}: warp mode {wm!r} not in {WARP_ALIGNMENT_MODES}")
        # 2.7: a refinement mode must never be used as a warp mode
        if wm in ("rigid", "similarity", "standard", "smoke", "affine2d"):
            problems.append(f"condition {c!r}: warp mode {wm!r} is a refinement mode (concept confusion)")
    # 2.4: xtilt and tltxf must not be the same file
    if rc.sources.xtilt_file and rc.sources.tltxf_file and \
            Path(rc.sources.xtilt_file) == Path(rc.sources.tltxf_file):
        problems.append("xtilt_file and tltxf_file resolve to the same path (2.4 conflation)")
    if require_geometry:
        g = rc.geometry
        for fname in ("tilt_axis_angle_deg", "raw_pixel_size_A", "aligned_pixel_size_A"):
            if getattr(g, fname) in (None, "", 0):
                problems.append(f"geometry.{fname} is empty/zero in a resolved config (2.2)")
        # §3/§4: target volume geometry must be present and physically consistent
        for fname in ("target_volume_shape_xyz", "target_pixel_size_A"):
            if getattr(g, fname) in (None, "", 0, []):
                problems.append(f"geometry.{fname} is empty in a resolved config (§4)")
        if g.target_volume_shape_xyz and g.target_pixel_size_A and g.target_volume_physical_A:
            exp = [round(float(s) * float(g.target_pixel_size_A), 2) for s in g.target_volume_shape_xyz]
            got = [round(float(x), 2) for x in g.target_volume_physical_A]
            if any(abs(a - b) > 1e-2 for a, b in zip(exp, got)):
                problems.append(f"target physical {got} != shape*pixel {exp} (§4 invariant)")
    # §3: in a resolved config no REQUIRED source may be empty; every declared condition
    # must resolve its converter inputs (right stack present, *_xf conditions have the .xf).
    if require_resolved and rc.resolved:
        for c in rc.conditions:
            role = CONDITION_STACK_ROLE.get(c)
            if role and not (getattr(rc.sources, role) or "").strip():
                problems.append(f"condition {c!r}: required {role} is empty in resolved config (§3)")
            if CONDITION_USES_SOURCE_XF.get(c) and not (rc.sources.final_xf_file or "").strip():
                problems.append(f"condition {c!r}: requires final_xf_file but it is empty (§7)")
        if not (rc.sources.final_tilt_file or "").strip():
            problems.append("final_tilt_file is empty in resolved config (§3)")
    if require_resolved and not rc.resolved:
        problems.append("config is not resolved (run `init` first)")
    return problems
