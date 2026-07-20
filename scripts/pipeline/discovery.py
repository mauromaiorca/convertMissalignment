#!/usr/bin/env python3
"""Deterministic source discovery for an eTomo/IMOD project.

Returns a typed :class:`SourceInventory`. Selection precedence per field:
1. explicit TOML (``[input]``/``[ctf]``);
2. explicit CLI override;
3. exact basename match;
4. scored eTomo discovery (best unambiguous score wins);
5. fail on ambiguity (two candidates with the equal best score).

Never selects ``*.prexf``, ``*_fid.xf``, ``rotation.xf``, ``*.tltxf`` for the
final ``.xf``. Reuses the deterministic ambiguity-failure idea from
``ctf.discover_ctf_inputs`` but adds explicit scoring and the six source types it
omitted (raw_tilt, xtilt, mdoc, newst.com, tilt.com, source_reconstruction).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Patterns that must never be chosen as the final raw->aligned .xf.
XF_BLOCKLIST = ("prexf", "_fid.xf", "rotation.xf", ".tltxf", ".xtilt")


class DiscoveryError(ValueError):
    pass


@dataclass
class Candidate:
    path: str
    score: int
    reason: str


@dataclass
class SourceInventory:
    basename: str
    data_dir: str
    raw_stack: Optional[str] = None
    aligned_stack: Optional[str] = None
    final_xf: Optional[str] = None
    tilt_file: Optional[str] = None
    raw_tilt_file: Optional[str] = None
    xtilt_file: Optional[str] = None       # .xtilt (X-axis tilt angles for reconstruction)
    tltxf_file: Optional[str] = None       # .tltxf (a transform; NEVER the final .xf) (2.4)
    defocus_file: Optional[str] = None
    mdoc_file: Optional[str] = None
    newst_com: Optional[str] = None
    tilt_com: Optional[str] = None
    ctf_com: Optional[str] = None
    source_reconstruction: Optional[str] = None
    report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _score_candidate(p: Path, basename: str, exact_names, suffix_patterns, blocklist=()) -> Optional[Candidate]:
    name = p.name
    low = name.lower()
    if any(b in low for b in blocklist):
        return None
    # exact-name match scores highest, then basename-prefixed, then suffix-only.
    for i, exact in enumerate(exact_names):
        if name == exact:
            return Candidate(str(p), 100 - i, f"exact name {exact}")
    if name.startswith(basename):
        for i, suf in enumerate(suffix_patterns):
            if low.endswith(suf):
                return Candidate(str(p), 60 - i, f"basename+{suf}")
    for i, suf in enumerate(suffix_patterns):
        if low.endswith(suf):
            return Candidate(str(p), 30 - i, f"suffix {suf}")
    return None


# eTomo creates backup/template copies of the command/parameter files in these
# subdirectories; the canonical file is the one in the project root, never these.
BACKUP_DIR_COMPONENTS = ("dfltcoms", "origcoms", "savecoms", "recovery", "_recovery",
                         "backup", "_backup", "old", "trash")


def _location_delta(path, data_dir) -> int:
    """Score adjustment by location: backup/template subdirs are strongly demoted;
    otherwise shallower (closer to the project root) wins."""
    try:
        rel = Path(path).relative_to(data_dir)
    except ValueError:
        rel = Path(path)
    dirs = [x.lower() for x in rel.parts[:-1]]
    if any(any(b == part or b in part for b in BACKUP_DIR_COMPONENTS) for part in dirs):
        return -1000
    return -len(dirs)            # root file (depth 0) beats an equally-named nested one


def _resolve(data_dir: Path, basename: str, *, exact_names, suffix_patterns,
             blocklist=(), explicit: Optional[str] = None, label: str,
             files: Optional[list[Path]] = None) -> tuple[Optional[str], dict]:
    """Resolve one source field deterministically, returning (path, report)."""
    rep: dict = {"label": label, "candidates": [], "rejected": []}
    if explicit:
        ep = Path(explicit)
        if not ep.is_file():
            raise DiscoveryError(f"{label}: explicit path does not exist: {ep}")
        rep["selected"] = str(ep); rep["selection_reason"] = "explicit (TOML/CLI)"
        return str(ep), rep
    cands: list[Candidate] = []
    for p in (files if files is not None else sorted(data_dir.rglob("*"))):
        if not p.is_file():
            continue
        c = _score_candidate(p, basename, exact_names, suffix_patterns, blocklist)
        if c:
            delta = _location_delta(p, data_dir)
            if delta <= -1000:
                rep["rejected"].append({"path": str(p), "reason": "backup/template subdir"})
                continue          # never select an eTomo backup/template copy
            c.score += delta      # prefer root over nested for otherwise-equal matches
            cands.append(c)
        elif any(b in p.name.lower() for b in blocklist):
            rep["rejected"].append({"path": str(p), "reason": f"blocklisted for {label}"})
    rep["candidates"] = [asdict(c) for c in sorted(cands, key=lambda c: -c.score)]
    if not cands:
        rep["selected"] = None; rep["selection_reason"] = "no candidate"
        return None, rep
    best = max(c.score for c in cands)
    top = [c for c in cands if c.score == best]
    if len(top) > 1:
        raise DiscoveryError(
            f"{label}: ambiguous — {len(top)} candidates tied at score {best}: "
            f"{[c.path for c in top]}. Refusing to guess; set it explicitly in [input].")
    rep["selected"] = top[0].path; rep["selection_reason"] = top[0].reason
    return top[0].path, rep


def discover_sources(data_dir: Path, basename: str, *, overrides: Optional[dict] = None,
                     _files: Optional[list[Path]] = None) -> SourceInventory:
    """Build the source inventory from one filesystem scan.

    ``overrides`` carries explicit TOML/CLI paths. ``_files`` is an internal cache
    used by basename inference to avoid re-walking GPFS for every candidate.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise DiscoveryError(f"data_dir is not a directory: {data_dir}")
    ov = overrides or {}
    files = _files if _files is not None else [p for p in sorted(data_dir.rglob("*")) if p.is_file()]
    inv = SourceInventory(basename=basename, data_dir=str(data_dir))
    report: dict = {}

    spec = {
        "raw_stack": dict(exact_names=[f"{basename}.mrc", f"{basename}.st"],
                          suffix_patterns=[".mrc", ".st"],
                          blocklist=["_ali.", "_rec", "_full_rec", "preview"]),
        "aligned_stack": dict(exact_names=[f"{basename}_ali.mrc", f"{basename}.ali"],
                              suffix_patterns=["_ali.mrc", ".ali"], blocklist=["_rec", "preview"]),
        "final_xf": dict(exact_names=[f"{basename}.xf"], suffix_patterns=[".xf"], blocklist=list(XF_BLOCKLIST)),
        "tilt_file": dict(exact_names=[f"{basename}.tlt", f"{basename}_ali.tlt"],
                          suffix_patterns=[".tlt"], blocklist=["_ali.tlt", ".rawtlt", ".xtilt"]),
        "raw_tilt_file": dict(exact_names=[f"{basename}.rawtlt"], suffix_patterns=[".rawtlt"]),
        "xtilt_file": dict(exact_names=[f"{basename}.xtilt"], suffix_patterns=[".xtilt"]),
        # .tltxf is a SEPARATE artifact from .xtilt (2.4): a transform, never the final .xf.
        "tltxf_file": dict(exact_names=[f"{basename}.tltxf"], suffix_patterns=[".tltxf"]),
        "defocus_file": dict(exact_names=[f"{basename}.defocus"], suffix_patterns=[".defocus"]),
        "mdoc_file": dict(exact_names=[f"{basename}.mrc.mdoc", f"{basename}.mdoc"], suffix_patterns=[".mdoc"]),
        "newst_com": dict(exact_names=["newst.com"], suffix_patterns=["newst.com"]),
        "tilt_com": dict(exact_names=["tilt.com"], suffix_patterns=["tilt.com"], blocklist=["ctf"]),
        "ctf_com": dict(exact_names=["ctfcorrection.com"], suffix_patterns=["ctfcorrection.com"]),
        "source_reconstruction": dict(exact_names=[f"{basename}_full_rec.mrc", f"{basename}_rec.mrc"],
                                      suffix_patterns=["_full_rec.mrc", "_rec.mrc"]),
    }
    for fieldname, kw in spec.items():
        path, rep = _resolve(data_dir, basename, label=fieldname,
                             explicit=ov.get(fieldname), files=files, **kw)
        setattr(inv, fieldname, path)
        report[fieldname] = rep
    inv.report = report
    return inv


def infer_basename(data_dir: Path) -> tuple[str, dict]:
    """Deterministically infer the eTomo basename (2.2/2.3: canonical discovery, no
    silent first-glob). Scores candidates by how many canonical source types resolve;
    fails with ranked candidates on a tie.

    Candidates: the directory name (with a trailing ``_Imod``/``_imod`` stripped), the
    directory name as-is, and the stems of any ``<x>_ali.mrc`` / ``<x>.mrc`` / ``<x>.st``
    present.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise DiscoveryError(f"data_dir is not a directory: {data_dir}")
    cands: set[str] = set()
    dname = data_dir.name
    for suf in ("_Imod", "_imod", "_IMOD"):
        if dname.endswith(suf):
            cands.add(dname[: -len(suf)])
    cands.add(dname)
    files = [p for p in sorted(data_dir.rglob("*")) if p.is_file()]
    for p in files:
        n = p.name
        if n.endswith("_ali.mrc"):
            cands.add(n[: -len("_ali.mrc")])
        elif n.endswith(".mrc") and not n.endswith("_rec.mrc"):
            cands.add(n[: -len(".mrc")])
        elif n.endswith(".st"):
            cands.add(n[: -len(".st")])
    def _quality(inv) -> int:
        # sum the MATCH QUALITY of each resolved field (exact-name >> basename-prefix
        # >> suffix), so a basename with exact-name files beats one that only matches
        # by file suffix (e.g. '<dir>' beats '<dir>_Imod' when files are '<dir>.mrc').
        total = 0
        for fieldname, rep in (inv.report or {}).items():
            sel = rep.get("selected")
            if not sel:
                continue
            for c in rep.get("candidates", []):
                if c.get("path") == sel:
                    total += int(c.get("score", 0))
                    break
        return total

    scored = []
    for base in sorted(cands):
        if not base:
            continue
        try:
            inv = discover_sources(data_dir, base, _files=files)
        except DiscoveryError:
            continue
        # require at least a stack + a tilt file to be a plausible project
        if (inv.raw_stack or inv.aligned_stack) and (inv.tilt_file or inv.raw_tilt_file):
            scored.append((_quality(inv), base, inv))
    report = {"candidates": [{"basename": b, "score": s} for s, b, _ in
                             sorted(scored, key=lambda x: -x[0])]}
    if not scored:
        raise DiscoveryError(
            f"could not infer a basename under {data_dir}: no candidate has both a stack and a "
            "tilt file. Pass --basename explicitly.")
    best = max(s for s, _, _ in scored)
    top = [b for s, b, _ in scored if s == best]
    if len(top) > 1:
        ranked = ", ".join(top)
        raise DiscoveryError(
            f"ambiguous basename under {data_dir}: candidates tied at score {best}: {ranked}. "
            f"Refusing to guess; rerun with --basename <one of: {ranked}>.")
    report["selected"] = top[0]
    return top[0], report


def check_section_consistency(inv: SourceInventory, *, measure_mrc=None, count_lines=None) -> dict:
    """Cross-check section counts: raw == aligned == .xf rows == .tlt rows.

    ``measure_mrc(path) -> n_sections`` and ``count_lines(path) -> int`` are
    injected so this stays import-light/testable. Returns a report; raises
    DiscoveryError on an unexplained mismatch.
    """
    counts: dict = {}
    if measure_mrc:
        if inv.raw_stack:
            counts["raw_sections"] = measure_mrc(inv.raw_stack)
        if inv.aligned_stack:
            counts["aligned_sections"] = measure_mrc(inv.aligned_stack)
    if count_lines:
        if inv.final_xf:
            counts["xf_rows"] = count_lines(inv.final_xf)
        if inv.tilt_file:
            counts["tilt_rows"] = count_lines(inv.tilt_file)
    distinct = set(counts.values())
    consistent = len(distinct) <= 1
    if not consistent:
        raise DiscoveryError(
            f"section-count mismatch across sources: {counts}. A real eTomo project "
            "must have equal raw/aligned/.xf/.tlt counts; set sources explicitly or "
            "fix the project.")
    return {"counts": counts, "consistent": consistent}
