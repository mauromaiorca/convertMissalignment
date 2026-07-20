#!/usr/bin/env python3
"""prepare_imod_reconstruction: the canonical reconstruction entry (explicit inputs).

Generates command files + run script (+ optional half-set files), and EITHER runs
locally (real ``tilt``/``submfg``), generates a Slurm job, or only writes the files
(``execution="skip"``). It never searches for a MissAlignment XML and never exports
a final ``.xf`` — those belong to Phase-3 finalize.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import command_files as CF
from . import halfsets as HS
from . import slurm as SL
from .model import ReconstructionRequest, ReconstructionResult, validate_request


def prepare_imod_reconstruction(req: ReconstructionRequest, *, run_dir: str | None = None,
                                monitor=None) -> ReconstructionResult:
    validate_request(req)
    out = Path(req.output_dir); out.mkdir(parents=True, exist_ok=True)
    res = ReconstructionResult(output_dir=str(out))

    in_stack = (req.aligned_stack if req.input_mode == "aligned_stack" else req.raw_stack)
    # default geometry: require explicit fullimage/thickness (no integer-division guessing)
    if not req.fullimage_xy or not req.thickness:
        res.notes.append("fullimage_xy/thickness not supplied; tilt.com will need them filled in")
    rec_path = out / f"{req.basename}_final_rec.mrc"
    res.output_rec = str(rec_path)
    tilt_com = out / "tilt_final.com"
    com_text = CF.build_tilt_com(
        in_stack=str(in_stack), out_rec=str(rec_path), tilt_file=str(req.tilt_file),
        fullimage_xy=req.fullimage_xy or (0, 0), thickness=req.thickness or 0,
        xtilt_file=req.xtilt_file)
    tilt_com.write_text(com_text)
    res.tilt_com = str(tilt_com)

    # half sets
    if req.halfmaps and req.tilt_file:
        angles = [float(x) for x in Path(req.tilt_file).read_text().splitlines() if x.strip()]
        halves = HS.split_halfsets(angles, mode=req.half_split_mode)
        res.half_files = HS.write_half_tilt_files(out, req.basename, angles, halves)
        for half in ("even", "odd"):
            hp = out / f"tilt_final_{half}.com"
            hp.write_text(CF.build_tilt_com(
                in_stack=str(in_stack), out_rec=str(out / f"{req.basename}_{half}_rec.mrc"),
                tilt_file=res.half_files[half], fullimage_xy=req.fullimage_xy or (0, 0),
                thickness=req.thickness or 0, xtilt_file=req.xtilt_file))

    # run script
    run_sh = out / "run_reconstruction.sh"
    run_sh.write_text("#!/usr/bin/env bash\nset -Eeuo pipefail\n"
                      f"cd {out}\nsubmfg {tilt_com.name}\n")
    run_sh.chmod(0o755)
    res.run_script = str(run_sh)

    if req.execution == "slurm":
        sb = out / f"reconstruct_{req.basename}.sbatch"
        sb.write_text(SL.reconstruction_sbatch(job_name=f"recon_{req.basename}",
                      tilt_com=str(tilt_com), work_dir=str(out), profile=req.cluster_profile,
                      run_dir=run_dir))
        sb.chmod(0o755)
        res.sbatch = str(sb)
        res.notes.append("slurm: submit the generated .sbatch on the cluster")
    elif req.execution == "local":
        if not (req.fullimage_xy and req.thickness):
            raise ValueError("local execution requires fullimage_xy and thickness")
        directives = "\n".join(ln for ln in com_text.splitlines()
                               if ln and not ln.startswith("#") and not ln.startswith("$")) + "\n"
        cp = subprocess.run(["tilt", "-StandardInput"], input=directives, text=True,
                            capture_output=True,
                            env={**os.environ, "IMOD_DIR": os.environ.get("IMOD_DIR", "/Applications/IMOD")})
        if cp.returncode != 0 or not rec_path.is_file():
            raise RuntimeError(f"tilt reconstruction failed: {cp.stdout[-300:]}{cp.stderr[-200:]}")
        res.executed = True
    else:
        res.notes.append("execution=skip: command files generated only")
    return res
