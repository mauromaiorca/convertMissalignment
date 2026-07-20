#!/usr/bin/env python3
"""Generate a standalone reconstruction sbatch (CPU). Reuses the shared diagnostic
preamble/monitor when a run_dir is supplied; otherwise emits a minimal safe job."""
from __future__ import annotations

from pathlib import Path


def reconstruction_sbatch(*, job_name: str, tilt_com: str, work_dir: str,
                          profile: str = "maxwell", run_dir: str | None = None,
                          cpus: int = 16) -> str:
    head = (f"#!/usr/bin/env bash\n#SBATCH --job-name={job_name}\n"
            f"#SBATCH --partition=cpu\n#SBATCH --time=0-08:00:00\n#SBATCH --cpus-per-task={cpus}\n")
    if run_dir:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from pipeline.jobs import _preamble, _monitor
            body = _preamble(job_name, Path(run_dir)) + _monitor(job_name, Path(run_dir))
        except Exception:
            body = "set -Eeuo pipefail\n"
    else:
        body = "set -Eeuo pipefail\n"
    return (head + f"# cluster profile: {profile}\n" + body +
            f'\ncd {work_dir}\n'
            f'if [ ! -f {Path(tilt_com).name} ]; then echo "missing {tilt_com}" >&2; exit 2; fi\n'
            f'submfg {Path(tilt_com).name}\n'
            f'echo "reconstruction done"\n')
