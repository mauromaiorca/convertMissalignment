#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
        fh.write("\n")
    os.replace(tmp, path)


def write_result(
    *,
    run_dir: Path,
    stage_id: str,
    status: str,
    exit_code: int = 0,
    failed_command: str = "",
    log_path: str = "",
    details: dict | None = None,
) -> Path:
    data = {
        "stage": stage_id,
        "status": status,
        "exit_code": int(exit_code),
        "failed_command": failed_command,
        "hostname": socket.getfqdn() or socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "log_path": log_path,
    }
    if details:
        data.update(details)
    path = Path(run_dir) / "manifests" / f"{stage_id}_stage_result.json"
    atomic_json(path, data)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Write an atomic v6 stage-result JSON file.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--failed-command", default="")
    parser.add_argument("--log-path", default="")
    args = parser.parse_args()
    write_result(
        run_dir=args.run_dir,
        stage_id=args.stage_id,
        status=args.status,
        exit_code=args.exit_code,
        failed_command=args.failed_command,
        log_path=args.log_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

