#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v6.config import load  # noqa: E402
from v6.stage_result import write_result  # noqa: E402


def _verify_toml(path: Path, expected: str) -> str:
    got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if got != expected:
        raise RuntimeError(f"TOML hash mismatch: {got} != {expected}")
    return got


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute one v6 scientific stage.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-toml-hash", required=True)
    args = parser.parse_args()
    try:
        toml_hash = _verify_toml(args.settings, args.expected_toml_hash)
        cfg = load(args.settings)
        if args.stage == "10_warp_ingest":
            from v6.stack_ingest import run
            run(cfg, settings=args.settings, run_dir=args.run_dir, toml_hash=toml_hash)
        elif args.stage == "20_initial_alignment_and_qc":
            from v6.legacy_affine_stage import run
            run(cfg, settings=args.settings, run_dir=args.run_dir, toml_hash=toml_hash)
        elif args.stage == "30_missalignment":
            _validate_missalignment_inputs(cfg, args.run_dir)
            raise RuntimeError(
                "30_missalignment requires the cluster MissAlignment runner and deterministic FINAL_XML "
                "contract; refusing to write a placeholder completion manifest"
            )
        else:
            raise RuntimeError(f"stage is not implemented in this pass: {args.stage}")
        return 0
    except Exception as exc:
        write_result(run_dir=args.run_dir, stage_id=args.stage, status="failed", exit_code=1,
                     failed_command=f"execute_stage {args.stage}", details={"error": str(exc)})
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def _validate_missalignment_inputs(cfg, run_dir: Path) -> None:
    ts = cfg.tilt_series[0]
    pre = Path(run_dir) / "warp" / "pre_missalign" / f"{ts.id}.xml"
    if not pre.is_file() or pre.stat().st_size <= 0:
        raise RuntimeError(f"pre_missalign XML missing or empty: {pre}")


if __name__ == "__main__":
    raise SystemExit(main())
