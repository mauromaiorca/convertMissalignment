#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from v6.config import V6ConfigError, load, sha256_file  # noqa: E402
from v6.stages import StagePlanningError, missalignment_stage, plan_stages  # noqa: E402


def _load(path: Path):
    try:
        return load(path)
    except V6ConfigError as exc:
        raise SystemExit(f"ERROR: {exc}")


def cmd_show(settings: Path) -> int:
    cfg = _load(settings)
    print(f"schema_version       : {cfg.schema_version}")
    print(f"software_version     : {cfg.project.get('software_version', '')}")
    print(f"project              : {cfg.project.get('name')}")
    print(f"output_dir           : {cfg.project.get('output_dir')}")
    for ts in cfg.tilt_series:
        print(f"tilt_series          : {ts.id}")
        print(f"source_mode          : {ts.source.mode}")
        print(f"alignment_backend    : {ts.warp.alignment_backend}")
        print(f"extra_binning        : {ts.binning.extra_projection_binning}")
    return 0


def cmd_validate(settings: Path) -> int:
    cfg = _load(settings)
    try:
        plan_stages(cfg)
    except StagePlanningError as exc:
        print(f"ERROR: {exc}")
        return 2
    print("[validate] v6 configuration OK")
    print(f"[validate] TOML SHA-256: {sha256_file(settings)}")
    return 0


def cmd_plan(settings: Path) -> int:
    cfg = _load(settings)
    for stage in plan_stages(cfg):
        print(f"{stage.stage_id}: {stage.scientific_purpose}")
    return 0


def cmd_status(settings: Path) -> int:
    cfg = _load(settings)
    out = Path(cfg.project["output_dir"])
    print(f"source mode          : {cfg.tilt_series[0].source.mode}")
    print(f"capabilities         : {cfg.tilt_series[0].capabilities}")
    try:
        stages = plan_stages(cfg)
    except StagePlanningError as exc:
        print(f"planned stages       : []")
        print(f"blocked stages       : {{'planning': {str(exc)!r}}}")
        return 2
    jobs_dir = out / "jobs"
    if (jobs_dir / "30_missalignment.sbatch").is_file() or _snapshot_validated(out, "pre_missalign"):
        stages.append(missalignment_stage(cfg))
    planned = [stage.stage_id for stage in stages]
    completed = []
    validated = []
    failed = []
    malformed = []
    results = {}
    for result in sorted((out / "manifests").glob("*_stage_result.json")):
        try:
            data = json.loads(result.read_text())
        except json.JSONDecodeError as exc:
            malformed.append(f"{result.name}: {exc}")
            continue
        stage = data.get("stage") or result.name.removesuffix("_stage_result.json")
        results[stage] = data
        status = data.get("status")
        if status == "completed":
            completed.append(stage)
        elif status == "validated":
            completed.append(stage)
            validated.append(stage)
        elif status == "failed":
            failed.append(stage)
    graph = _read_json(out / "manifests" / "job_graph.json")
    blocked = dict(graph.get("blocked") or {})
    for item in malformed:
        blocked[f"malformed_result:{item.split(':', 1)[0]}"] = item
    next_action = _next_action(out, planned, results, blocked)
    print(f"planned stages       : {planned}")
    print(f"completed stages     : {completed}")
    print(f"validated stages     : {validated}")
    print(f"failed stages        : {failed}")
    print(f"blocked stages       : {blocked}")
    if next_action:
        label, cmd = next_action
        print(label)
        print(f"  {cmd}")
    else:
        print("next valid action    : none")
    return 0


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _snapshot_validated(out: Path, snapshot_type: str) -> bool:
    data = _read_json(out / "warp" / snapshot_type / "snapshot_manifest.json")
    return bool(data) and data.get("status") != "declared" and data.get("snapshot_type") == snapshot_type


def _next_action(out: Path, planned: list[str], results: dict, blocked: dict) -> tuple[str, str] | None:
    jobs = {
        "10_warp_ingest": out / "jobs" / "10_warp_ingest.sbatch",
        "20_initial_alignment_and_qc": out / "jobs" / "20_initial_alignment_and_qc.sbatch",
        "30_missalignment": out / "jobs" / "30_missalignment.sbatch",
    }
    labels = {
        "10_warp_ingest": "[next] submit Warp ingest:",
        "20_initial_alignment_and_qc": "[next] submit initial alignment:",
        "30_missalignment": "[next] submit MissAlignment:",
    }
    if any((results.get(stage) or {}).get("status") == "failed" for stage in planned):
        return None
    if "10_warp_ingest" in planned and (results.get("10_warp_ingest") or {}).get("status") != "validated":
        if (results.get("10_warp_ingest") or {}).get("status") == "completed":
            blocked["10_warp_ingest"] = "10_warp_ingest completed but is not validated"
            return None
        if jobs["10_warp_ingest"].is_file() and "10_warp_ingest" not in blocked:
            return labels["10_warp_ingest"], f"sbatch {jobs['10_warp_ingest']}"
        return None
    if "20_initial_alignment_and_qc" in planned and (results.get("20_initial_alignment_and_qc") or {}).get("status") != "validated":
        if (results.get("10_warp_ingest") or {}).get("status") == "completed":
            blocked["20_initial_alignment_and_qc"] = "10_warp_ingest completed but is not validated"
            return None
        if jobs["20_initial_alignment_and_qc"].is_file() and "20_initial_alignment_and_qc" not in blocked:
            return labels["20_initial_alignment_and_qc"], f"sbatch {jobs['20_initial_alignment_and_qc']}"
        return None
    if "30_missalignment" in planned and (results.get("30_missalignment") or {}).get("status") not in ("completed", "validated"):
        if (results.get("20_initial_alignment_and_qc") or {}).get("status") == "completed":
            blocked["30_missalignment"] = "20_initial_alignment_and_qc completed but is not validated"
            return None
        if jobs["30_missalignment"].is_file() and "30_missalignment" not in blocked:
            return labels["30_missalignment"], f"sbatch {jobs['30_missalignment']}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a v6 Warp project.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("show-resolved", "validate", "plan", "status"):
        p = sub.add_parser(name)
        p.add_argument("settings", type=Path)
    args = parser.parse_args()
    if args.cmd == "show-resolved":
        return cmd_show(args.settings)
    if args.cmd == "validate":
        return cmd_validate(args.settings)
    if args.cmd == "plan":
        return cmd_plan(args.settings)
    if args.cmd == "status":
        return cmd_status(args.settings)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
