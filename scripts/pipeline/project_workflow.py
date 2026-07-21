#!/usr/bin/env python3
"""Project preparation handlers for the v8 project CLI.

Verbs: ``validate``, ``prepare``, ``status``, ``collect-debug``. These wire the
new infrastructure (discovery, runlayout, runlog, capabilities, jobs) around the
existing ``orchestrate`` engine, which still performs the heavy IMOD work.
Project preparation imports the native IMOD geometry into Warp synchronously.
Generated Slurm jobs are retained for re-import/recovery, reconstruction,
MissAlignment and export.

Nothing here submits Slurm jobs or runs CUDA/MissAlignment.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from . import capabilities as CAP
from . import discovery as DISC
from . import jobs as JOBS
from . import runlog as RL
from .runlayout import RunLayout, dataset_id_from_config


def _run_id(layout: RunLayout) -> str:
    return "r" + hashlib.sha256(str(layout.run_dir).encode()).hexdigest()[:12]


def _validate_imported_warp_dataset(layout: RunLayout) -> dict:
    """Validate the minimal v8 contract required by all downstream jobs."""
    project = layout.training_dir.resolve()
    marker = project / "_converted.marker"
    validation = project / "conversion_validation.json"
    xmls = [path for path in project.glob("*.xml") if path.is_file() and path.stat().st_size > 0]
    if not marker.is_file():
        raise RuntimeError(f"Warp import did not create conversion marker: {marker}")
    if len(xmls) != 1:
        raise RuntimeError(
            f"Warp import must create exactly one non-empty root XML in {project}; "
            f"found {len(xmls)}"
        )
    if not validation.is_file() or validation.stat().st_size <= 0:
        raise RuntimeError(f"Warp import did not create validation report: {validation}")
    if not layout.dataset_manifest.is_file() or layout.dataset_manifest.stat().st_size <= 0:
        raise RuntimeError(
            f"Warp import was converted but not published as a v8 dataset: "
            f"{layout.dataset_manifest}"
        )
    return {
        "status": "complete",
        "dataset_id": layout.dataset_id,
        "warp_project": str(layout.training_dir),
        "resolved_warp_project": str(project),
        "xml": str(xmls[0]),
        "conversion_validation": str(validation),
        "dataset_manifest": str(layout.dataset_manifest),
        "marker": str(marker),
    }


def _synchronous_warp_import(
    layout: RunLayout,
    cfg: dict,
    staging_manifest: Path,
    *,
    force: bool = False,
) -> dict:
    """Run IMOD→Warp import now, without Slurm, using the cluster Warp module.

    The generated import batch remains available as a recovery/re-import path.
    Conversion itself is idempotent and is skipped by ``run_warp_conversion.py``
    when the current conversion contract is already present.
    """
    if not force:
        try:
            return {"execution": "reused", **_validate_imported_warp_dataset(layout)}
        except RuntimeError:
            pass

    if not staging_manifest.is_file() or staging_manifest.stat().st_size <= 0:
        raise RuntimeError(
            "synchronous Warp import requires the staged conversion manifest, but it "
            f"is missing: {staging_manifest}"
        )

    cluster_cfg = cfg.get("cluster", {}) or {}
    module_init = str(
        cluster_cfg.get("module_init_script")
        or "/usr/share/Modules/init/bash"
    )
    warp_module = str(cluster_cfg.get("warp_module") or "").strip()
    environment = str(cluster_cfg.get("environment") or "").strip()
    # When no explicit [cluster].environment is configured, the scientific Python is the
    # one the modules put on PATH (e.g. CSSB's `missalign` module ships its own python with
    # warpylib) — NOT this launcher's sys.executable, which is the base env without warpylib.
    # A bare "python" is resolved with `command -v` in the shell after the module loads.
    python_exe = (
        str(Path(environment).expanduser() / "bin" / "python")
        if environment
        else "python"
    )
    conversion_script = Path(__file__).resolve().parents[1] / "run_warp_conversion.py"

    module_lines = []
    if warp_module:
        module_lines.extend([
            f"if [[ -r {shlex.quote(module_init)} ]]; then",
            f"  source {shlex.quote(module_init)}",
            "elif [[ -r /etc/profile.d/modules.sh ]]; then",
            "  source /etc/profile.d/modules.sh",
            "else",
            "  echo 'ERROR: Environment Modules initialisation not found' >&2",
            "  exit 2",
            "fi",
            "module purge",
            f"module load {shlex.quote(warp_module)}",
        ])
        # warpylib + MissAlignment live in separate modules (CSSB: cssb/rarely + missalign);
        # the warp module only ships the WarpTools binaries. [cluster].missalign_modules
        # overrides (set [] to disable).
        missalign_modules = cluster_cfg.get("missalign_modules")
        if missalign_modules is None:
            missalign_modules = ["cssb/rarely", "missalign"]
        module_lines.extend(f"module load {shlex.quote(m)}" for m in missalign_modules)

    force_arg = " --force" if force else ""
    shell = "\n".join([
        "set -euo pipefail",
        "export PATH=\"/usr/local/bin:/usr/bin:/bin:${PATH:-}\"",
        *module_lines,
        "export LC_ALL=C",
        "export LANG=C",
        f"PIPELINE_PYTHON={shlex.quote(python_exe)}",
        'if [[ "$PIPELINE_PYTHON" != */* ]]; then PIPELINE_PYTHON="$(command -v "$PIPELINE_PYTHON" 2>/dev/null || true)"; fi',
        '[[ -n "$PIPELINE_PYTHON" && -x "$PIPELINE_PYTHON" ]] || { echo "ERROR: Python not found/executable: $PIPELINE_PYTHON" >&2; exit 2; }',
        "\"$PIPELINE_PYTHON\" - <<'PY_WARP_IMPORT'",
        "import mrcfile",
        "import numpy",
        "import warpylib",
        "print('Warp import Python ready')",
        "PY_WARP_IMPORT",
        f'"$PIPELINE_PYTHON" {shlex.quote(str(conversion_script))} '
        f'--staging-manifest {shlex.quote(str(staging_manifest))} '
        f'--training-dir {shlex.quote(str(layout.training_dir))}{force_arg}',
    ])

    print(f"[prepare] importing IMOD into Warp dataset {layout.dataset_id} (synchronous)", flush=True)
    completed = subprocess.run(
        ["bash", "-lc", shell],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=os.environ.copy(),
        check=False,
    )
    if completed.returncode != 0:
        fallback = layout.batch_path("import", "import_imod_to_warp.sbatch")
        raise RuntimeError(
            f"synchronous Warp import failed with exit code {completed.returncode}. "
            f"The recovery batch remains available: sbatch {fallback}"
        )
    return {"execution": "synchronous", **_validate_imported_warp_dataset(layout)}


def _resolve(cfg: dict, args) -> dict:
    """Resolve basename/condition/mode/out_dir/data_dir from config + CLI."""
    basename = (getattr(args, "basename", None) or cfg.get("project", {}).get("basename")
                or cfg.get("project", {}).get("name") or "series")
    conds = (getattr(args, "condition", None) or
             cfg.get("conversion", {}).get("initial_conditions", ["ali_identity"]))
    condition = conds[0] if isinstance(conds, list) else conds
    mode = (getattr(args, "refinement_mode", None)
            or cfg.get("missalignment", {}).get("refinement_mode", "standard"))
    out_dir = Path(getattr(args, "out_dir", None) or cfg.get("paths", {}).get("output_dir") or ".")
    data_dir = getattr(args, "data_dir", None) or cfg.get("paths", {}).get("data_root")
    dataset_id = getattr(args, "dataset", None) or dataset_id_from_config(cfg)
    return {"basename": basename, "condition": condition, "mode": mode,
            "dataset_id": dataset_id, "out_dir": out_dir,
            "data_dir": Path(data_dir) if data_dir else None}


def _inventory(cfg: dict, data_dir, basename: str) -> DISC.SourceInventory:
    """Return the exact resolved source inventory, or discover only for legacy input.

    A resolved TOML is consume-only: no globbing, basename guessing or candidate
    selection is repeated during prepare.
    """
    inp = cfg.get("input", {}) or {}
    ctf = cfg.get("ctf", {}) or {}
    resolved = bool((cfg.get("provenance", {}) or {}).get("resolved"))
    if resolved:
        inv = DISC.SourceInventory(basename=basename, data_dir=str(data_dir or ""))
        mapping = {
            "raw_stack": inp.get("raw_stack"),
            "aligned_stack": inp.get("aligned_stack"),
            "final_xf": inp.get("final_xf_file"),
            "tilt_file": inp.get("final_tilt_file"),
            "raw_tilt_file": inp.get("raw_tilt_file"),
            "xtilt_file": inp.get("xtilt_file"),
            "tltxf_file": inp.get("tltxf_file"),
            "defocus_file": inp.get("defocus_file") or ctf.get("defocus_file"),
            "mdoc_file": inp.get("mdoc_file"),
            "newst_com": inp.get("newst_com"),
            "tilt_com": inp.get("tilt_com"),
            "ctf_com": inp.get("ctf_com") or ctf.get("command_file"),
            "source_reconstruction": inp.get("source_reconstruction"),
        }
        for key, value in mapping.items():
            setattr(inv, key, value or None)
        inv.report = {"selection": "resolved_toml", "rediscovery": False}
        return inv

    overrides = {
        "raw_stack": inp.get("raw_stack"), "aligned_stack": inp.get("aligned_stack"),
        "final_xf": inp.get("final_xf_file"), "tilt_file": inp.get("final_tilt_file"),
        "defocus_file": ctf.get("defocus_file"), "ctf_com": ctf.get("command_file"),
    }
    overrides = {k: v for k, v in overrides.items() if v}
    if not data_dir or not Path(data_dir).is_dir():
        inv = DISC.SourceInventory(basename=basename, data_dir=str(data_dir or ""))
        for k, v in overrides.items():
            setattr(inv, k, v)
        inv.report = {"note": "no data_dir; explicit overrides only"}
        return inv
    return DISC.discover_sources(Path(data_dir), basename, overrides=overrides)


def _source_hashes(inv: DISC.SourceInventory) -> dict:
    """Hash small source files fully; large stacks get a documented partial hash."""
    out = {}
    for fieldname in ("raw_stack", "aligned_stack", "final_xf", "tilt_file",
                      "raw_tilt_file", "xtilt_file", "defocus_file", "mdoc_file",
                      "newst_com", "tilt_com", "ctf_com"):
        p = getattr(inv, fieldname, None)
        if not p or not Path(p).is_file():
            continue
        path = Path(p)
        size = path.stat().st_size
        h = hashlib.sha256()
        if size <= 64 * 1024 * 1024:
            with path.open("rb") as fh:
                for c in iter(lambda: fh.read(1 << 20), b""):
                    h.update(c)
            mode = "full_sha256"
        else:
            # partial: first+last 8 MiB + size (documented strategy for big stacks)
            with path.open("rb") as fh:
                h.update(fh.read(8 << 20))
                fh.seek(-(8 << 20), 2)
                h.update(fh.read(8 << 20))
            h.update(str(size).encode())
            mode = "partial_sha256_head8M_tail8M_size"
        out[fieldname] = {"path": str(path), "size": size, "sha256": h.hexdigest()[:32], "mode": mode}
    return out


def _sha256_file(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _git_info(repo_root: Path) -> dict:
    def run(args):
        cp = subprocess.run(args, cwd=repo_root, text=True, capture_output=True)
        return cp.stdout.strip() if cp.returncode == 0 else None
    status = run(["git", "status", "--porcelain"])
    return {"commit": run(["git", "rev-parse", "HEAD"]), "dirty": bool(status),
            "status_porcelain": status}


def _write_code_provenance(settings_path: Path, layout: RunLayout, written: dict) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    sources = {
        "setup_missalign_project.py": repo_root / "setup_missalign_project.py",
        "scripts/pipeline/imod_reconstruction.py": repo_root / "scripts" / "pipeline" / "imod_reconstruction.py",
        "scripts/pipeline/warptools_reconstruction.py": repo_root / "scripts" / "pipeline" / "warptools_reconstruction.py",
        "scripts/pipeline/pre_conversion_reconstruction.py": repo_root / "scripts" / "pipeline" / "pre_conversion_reconstruction.py",
        "scripts/geometry/quarter_turn.py": repo_root / "scripts" / "geometry" / "quarter_turn.py",
        "scripts/pipeline/accept_pre_conversion.py": repo_root / "scripts" / "pipeline" / "accept_pre_conversion.py",
        "prepare_missalignment_input.py": repo_root / "prepare_missalignment_input.py",
        "scripts/pipeline/missalignment_input.py": repo_root / "scripts" / "pipeline" / "missalignment_input.py",
        "scripts/pipeline/dataset_selection.py": repo_root / "scripts" / "pipeline" / "dataset_selection.py",
        "scripts/pipeline/reconstruction_validation.py": repo_root / "scripts" / "pipeline" / "reconstruction_validation.py",
        "scripts/clone_warp_projects.py": repo_root / "scripts" / "clone_warp_projects.py",
        "tools/audit_quarter_turn.py": repo_root / "tools" / "audit_quarter_turn.py",
        "scripts/etomo_to_warp.py": repo_root / "scripts" / "etomo_to_warp.py",
        "scripts/pipeline/jobs.py": repo_root / "scripts" / "pipeline" / "jobs.py",
        "scripts/export_condition_results.py": repo_root / "scripts" / "export_condition_results.py",
        "scripts/warp_to_imod_affine.py": repo_root / "scripts" / "warp_to_imod_affine.py",
    }
    data = {
        "repository_root": str(repo_root),
        "software_version": (repo_root / "VERSION").read_text().strip() if (repo_root / "VERSION").is_file() else "unknown",
        "git": _git_info(repo_root),
        "project_settings": {"path": str(settings_path), "sha256": _sha256_file(settings_path)},
        "source_files": {k: {"path": str(v), "sha256": _sha256_file(v)} for k, v in sources.items()},
        "generated_jobs": {
            k: {"path": v, "sha256": _sha256_file(Path(v))}
            for k, v in sorted(written.items()) if str(k).endswith(".sbatch")
        },
        "stale_code_policy": {
            "reconstruction_checks_executor_hash": True,
            "warptools_checks_executor_hash": True,
            "reconstruction_checks_project_settings_hash": True,
            "override": "--allow-provenance-mismatch",
        },
    }
    _atomic(layout.manifest("code_provenance.json"), data)
    return data


# --------------------------------------------------------------------------- #
# verbs
# --------------------------------------------------------------------------- #
def cmd_validate(cfg: dict, args) -> int:
    r = _resolve(cfg, args)
    print(f"[validate] basename={r['basename']} condition={r['condition']} mode={r['mode']}")
    # discovery (deterministic; fails on ambiguity)
    try:
        inv = _inventory(cfg, r["data_dir"], r["basename"])
    except DISC.DiscoveryError as exc:
        print(f"ERROR: discovery failed: {exc}")
        return 2
    missing = [k for k in ("aligned_stack", "raw_stack") if not getattr(inv, k)]
    if missing and r["data_dir"]:
        print(f"WARNING: could not resolve {missing} (set them in [input])")
    # capabilities (local snapshot; cluster items are expected unavailable here)
    caps = CAP.probe_all()
    print("[validate] capabilities:")
    for k, v in caps.items():
        print(f"    {k:22s} {v['state']}")
    # warp conversion feasibility
    wc = CAP.can_convert_warp()
    print(f"[validate] warp conversion locally: {wc.state} "
          f"({'will run in prepare' if wc.available else 'deferred to cluster preflight'})")
    # refinement-mode availability (2.15): clear status for rigid/similarity
    from . import project_config as PC
    fork = PC.constrained_fork_status()
    if r["mode"] in PC.STOCK_REFINEMENT_MODES:
        print(f"[validate] refinement_mode {r['mode']!r}: stock path (available)")
    else:
        print(f"[validate] refinement_mode {r['mode']!r}: constrained fork "
              f"{'available' if fork['available'] else 'UNAVAILABLE'} ({fork['reason']})")
        if not fork["available"]:
            print(f"[validate] WARNING: {r['mode']!r} will be refused at prepare unless the fork is "
                  "installed (cluster_integration/missalignment_patch) or --allow-unavailable-mode is set.")
    print("[validate] configuration OK")
    return 0


def cmd_prepare(cfg: dict, args, *, orchestrate_fn, inputs_builder) -> int:
    r = _resolve(cfg, args)
    toml_binning = int((cfg.get("multiresolution", {}) or {}).get("extra_projection_binning", 1))
    cli_binning = getattr(args, "extra_binning", None)
    if cli_binning is not None and int(cli_binning) != toml_binning:
        print(
            "ERROR: --extra-binning is a compatibility option and must match "
            f"[multiresolution].extra_projection_binning ({toml_binning}); got {cli_binning}. "
            "Edit the resolved TOML instead."
        )
        return 2
    # consume-only (§5): prepare must consume the RESOLVED TOML written by `init`. A
    # HARD FAILURE on unresolved input — no silent discovery fallback in production.
    resolved = bool((cfg.get("provenance", {}) or {}).get("resolved"))
    if not resolved and not getattr(args, "allow_unresolved_legacy", False):
        print("ERROR: unresolved config; run setup_missalign_project.py init <SETTINGS> first and "
              "pass the generated project_settings.toml. (Override only for migration with "
              "--allow-unresolved-legacy.)")
        return 2
    # 2.15: refuse rigid/similarity BEFORE generating/submitting jobs unless the fork
    # is available (or explicitly overridden). Stock modes pass through.
    from . import project_config as PC
    try:
        PC.assert_refinement_mode_available(
            r["mode"], allow_override=getattr(args, "allow_unavailable_mode", False))
    except PC.ModeUnavailableError as exc:
        print(f"ERROR: {exc}")
        return 2
    layout = RunLayout.from_settings(
        out_dir=r["out_dir"], basename=r["basename"], condition=r["condition"],
        refinement_mode=r["mode"], dataset_id=r["dataset_id"]
    ).create()
    run_id = _run_id(layout)
    rl = RL.RunLogger(layout.run_dir, run_id=run_id, phase="prepare")
    rl.write_environment(name="prepare")
    rl.log_event(step="P00", event="prepare_start", status="info",
                 message=f"run_dir={layout.run_dir}", data=layout.to_dict())
    try:
        # P01-P02 discovery + inventory + hashes
        inv = _inventory(cfg, r["data_dir"], r["basename"])
        _atomic(layout.manifest("source_inventory.json"), inv.to_dict())
        hashes = _source_hashes(inv)
        _atomic(layout.manifest("source_hashes.json"), hashes)
        rl.log_event(step="P01", event="discover_source", status="ok",
                     data={"selected": {k: getattr(inv, k) for k in
                           ("raw_stack", "aligned_stack", "final_xf", "tilt_file", "ctf_com")}})
        from .project_publish import publish_imod_import
        imported_imod_manifest = publish_imod_import(layout, inv)
        rl.log_event(step="P02", event="publish_imod_import", status="ok",
                     data={"manifest": str(imported_imod_manifest)})
        _atomic(layout.provenance_dir / "coordinate_frames.json", {
            "schema_version": 1,
            "frames": {
                "imod_source_detector": {
                    "shape_xyz": (cfg.get("geometry", {}) or {}).get("raw_shape_xyz"),
                    "pixel_size_A": (cfg.get("geometry", {}) or {}).get("raw_pixel_size_A"),
                },
                "imod_reconstruction_mrc": {
                    "shape_xyz": (cfg.get("geometry", {}) or {}).get("target_volume_shape_xyz"),
                    "axis_contract": "X,Y(thickness),Z(detector-vertical)",
                },
                "warp_reconstruction": {
                    "axis_contract": "X,Y(detector-vertical),Z(thickness)",
                    "mapping_from_imod_mrc": "X,Z,Y",
                },
            },
            "detector_quarter_turn_scope": "detector frame only",
        })
        _atomic(layout.provenance_dir / "artifact_registry.json", {
            "schema_version": 1,
            "artifacts": {
                "imod_import": {
                    "artifact_type": "imported_imod_project",
                    "manifest": str(imported_imod_manifest),
                    "status": "complete",
                },
                layout.dataset_id: {
                    "artifact_type": "warp_tilt_series_dataset",
                    "dataset_id": layout.dataset_id,
                    "manifest": str(layout.dataset_manifest),
                    "status": "planned",
                },
            },
        })
        _atomic(layout.provenance_dir / "software_versions.json", {
            "schema_version": 1,
            "pipeline_version": (Path(__file__).resolve().parents[2] / "VERSION").read_text().strip(),
            "cluster_runtime": "recorded by generated batch environment reports",
        })

        # Heavy IMOD/Warp preparation runs in the private compatibility workspace.
        inputs = inputs_builder(cfg, inv, r)
        oargs = _orch_args(args, r)
        result = orchestrate_fn(config=cfg, out_dir=layout.internal_runtime_dir, data_dir=r["data_dir"],
                                basename=r["basename"], inputs=inputs, args=oargs)
        rl.log_event(step="P14", event="orchestrate", status="ok",
                     data={"steps_run": result.steps_run, "steps_skipped": result.steps_skipped,
                           "manifest": str(result.manifest_path)})
        for w in result.warnings:
            rl.log_event(step="P14", event="warning", status="warn", message=w)

        # P15-P17 generate cluster jobs from the resolved MissAlignment settings.
        # Smoke and full use independent Warp snapshots created synchronously after reconstruction validation.
        man = json.loads(Path(result.manifest_path).read_text())
        ma = man.get("missalignment", {})
        run_script = ma.get("run_script", str(layout.config_dir / "run_missalignment.sh"))
        smoke_cmd = ""
        ma_cmd = ""
        try:
            from .orchestrate import _load_03run
            r3 = _load_03run()
            configs = layout.config_yaml.parent
            configs.mkdir(parents=True, exist_ok=True)
            smoke_dir = layout.smoke_warp_dir
            full_dir = layout.full_warp_dir
            smoke_yaml = configs / "config.smoke.yaml"
            full_yaml = layout.config_yaml
            smoke_yaml.write_text(r3.config_text(smoke_dir, "smoke"))
            full_yaml.write_text(r3.config_text(full_dir, r["mode"]))
            ma_cfg = cfg.get("missalignment", {})
            executable = str(ma_cfg.get("executable", "miss-alignment"))
            train_dev = str(ma_cfg.get("training_devices", "0"))
            recon_dev = str(ma_cfg.get("reconstruction_devices", "0"))
            dl_per = int(ma_cfg.get("dataloaders_per_trainer", 1))
            smoke_cmd = r3.shell_quote(r3.missalignment_command(
                config_path=smoke_yaml, training_devices=train_dev,
                reconstruction_devices=recon_dev, dataloaders_per_trainer=dl_per,
                prepare_stacks=None, start_at_iteration=0, executable=executable))
            ma_cmd = r3.shell_quote(r3.missalignment_command(
                config_path=full_yaml, training_devices=train_dev,
                reconstruction_devices=recon_dev, dataloaders_per_trainer=dl_per,
                prepare_stacks=None, start_at_iteration=0, executable=executable))
        except Exception as exc:
            raise RuntimeError(f"failed to generate isolated MissAlignment configs: {exc}") from exc
        # §9: surface the Warp staging manifest at the canonical layout path so the
        # import job can populate the training dir before MissAlignment.
        warp_state = man.get("warp") or {}
        staging_src = warp_state.get("staging_manifest")
        staging_dst = layout.manifest("warp_staging_manifest.json")
        if staging_src and Path(staging_src).is_file():
            staging_data = json.loads(Path(staging_src).read_text())
            staging_data.update({
                "training_dir": str(layout.training_dir),
                "project_root": str(layout.run_dir),
                "dataset_id": layout.dataset_id,
                "refinement_mode": r["mode"],
                "layout_version": 8,
            })
            _atomic(staging_dst, staging_data)
        if layout.internal_warp_project.joinpath("_converted.marker").is_file():
            from .project_publish import publish_warp_dataset
            publish_warp_dataset(layout)
        input_record = {
            "schema_version": 1,
            "dataset_id": layout.dataset_id,
            "dataset_manifest": str(layout.dataset_manifest),
            "warp_project": str(layout.training_dir),
            "selection_policy": "explicit dataset identifier",
        }
        _atomic(layout.missalignment_run_dir / "input" / "selected_dataset.json", input_record)
        # Cluster settings from the canonical config
        from . import project_config as PC
        cluster_cfg = PC.from_dict(cfg).cluster
        reconstruction_job_cfg = dict(cfg.get("reconstruction", {}) or {})
        reconstruction_job_cfg["cluster"] = (
            (cfg.get("cluster", {}) or {}).get("reconstruction_cluster") or {}
        )
        reconstruction_job_cfg["warptools_cluster"] = (
            (cfg.get("cluster", {}) or {}).get("warptools_reconstruction_cluster") or {}
        )
        written = JOBS.generate_jobs(
            layout, profile=cfg.get("cluster", {}).get("profile", "maxwell"),
            ma_command=ma_cmd, smoke_command=smoke_cmd, run_script=run_script, cluster=cluster_cfg,
            settings_path=str(getattr(args, "settings", "SETTINGS.toml")),
            working_recon=bool(cfg.get("reconstruction", {}).get("working", {}).get("enabled", False)),
            halfmaps=bool(cfg.get("reconstruction", {}).get("final", {}).get("halfmaps", False)),
            warp_staging_manifest=str(staging_dst) if staging_dst.is_file() else "",
            reconstruction_config=reconstruction_job_cfg)
        code_provenance = _write_code_provenance(Path(getattr(args, "settings", "SETTINGS.toml")), layout, written)
        rl.log_event(step="P16", event="generate_jobs", status="ok", data={"jobs": list(written)})

        # Import is part of setup. The generated Slurm batch remains a recovery path,
        # but a successfully prepared v8 project must already contain a validated Warp
        # dataset and be immediately ready for reconstruction.
        warp_import = _synchronous_warp_import(
            layout,
            cfg,
            staging_dst,
            force=bool(getattr(args, "force_warp_import", False)),
        )
        rl.log_event(step="P17", event="import_imod_to_warp", status="ok", data=warp_import)

        # job graph + prepare manifest
        _atomic(layout.manifest("job_graph.json"), _job_graph(layout, written))
        prep = {
            "run_id": run_id, "run_dir": str(layout.run_dir), "layout": layout.to_dict(),
            "source_inventory": str(layout.manifest("source_inventory.json")),
            "source_hashes": str(layout.manifest("source_hashes.json")),
            "orchestrate_manifest": str(result.manifest_path),
            "steps_run": result.steps_run, "steps_skipped": result.steps_skipped,
            "warnings": result.warnings, "jobs": written,
            "warp": man.get("warp"), "ctf_mode": man.get("ctf_mode"),
            "consumed_resolved_toml": resolved,
            "final_ctf_deferred_to_export": man.get("final_ctf_deferred_to_export"),
            "capabilities": CAP.probe_all(),
            "code_provenance": str(layout.manifest("code_provenance.json")),
            "warp_import": warp_import,
            "reimport_batch": str(layout.batch_path("import", "import_imod_to_warp.sbatch")),
            "prepare_missalignment_input_command": (
                f"python {Path(__file__).resolve().parents[2] / 'prepare_missalignment_input.py'} "
                f"--directory {layout.run_dir}"
            ),
            "next": [f"sbatch {layout.batch_path('warp_data', 'reconstruct.sbatch')}"],
        }
        _atomic(layout.manifest("prepare_manifest.json"), prep)
        # §13: deterministic result manifest. backend/condition/training_directory are known
        # NOW and recorded once; the XML fields are filled by MissAlignment completion or an EXPLICIT
        # --xml at finalize (never selected by mtime, 2.16). training_directory is the single
        # canonical path (§6) that the cluster jobs and finalize also consume.
        _atomic(layout.manifest("result_manifest.json"), {
            "schema_version": 1, "result_backend": PC.from_dict(cfg).result_backend,
            "condition": r["condition"], "refinement_mode": r["mode"],
            "dataset_id": layout.dataset_id,
            "training_directory": str(layout.full_warp_dir),
            "pre_missalign_directory": str(layout.pre_missalign_dir),
            "smoke_directory": str(layout.smoke_warp_dir),
            "initial_xml": None, "final_xml": None, "final_iteration": None,
            "prepare_manifest": str(layout.manifest("prepare_manifest.json")),
            "orchestrate_manifest": str(result.manifest_path)})
        current_status = {}
        status_path = layout.run_dir / "project_status.json"
        if status_path.is_file():
            current_status = json.loads(status_path.read_text())
        datasets = dict(current_status.get("datasets") or {})
        datasets.setdefault(layout.dataset_id, {
            "status": "planned",
            "manifest": str(layout.dataset_manifest),
            "pixel_size_A": (cfg.get("datasets", {}) or {}).get("native_pixel_size_A"),
        })
        _atomic(status_path, {
            "schema_version": 1,
            "layout_version": 8,
            "status": "ready_for_reconstruction",
            "native_dataset_id": layout.dataset_id,
            "selected_dataset_id": current_status.get("selected_dataset_id") or layout.dataset_id,
            "condition": layout.condition,
            "datasets": datasets,
            "next": [str(layout.batch_path("warp_data", "reconstruct.sbatch"))],
            "reimport_batch": str(layout.batch_path("import", "import_imod_to_warp.sbatch")),
            "prepare_missalignment_input": {
                "status": "waiting_for_reconstruction_acceptance",
                "command": (
                    f"python {Path(__file__).resolve().parents[2] / 'prepare_missalignment_input.py'} "
                    f"--directory {layout.run_dir}"
                ),
            },
        })
        rl.log_event(step="P18", event="prepare_done", status="ok",
                     message=str(layout.manifest("prepare_manifest.json")))
        print(f"[prepare] project: {layout.run_dir}")
        print(f"[prepare] Warp import: {warp_import['execution']} ({warp_import['xml']})")
        print("[prepare] project is ready; reconstruct the imported Warp dataset:")
        print(f"  sbatch {layout.batch_path('warp_data', 'reconstruct.sbatch')}")
        print("[prepare] after reconstruction inspection and acceptance, prepare MissAlignment input synchronously:")
        print(
            f"  python {Path(__file__).resolve().parents[2] / 'prepare_missalignment_input.py'} "
            f"--directory {layout.run_dir}"
        )
        print("[prepare] recovery/re-import batch (normally not needed):")
        print(f"  sbatch {layout.batch_path('import', 'import_imod_to_warp.sbatch')}")
        return 0
    except Exception as exc:
        rl.write_postmortem(exc, step=rl._last_step,
                            input_files=[Path(p) for p in
                                         (inv.raw_stack, inv.aligned_stack) if p] if 'inv' in dir() else [])
        print(f"ERROR: prepare failed: {exc}")
        print(f"       postmortem: {layout.diagnostics_dir}/postmortem/failure.json")
        print(f"       collect a bundle: prepare_imod_to_warp.py collect-debug <settings>")
        return 1


def cmd_status(cfg: dict, args) -> int:
    r = _resolve(cfg, args)
    layout = RunLayout.from_settings(
        out_dir=r["out_dir"], basename=r["basename"], condition=r["condition"],
        refinement_mode=r["mode"], dataset_id=r["dataset_id"]
    )
    rd = layout.run_dir
    if not rd.exists():
        print(f"[status] no run dir yet at {rd} (run 'prepare' first)")
        return 0
    print(f"[status] run_dir: {rd}")
    for name in ("prepare_manifest.json", "missalignment_run_manifest.json", "result_manifest.json",
                 "finalize_manifest.json", "final_validation.json"):
        p = layout.manifest(name)
        print(f"    {name:28s} {'present' if p.exists() else '-'}")
    for snapshot, rel in (
        ("before", str(layout.results_dir / "reconstructions" / "before")),
        ("smoke", str(layout.results_dir / "reconstructions" / "smoke")),
        ("final", str(layout.final_reconstruction)),
    ):
        target = Path(rel)
        found = list(target.glob("*/reconstruction_manifest.json")) if target.exists() else []
        print(f"    reconstruction:{snapshot:14s} {'present' if found else '-'}")
    # phase verdicts
    snapshot_manifest = layout.manifest("warp_snapshot_manifest.json")
    print(f"    missalignment_input         {'prepared' if snapshot_manifest.exists() else '-'}")
    smoke = layout.results_dir / "smoke_verdict.json"
    print(f"    smoke_verdict               {'present' if smoke.exists() else '-'}")
    pm = layout.diagnostics_dir / "postmortem" / "failure.json"
    if pm.exists():
        fail = json.loads(pm.read_text())
        print(f"    LAST FAILURE: {fail.get('phase')}/{fail.get('step')}: {fail.get('exception_message')}")
    ev = rd / "logs" / "events.jsonl"
    if ev.exists():
        lines = ev.read_text().splitlines()
        print(f"    events: {len(lines)} (last: {json.loads(lines[-1])['event'] if lines else '-'})")
    return 0


def cmd_collect_debug(cfg: dict, args) -> int:
    r = _resolve(cfg, args)
    layout = RunLayout.from_settings(
        out_dir=r["out_dir"], basename=r["basename"], condition=r["condition"],
        refinement_mode=r["mode"], dataset_id=r["dataset_id"]
    )
    if not layout.run_dir.exists():
        print(f"ERROR: no run dir at {layout.run_dir}")
        return 2
    run_id = _run_id(layout)
    bundle = RL.collect_debug_bundle(layout.run_dir, run_id,
                                     include_checkpoints=getattr(args, "include_checkpoints", False))
    print(f"[collect-debug] wrote {bundle}")
    return 0


def cmd_regenerate_jobs(cfg: dict, args) -> int:
    from . import project_config as PC
    r = _resolve(cfg, args)
    layout = RunLayout.from_settings(
        out_dir=r["out_dir"], basename=r["basename"], condition=r["condition"],
        refinement_mode=r["mode"], dataset_id=r["dataset_id"]
    ).create()
    prepare_manifest = layout.manifest("prepare_manifest.json")
    if not prepare_manifest.is_file():
        print(f"ERROR: no prepare manifest at {prepare_manifest}; run prepare first")
        return 2
    backup = layout.jobs_dir / "backups" / ("regen_" + hashlib.sha256(str(layout.jobs_dir).encode()).hexdigest()[:12])
    backup.mkdir(parents=True, exist_ok=True)
    for path in layout.jobs_dir.rglob("*.sbatch"):
        target = backup / path.relative_to(layout.jobs_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path.read_text())
        target.chmod(path.stat().st_mode)
    cluster_cfg = PC.from_dict(cfg).cluster
    reconstruction_job_cfg = dict(cfg.get("reconstruction", {}) or {})
    reconstruction_job_cfg["cluster"] = ((cfg.get("cluster", {}) or {}).get("reconstruction_cluster") or {})
    reconstruction_job_cfg["warptools_cluster"] = ((cfg.get("cluster", {}) or {}).get("warptools_reconstruction_cluster") or {})
    ma = cfg.get("missalignment", {}) or {}
    executable = str(ma.get("executable", "miss-alignment"))
    full_yaml = layout.config_yaml
    smoke_yaml = layout.config_yaml.parent / "config.smoke.yaml"
    written = JOBS.generate_jobs(
        layout, profile=cfg.get("cluster", {}).get("profile", "maxwell"),
        ma_command=f"{executable} --config-file {full_yaml}",
        smoke_command=f"{executable} --config-file {smoke_yaml}",
        run_script=str(layout.config_dir / "run_missalignment.sh"),
        settings_path=str(args.settings),
        cluster=cluster_cfg,
        warp_staging_manifest=str(layout.manifest("warp_staging_manifest.json")),
        reconstruction_config=reconstruction_job_cfg,
    )
    prov = _write_code_provenance(Path(args.settings), layout, written)
    print(f"[regenerate-jobs] backed up previous sbatch files: {backup}")
    print(f"[regenerate-jobs] code provenance: {layout.manifest('code_provenance.json')}")
    print(f"[regenerate-jobs] project_settings sha256: {prov['project_settings']['sha256']}")
    return 0


# --------------------------------------------------------------------------- #
def _orch_args(args, r) -> SimpleNamespace:
    """Map subcommand args onto the orchestrate() args namespace (back-compat)."""
    return SimpleNamespace(
        extra_binning=getattr(args, "extra_binning", None),
        ctf_mode=getattr(args, "ctf_mode", None),
        condition=[r["condition"]],
        refinement_mode=r["mode"],
        working_reconstruction=getattr(args, "working_reconstruction", False),
        generate_slurm=getattr(args, "generate_slurm", True),
        cluster_profile=getattr(args, "cluster_profile", None),
        submit=False,  # submission is a cluster step, never from prepare
        resume=getattr(args, "resume", False),
        from_step=getattr(args, "from_step", None),
        only_step=getattr(args, "only_step", None),
        force_geometry=getattr(args, "force_geometry", False),
        # The v8 setup always performs the public Warp import after orchestration
        # through _synchronous_warp_import(). Keep the legacy in-process path off so
        # there is one conversion entry point and one validation contract.
        local_warp_convert=False,
    )


def _job_graph(layout: RunLayout, written: dict) -> dict:
    dataset = layout.dataset_id
    return {
        "project_root": str(layout.run_dir),
        "dataset_id": dataset,
        "operations": {
            "import_imod_to_warp": {
                "execution": "synchronous_during_setup",
                "depends_on": [],
                "recovery_job": written.get("import/import_imod_to_warp.sbatch"),
            },
            "reconstruct_imported_imod": {
                "depends_on": [],
                "job": written.get("import/reconstruct_imported_imod.sbatch"),
            },
            "reconstruct_warp_dataset": {
                "depends_on": ["import_imod_to_warp"],
                "job": written.get(f"warp_data/{dataset}/reconstruct.sbatch"),
                "next_step_policy": "automatic technical validation on successful reconstruction",
            },
            "prepare_missalignment_input": {
                "execution": "synchronous_local_command",
                "depends_on": ["reconstruct_warp_dataset"],
                "command": (
                    f"python {Path(__file__).resolve().parents[2] / 'prepare_missalignment_input.py'} "
                    f"--directory {layout.run_dir}"
                ),
                "recovery_job": written.get(f"missalignment/{dataset}/prepare_input.sbatch"),
            },
            "run_smoke": {
                "depends_on": ["prepare_missalignment_input"],
                "job": written.get(f"missalignment/{dataset}/run_smoke.sbatch"),
            },
            "run_full": {
                "depends_on": ["run_smoke"],
                "job": written.get(f"missalignment/{dataset}/run_full.sbatch"),
            },
            "export_imod": {
                "depends_on": ["run_full"],
                "job": written.get(f"export/{dataset}/export_imod_and_reconstruct.sbatch"),
            },
        },
    }


def _atomic(path: Path, obj) -> None:
    import os
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)
